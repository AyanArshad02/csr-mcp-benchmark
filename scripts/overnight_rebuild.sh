#!/usr/bin/env bash
# scripts/overnight_rebuild.sh — rebuild every number from the fixed code, in order.
#
# Safe to sleep through. Run under caffeinate:
#   caffeinate -dimsu bash scripts/overnight_rebuild.sh 2>&1 | tee /tmp/overnight.log
#
# Order matters: scoring and guardrail must both run AFTER the score.py/reattach.py
# fixes, and the audit sample must be drawn from the final guardrail output.
set -uo pipefail
cd "$(dirname "$0")/.."

PIPELINES="baseline_openai langgraph_openai openai_agent_openai baseline_bedrock langgraph_bedrock"
FAILED=""

step() { echo; echo "############ $* ############"; date "+%H:%M:%S"; }
track() { [ "$1" -ne 0 ] && FAILED="$FAILED $2"; }

step "STEP 0 — unit tests (abort if red)"
uv run pytest guardrail/test_reattach.py measure/test_scorer.py -q || {
    echo "TESTS FAILED — aborting, nothing was overwritten."; exit 1; }

step "STEP 1 — Phase 2 scoring (5 pipelines)"
for p in $PIPELINES; do
    echo "--- scoring $p ---"
    uv run python -m measure.run_scoring "results/$p.jsonl" \
        --out "results/scored_$p.jsonl" --cache-dir .cache
    track $? "score:$p"
done

step "STEP 2 — guardrail passes (5 pipelines)"
for p in $PIPELINES; do
    echo "--- guardrail $p ---"
    uv run python -m guardrail.run_guardrail \
        --input "results/$p.jsonl" \
        --output "results/guardrail_$p.jsonl" --cache-dir .cache
    track $? "guardrail:$p"
done

step "STEP 3 — determinism check (Phase 2 CSR must equal guardrail 'before' CSR)"
uv run python - <<'EOF'
import json
for p in ["baseline_openai","langgraph_openai","openai_agent_openai",
          "baseline_bedrock","langgraph_bedrock"]:
    a = [json.loads(l) for l in open(f"results/scored_{p}.jsonl") if l.strip()]
    b = [json.loads(l) for l in open(f"results/guardrail_{p}.jsonl") if l.strip()]
    av = [r["csr"] for r in a if r.get("csr") is not None]
    bv = [r["orig_csr"] for r in b if r.get("orig_csr") is not None]
    ma, mb = sum(av)/len(av), sum(bv)/len(bv)
    print(f"{p:<24} phase2={ma:.3f}  guardrail_before={mb:.3f}  "
          f"{'OK' if abs(ma-mb) < 0.005 else 'MISMATCH'}")
EOF
track $? "determinism"

step "STEP 4 — bootstrap 95% confidence intervals"
uv run python -m measure.bootstrap_ci
track $? "bootstrap"

step "STEP 5 — guardrail lift decomposition (new attribution vs re-granularization)"
uv run python -m measure.lift_decomposition
track $? "lift_decomposition"

step "STEP 6 — rebuild the 50-sample manual audit set"
uv run python -m measure.build_audit_sample \
    --input results/guardrail_langgraph_openai.jsonl \
    --out   results/audit_sample_50.jsonl
track $? "audit_sample"

step "DONE"
date "+%H:%M:%S"
if [ -n "$FAILED" ]; then
    echo "STEPS THAT FAILED:$FAILED"
else
    echo "All steps completed successfully."
fi
echo "Key outputs: results/confidence_intervals.txt, results/lift_decomposition.txt,"
echo "             results/audit_sample_50.jsonl"
