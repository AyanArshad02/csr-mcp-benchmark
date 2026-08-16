#!/usr/bin/env bash
# scripts/rerun_langgraph_bedrock.sh — regenerate the LangGraph-Bedrock arm after the
# max_tokens fix, then rebuild every downstream number.
#
#   caffeinate -dimsu bash scripts/rerun_langgraph_bedrock.sh 2>&1 | tee /tmp/lgb_rerun.log
#
# Why: langchain-aws silently defaulted ChatBedrock to max_tokens=1024, truncating
# 52/127 answers in this arm while the baseline arm ran at 2048. See NOTES.md.
set -uo pipefail
cd "$(dirname "$0")/.."

FAILED=""
step() { echo; echo "############ $* ############"; date "+%H:%M:%S"; }
track() { [ "$1" -ne 0 ] && FAILED="$FAILED $2"; }

step "STEP 0 — unit tests"
uv run pytest guardrail/test_reattach.py measure/test_scorer.py -q || {
    echo "TESTS FAILED — aborting."; exit 1; }

step "STEP 1 — MCP server reachable?"
curl -s -o /dev/null -w "  /mcp -> %{http_code}\n" http://127.0.0.1:8000/mcp
if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "MCP SERVER NOT LISTENING ON :8000 — aborting before spending Bedrock calls."; exit 1
fi

step "STEP 2 — regenerate langgraph_bedrock (127 live Bedrock calls, ~40 min)"
uv run python -m pipelines.langgraph_agent --provider bedrock \
    --out results/langgraph_bedrock.jsonl
track $? "pipeline:langgraph_bedrock"

step "STEP 3 — truncation check (must be 0; was 52/127 before the fix)"
uv run python - <<'EOF'
import json
CLEAN = (']', '.', '!', '?', ':', '`')
rows = [json.loads(l) for l in open("results/langgraph_bedrock.jsonl") if l.strip()]
bad = [r["question_id"] for r in rows if not r["answer"].rstrip().endswith(CLEAN)]
print(f"  answers not ending cleanly: {len(bad)}/{len(rows)}  {bad[:10]}")
print("  OK" if len(bad) <= 5 else "  STILL TRUNCATED — investigate before trusting results")
EOF

step "STEP 4 — rescore + re-guardrail ALL 5 pipelines (regex fix touches every file)"
for p in baseline_openai langgraph_openai openai_agent_openai baseline_bedrock langgraph_bedrock; do
    echo "--- scoring $p ---"
    uv run python -m measure.run_scoring "results/$p.jsonl" \
        --out "results/scored_$p.jsonl" --cache-dir .cache
    track $? "score:$p"
done
for p in baseline_openai langgraph_openai openai_agent_openai baseline_bedrock langgraph_bedrock; do
    echo "--- guardrail $p ---"
    uv run python -m guardrail.run_guardrail --input "results/$p.jsonl" \
        --output "results/guardrail_$p.jsonl" --cache-dir .cache
    track $? "guardrail:$p"
done

step "STEP 5 — determinism check"
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

step "STEP 6 — bootstrap CIs / lift decomposition / audit sample"
uv run python -m measure.bootstrap_ci;        track $? "bootstrap"
uv run python -m measure.lift_decomposition;  track $? "lift_decomposition"
uv run python -m measure.build_audit_sample \
    --input results/guardrail_langgraph_openai.jsonl \
    --out   results/audit_sample_50.jsonl;    track $? "audit_sample"

step "DONE"
date "+%H:%M:%S"
[ -n "$FAILED" ] && echo "STEPS THAT FAILED:$FAILED" || echo "All steps completed successfully."
