#!/usr/bin/env bash
# Ticket 9 / gate G2 — OpenAI Agents SDK mechanism ablation, pre-registered in DECISIONS.md
# (2026-10-12, addendum 2026-10-13). One change per step:
#   E0 original config | E1 +temperature 0 | E2 E1 + Kapa tool description
#   E3 E1 + Chat Completions API | E4 E1 + Kapa tool description + Kapa system prompt
# Writes results/ablation_sdk_<run>.jsonl, scores each with the locked scorer.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y-%m-%d); LOGS=results/run_logs; mkdir -p "$LOGS"

CSR_MCP_PORT=8000 uv run python -m server.main > "$LOGS/${STAMP}_ablation_server_8000.log" 2>&1 & S1=$!
CSR_MCP_PORT=8001 CSR_TOOL_DESCRIPTION=kapa uv run python -m server.main > "$LOGS/${STAMP}_ablation_server_8001.log" 2>&1 & S2=$!
trap 'kill $S1 $S2 2>/dev/null || true' EXIT
sleep 15

run() {  # name, flags...
  local name=$1; shift
  uv run python -m pipelines.openai_agent --out "results/ablation_sdk_${name}.jsonl" "$@" \
      > "$LOGS/${STAMP}_ablation_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] generated ${name}: $(wc -l < results/ablation_sdk_${name}.jsonl) rows, $(grep -c FAILED "$LOGS/${STAMP}_ablation_${name}.log" || true) failed"
}
echo "[$(date +%H:%M:%S)] generating E0-E4 in parallel"
PIDS=()
run e0_original & PIDS+=($!)
run e1_temp0 --temperature 0 & PIDS+=($!)
run e2_kapa_tooldesc --temperature 0 --mcp-url http://127.0.0.1:8001/mcp & PIDS+=($!)
run e3_chat_completions --temperature 0 --api chat_completions & PIDS+=($!)
run e4_kapa_full --temperature 0 --mcp-url http://127.0.0.1:8001/mcp --system-prompt kapa & PIDS+=($!)
wait "${PIDS[@]}"   # only the five runs; the servers never exit on their own
kill $S1 $S2 2>/dev/null || true

# E4: Kapa citation format -> locked [url] format (primary: Kapa brackets only; sensitivity: + inline links)
uv run python -m measure.normalize_kapa_citations results/ablation_sdk_e4_kapa_full.jsonl results/ablation_sdk_e4_kapa_full_norm.jsonl
uv run python -m measure.normalize_kapa_citations results/ablation_sdk_e4_kapa_full.jsonl results/ablation_sdk_e4_kapa_full_norm_inline.jsonl --also-inline-links

for f in e0_original e1_temp0 e2_kapa_tooldesc e3_chat_completions e4_kapa_full_norm e4_kapa_full_norm_inline; do
  echo "[$(date +%H:%M:%S)] scoring ${f}"
  HF_HUB_OFFLINE=1 uv run python -m measure.run_scoring "results/ablation_sdk_${f}.jsonl" > "$LOGS/${STAMP}_ablation_score_${f}.log" 2>&1
done
echo "[$(date +%H:%M:%S)] all done"
