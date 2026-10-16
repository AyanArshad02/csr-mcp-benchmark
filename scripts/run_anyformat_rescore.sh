#!/usr/bin/env bash
# Pre-registered any-format re-score (DECISIONS.md 2026-10-14). Inputs: results/anyformat_<run>.jsonl
# (citations in any of [url] / [text](url) / [[title](url)] form, converted to [url]).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=results/run_logs; STAMP=$(date +%Y-%m-%d)
RUNS="baseline_openai langgraph_openai openai_agent_openai baseline_bedrock langgraph_bedrock ablation_sdk_e0_original ablation_sdk_e1_temp0 ablation_sdk_e2_kapa_tooldesc ablation_sdk_e3_chat_completions ablation_sdk_e4_kapa_full"
for r in $RUNS; do
  echo "[$(date +%H:%M:%S)] DeBERTa scoring anyformat_${r}"
  HF_HUB_OFFLINE=1 uv run python -m measure.run_scoring "results/anyformat_${r}.jsonl" \
      --out "results/scored_anyformat_${r}.jsonl" > "$LOGS/${STAMP}_score_anyformat_${r}.log" 2>&1
done
for r in $RUNS; do
  echo "[$(date +%H:%M:%S)] gpt-4o judge anyformat_${r}"
  uv run python -m measure.rescore_judge --judge gpt-4o --runs "anyformat_${r}" >> "$LOGS/${STAMP}_rescore_gpt4o_anyformat.log" 2>&1
done
uv run python -m measure.anyformat_report
echo "[$(date +%H:%M:%S)] all done"
