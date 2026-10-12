#!/usr/bin/env bash
# Ticket 16 — run guardrail v2 (tau = 0.9, pre-registered) on all 5 pipeline arms.
# Writes results/guardrail_v2_<arm>.jsonl; one log per arm in results/run_logs/.
# The headline arm (OpenAI Agents SDK) runs first. Stops on the first failure.
# NLI + decomposition are disk-cached (.cache), so an interrupted run resumes cheaply.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y-%m-%d)
for ARM in openai_agent_openai langgraph_openai baseline_openai langgraph_bedrock baseline_bedrock; do
  LOG="results/run_logs/${STAMP}_guardrail_v2_${ARM}.log"
  echo "[$(date +%H:%M:%S)] start ${ARM} -> ${LOG}"
  uv run python -m guardrail.run_guardrail --guardrail v2 \
      --input "results/${ARM}.jsonl" \
      --output "results/guardrail_v2_${ARM}.jsonl" > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] done  ${ARM}"
  sed -n '/GUARDRAIL SUMMARY/,$p' "${LOG}"
done
echo "[$(date +%H:%M:%S)] all arms done"
