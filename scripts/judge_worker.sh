#!/usr/bin/env bash
# Judges each any-format file in results/repeats/ as soon as its DeBERTa scoring has finished
# (the judge reuses the decompositions cached by that scoring). Same command and skip rule as
# scripts/run_repeats_and_v3a.sh.
cd "$(dirname "$0")/.."
OUT=results/repeats
LEFT="$*"
while [ -n "$LEFT" ]; do
  NEXT=""
  for v in $LEFT; do
    [ -f "$OUT/rescored_gpt-4o_$v.jsonl" ] && continue
    if [ -f "$OUT/scored_${v}_summary.txt" ]; then
      uv run python -m measure.rescore_judge --judge gpt-4o --runs "repeats/$v" >> "$OUT/logs/judge_worker.log" 2>&1
      echo "[$(date +%H:%M:%S)] judged $v"
    else NEXT="$NEXT $v"; fi
  done
  LEFT="${NEXT# }"; [ -n "$LEFT" ] && sleep 60
done
echo "[$(date +%H:%M:%S)] judge worker done"
