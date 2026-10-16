#!/usr/bin/env bash
# Extra scoring worker for scripts/run_repeats_and_v3a.sh: same command and skip rule, for a given
# list of results/repeats/ files. Used to resume and to run a second worker in parallel.
cd "$(dirname "$0")/.."
OUT=results/repeats
for v in "$@"; do
  [ -f "$OUT/scored_${v}_summary.txt" ] && continue   # written last, so a partial file is redone
  HF_HUB_OFFLINE=1 uv run python -m measure.run_scoring "$OUT/$v.jsonl" --out "$OUT/scored_$v.jsonl" > "$OUT/logs/score_$v.log" 2>&1
  echo "[$(date +%H:%M:%S)] scored $v"
done
