#!/usr/bin/env bash
# Tickets 15 + 19 (v3a), pre-registered in DECISIONS.md 2026-10-16.
#   Repeats: 6 gpt-4o-mini setups x 2 fresh runs (r2, r3); the published run is r1.
#   v3a:     source IDs at the tool contract, SDK defaults + LangGraph x 3 runs.
# Everything goes to results/repeats/; canonical files are never touched.
# Resumable: a generation step is skipped when its output already has all rows, a scoring step
# when its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/repeats; LOGS=$OUT/logs; mkdir -p "$LOGS"
N=$(wc -l < eval_set/eval.jsonl | tr -d ' ')
say() { echo "[$(date +%H:%M:%S)] $*"; }

CSR_MCP_PORT=8000 uv run python -m server.main > "$LOGS/server_8000.log" 2>&1 & S0=$!
CSR_MCP_PORT=8001 CSR_TOOL_DESCRIPTION=kapa uv run python -m server.main > "$LOGS/server_8001.log" 2>&1 & S1=$!
CSR_MCP_PORT=8002 CSR_SOURCE_IDS=1 uv run python -m server.main > "$LOGS/server_8002.log" 2>&1 & S2=$!
trap 'kill $S0 $S1 $S2 2>/dev/null || true' EXIT
sleep 20

# name | module | flags
JOBS=()
for r in r2 r3; do
  JOBS+=("baseline_$r|baseline|--no-cache")
  JOBS+=("langgraph_$r|langgraph_agent|")
  JOBS+=("sdk_defaults_$r|openai_agent|")
  JOBS+=("sdk_e1_temp0_$r|openai_agent|--temperature 0")
  JOBS+=("sdk_e3_chat_completions_$r|openai_agent|--temperature 0 --api chat_completions")
  JOBS+=("sdk_e4_kapa_full_$r|openai_agent|--temperature 0 --mcp-url http://127.0.0.1:8001/mcp --system-prompt kapa")
done
for r in r1 r2 r3; do
  JOBS+=("v3a_sdk_$r|openai_agent|--mcp-url http://127.0.0.1:8002/mcp --system-prompt ids")
  JOBS+=("v3a_langgraph_$r|langgraph_agent|--mcp-url http://127.0.0.1:8002/mcp --system-prompt ids")
done

gen() {  # one generation job; up to 2 attempts if rows are missing
  IFS='|' read -r name mod flags <<< "$1"
  local f="$OUT/$name.jsonl"
  for attempt in 1 2; do
    [ -f "$f" ] && [ "$(wc -l < "$f" | tr -d ' ')" -ge "$N" ] && return 0
    # shellcheck disable=SC2086
    uv run python -m "pipelines.$mod" --out "$f" $flags > "$LOGS/gen_${name}_a${attempt}.log" 2>&1
  done
  say "WARNING $name: $(wc -l < "$f" | tr -d ' ')/$N rows after 2 attempts"
}

say "generating ${#JOBS[@]} runs, 6 at a time"
for ((i = 0; i < ${#JOBS[@]}; i += 6)); do
  PIDS=()
  for job in "${JOBS[@]:i:6}"; do gen "$job" & PIDS+=($!); done
  wait "${PIDS[@]}"
  say "batch $((i / 6 + 1)) done"
done
kill $S0 $S1 $S2 2>/dev/null || true

# Views: strict (requested format) and any format (DECISIONS.md 2026-10-14 rule)
STRICT=(); ANY=()
for job in "${JOBS[@]}"; do
  name=${job%%|*}; f=$OUT/$name.jsonl
  case $name in
    sdk_e4_kapa_full_*)
      uv run python -m measure.normalize_kapa_citations "$f" "$OUT/${name}_norm.jsonl" > /dev/null
      STRICT+=("${name}_norm") ;;
    v3a_*)
      uv run python -m guardrail.map_source_ids "$f" "$OUT/${name}_mapped.jsonl" > "$LOGS/map_${name}.log"
      STRICT+=("${name}_mapped"); f=$OUT/${name}_mapped.jsonl ;;
    *) STRICT+=("$name") ;;
  esac
  uv run python -m measure.normalize_kapa_citations "$f" "$OUT/anyformat_${name}.jsonl" --also-inline-links > /dev/null
  ANY+=("anyformat_${name}")
done

say "DeBERTa scoring (decompositions are cached here, the judge reuses them)"
for v in "${STRICT[@]}" "${ANY[@]}"; do
  [ -f "$OUT/scored_${v}_summary.txt" ] && continue   # written last, so a partial run is redone
  HF_HUB_OFFLINE=1 uv run python -m measure.run_scoring "$OUT/$v.jsonl" --out "$OUT/scored_$v.jsonl" \
      > "$LOGS/score_$v.log" 2>&1 || say "WARNING scoring failed: $v"
  say "scored $v"
done

say "gpt-4o judge, any-format view"
for v in "${ANY[@]}"; do
  [ -f "$OUT/rescored_gpt-4o_$v.jsonl" ] && continue
  uv run python -m measure.rescore_judge --judge gpt-4o --runs "repeats/$v" >> "$LOGS/judge.log" 2>&1 \
      || say "WARNING judge failed: $v"
  say "judged $v"
done
say "all done"
