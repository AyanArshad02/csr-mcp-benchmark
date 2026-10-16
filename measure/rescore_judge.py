"""measure/rescore_judge.py — CSR re-scored with an independent LLM judge (Ticket 18).

Same CSR definition as measure/score.py, with one substitution: the support check (Check 1) is the
LLM judge chosen in Ticket 17 (DECISIONS.md 2026-10-13) instead of DeBERTa NLI. Everything else is
identical: same citation parsing, same cached atomic-claim decomposition, same premise (the
concatenated chunks the unit cites), same trace-grounding check, same CSR denominator.
A claim is supported iff the judge says CORRECT (PARTIAL and WRONG are not support: strict, like the
headline gold in the verifier benchmark).

Only CSR is computed. For each retrieved gold URL we judge the claims of units that cite it and stop
at the first supported one; that gives exactly the CSR a full pass would, for a fraction of the calls.
ALCE precision/recall are not re-computed here.

Usage:
    uv run python -m measure.rescore_judge --dry-run              # count calls/tokens, no API
    uv run python -m measure.rescore_judge [--judge gpt-4o|claude] [--runs a,b,...]
Output: results/rescored_<judge>_<run>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import diskcache
from dotenv import load_dotenv

from measure.score import _concat_chunks, make_decompose_fn, parse_cited_units
from measure.verifiers import llm_judge

load_dotenv()

RUNS = ["baseline_openai", "langgraph_openai", "openai_agent_openai", "baseline_bedrock",
        "langgraph_bedrock", "ablation_sdk_e0_original", "ablation_sdk_e1_temp0",
        "ablation_sdk_e2_kapa_tooldesc", "ablation_sdk_e3_chat_completions", "ablation_sdk_e4_kapa_full_norm"]


def judge_csr(row: dict, decompose, judge) -> dict:
    """Return csr / numerator / denominator and per-gold-URL outcome for one answer."""
    url_to_content: dict[str, str] = {}
    for ch in row.get("retrieved_chunks", []):
        if ch.get("source_url") and ch["source_url"] not in url_to_content:
            url_to_content[ch["source_url"]] = ch.get("content", "")
    retrieved_gold = [u for u in dict.fromkeys(row.get("gold_source_urls", [])) if u in url_to_content]
    units = parse_cited_units(row.get("answer", ""))
    outcome: dict[str, str] = {}
    for gold in retrieved_gold:
        citing = [(s, urls) for s, urls in units if gold in urls]
        if not citing:
            outcome[gold] = "dropped"; continue
        outcome[gold] = "wrong_attributed"
        for sentence, urls in citing:
            premise = _concat_chunks([u for u in urls if u in url_to_content], url_to_content)
            for claim in decompose(sentence, row.get("question", "")):
                if judge(premise, claim):
                    outcome[gold] = "survived"; break
            if outcome[gold] == "survived":
                break
    num = sum(v == "survived" for v in outcome.values())
    den = len(retrieved_gold)
    return {"csr": num / den if den else None, "csr_numerator": num, "csr_denominator": den,
            "gold_outcomes": outcome}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["gpt-4o", "claude"], default="gpt-4o")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--dry-run", action="store_true", help="count judge calls and tokens; no API calls")
    args = ap.parse_args()
    cache = diskcache.Cache(".cache")
    cached_decompose = make_decompose_fn(None, cache)   # cache-only: decompositions exist from scoring
    misses = {"n": 0}

    def decompose(sentence: str, question: str) -> list[str]:
        # A cache miss means the original scorer's decomposition call failed, and score_answer then
        # used the whole sentence as one claim; do exactly the same here.
        try:
            return cached_decompose(sentence, question)
        except Exception:
            misses["n"] += 1
            return [sentence]

    calls = {"n": 0, "chars": 0}
    if args.dry_run:
        def judge(premise: str, claim: str) -> bool:
            calls["n"] += 1; calls["chars"] += len(premise) + len(claim)
            return False                            # worst case: no early stop
    else:
        if args.judge == "gpt-4o":
            from pipelines.strategy.openai_s import OpenAIStrategy
            strat = OpenAIStrategy(model="gpt-4o", cache=cache)
        else:
            from pipelines.strategy.bedrock_s import BedrockStrategy
            strat = BedrockStrategy(cache=cache)
        chat = lambda msgs: strat.chat(msgs, temperature=0.0, max_tokens=300)

        def judge(premise: str, claim: str) -> bool:
            calls["n"] += 1
            return llm_judge(chat, premise, claim)[0].supported

    for run in args.runs.split(","):
        rows = [json.loads(l) for l in open(f"results/{run}.jsonl") if l.strip()]
        before = calls["n"]
        out = []
        for i, row in enumerate(rows, 1):
            res = judge_csr(row, decompose, judge)
            out.append({"question_id": row["question_id"], "q_type": row.get("q_type"), **res})
            if not args.dry_run and i % 25 == 0:
                print(f"  {run}: {i}/{len(rows)}", flush=True)
        n = calls["n"] - before
        if args.dry_run:
            print(f"{run:36s} worst-case judge calls {n:5d}")
        else:
            src = Path(f"results/{run}.jsonl")   # run may sit in a subfolder (results/repeats/...)
            path = src.with_name(f"rescored_{args.judge}_{src.name}")
            with path.open("w") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
            vals = [r["csr"] for r in out if r["csr"] is not None]
            print(f"{run:36s} CSR {sum(vals)/len(vals):.3f}  ({n} judge calls) → {path}", flush=True)
    print(f"decomposition cache misses (whole sentence used, as the scorer did): {misses['n']}")
    if args.dry_run:
        tok = calls["chars"] / 4 + calls["n"] * 350          # chars/4 + rubric overhead per call
        print(f"\nTOTAL worst-case calls {calls['n']}, ~{tok/1e6:.1f}M input tokens "
              f"(~${tok/1e6*2.5 + calls['n']*60/1e6*10:.0f} at gpt-4o list price; early stopping cuts this)")


if __name__ == "__main__":
    main()
