"""measure/build_csr_validation_sample.py — Human validation sample for CSR itself (Ticket 11, gate G4).

Design (fixed 2026-10-13, before any label existed):
  - Unit = one (question, pipeline, retrieved gold URL): exactly what CSR counts.
  - 6 questions per headline pipeline (5 pipelines → 30 units), seeded random among questions where
    the agent retrieved at least one gold URL; one retrieved gold URL picked at random per question.
  - Human question: "Is this gold source cited in the answer, on a sentence its chunk genuinely
    supports?" → SURVIVED / NOT_SURVIVED.
  - Each scorer's verdict for the same unit is stored (DeBERTa, gpt-4o judge, Claude judge) and kept
    hidden from the labeller until the human verdict is locked.
Note: the scorer's premise is the concatenation of ALL chunks a sentence cites; the human judges the
gold chunk alone (the CSR definition in CLAUDE.md). Disagreements from that difference are reported.

Usage:  uv run python -m measure.build_csr_validation_sample  → results/csr_validation_sample.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import diskcache

from measure.rescore_judge import judge_csr
from measure.score import make_decompose_fn, make_nli_fn, parse_cited_units

ARMS = ["baseline_openai", "langgraph_openai", "openai_agent_openai", "baseline_bedrock", "langgraph_bedrock"]
SEED, PER_ARM = 42, 6


class _CacheOnlyNLI:
    def predict(self, *a, **k):
        raise RuntimeError("NLI cache miss: every pair here was scored before")


def main() -> None:
    cache = diskcache.Cache(".cache")
    nli = make_nli_fn(_CacheOnlyNLI(), cache)
    cached_decompose = make_decompose_fn(None, cache)

    def decompose(s: str, q: str) -> list[str]:
        try:
            return cached_decompose(s, q)
        except Exception:
            return [s]   # same fallback as score_answer

    deberta = lambda premise, claim: nli(premise, claim) == "entailment"
    rng = random.Random(SEED)
    sample: list[dict] = []
    for arm in ARMS:
        raw = {r["question_id"]: r for r in map(json.loads, open(f"results/{arm}.jsonl"))}
        judged = {j: {r["question_id"]: r for r in map(json.loads, open(f"results/rescored_{j}_{arm}.jsonl"))}
                  for j in ("gpt-4o", "claude")}
        eligible = sorted(q for q, r in judged["gpt-4o"].items() if r["csr_denominator"])
        for qid in rng.sample(eligible, PER_ARM):
            row = raw[qid]
            gold = rng.choice(sorted(judged["gpt-4o"][qid]["gold_outcomes"]))
            chunk = next(c["content"] for c in row["retrieved_chunks"] if c["source_url"] == gold)
            citing = [s for s, urls in parse_cited_units(row["answer"]) if gold in urls]
            sample.append({
                "arm": arm, "question_id": qid, "question": row["question"], "answer": row["answer"],
                "gold_url": gold, "gold_chunk": chunk, "citing_sentences": citing,
                "scorer_deberta": judge_csr(row, decompose, deberta)["gold_outcomes"][gold],
                "scorer_gpt4o": judged["gpt-4o"][qid]["gold_outcomes"][gold],
                "scorer_claude": judged["claude"][qid]["gold_outcomes"][gold],
            })
    rng.shuffle(sample)
    for k, s in enumerate(sample, 1):
        s["sample_id"] = k
        s["verdict"] = ""   # human fills: SURVIVED | NOT_SURVIVED
    out = Path("results/csr_validation_sample.jsonl")
    with out.open("w") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(sample)} units → {out}")
    for k in ("scorer_deberta", "scorer_gpt4o", "scorer_claude"):
        print(f"  {k}: {dict(Counter(s[k] for s in sample))}")
    print(f"  units where the gold URL is never cited: {sum(not s['citing_sentences'] for s in sample)}")


if __name__ == "__main__":
    main()
