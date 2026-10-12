"""measure/build_v2_audit_sample.py — Held-out human audit sample for guardrail v2 (Ticket 16).

Design (fixed 2026-10-11, before any label existed):
  - Stratum INSERTED: 50 random v2 insertions (min P(entailment) >= 0.9), pooled across all 5 arms.
    Gives the headline insertion precision at the pre-registered tau.
  - Stratum BAND: 15 random v2 candidates with P < 0.9: every claim argmax-entailed, but rejected
    by tau (P can be < 0.5 under a 3-way argmax). Shows whether the threshold rejects the right
    things; with stratum weights it gives precision for "argmax only" in the sensitivity sweep.
  - Excludes every (arm, question, sentence, url) already audited in Ticket 4 (the dev set).
  - Seeded; output order shuffled so the labeller cannot tell the strata apart.
The `verdict` field is left blank for the human.

Usage:
    uv run python -m measure.build_v2_audit_sample
Output: results/audit_v2_heldout.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from measure.paired_tests import PIPELINES

SEED, N_INSERTED, N_BAND, TAU = 42, 50, 15, 0.9


def main() -> None:
    audited = {(a["question_id"], a["sentence"], a["inserted_url"])
               for a in map(json.loads, open("results/audit_sample_50.jsonl"))}
    pool: list[dict] = []
    for arm in PIPELINES:
        for line in open(f"results/guardrail_v2_{arm}.jsonl"):
            row = json.loads(line)
            content = {}
            for ch in row["retrieved_chunks"]:
                content.setdefault(ch["source_url"], ch["content"])
            for c in row["candidates"]:
                if arm == "langgraph_openai" and (row["question_id"], c["sentence"], c["url"]) in audited:
                    continue
                pool.append({
                    "arm": arm, "question_id": row["question_id"], "question": row["question"],
                    "answer": row["answer_original"], "sentence": c["sentence"], "claims": c["claims"],
                    "inserted_url": c["url"], "chunk_content": content[c["url"]],
                    "min_p_entail": c["min_p_entail"],
                    "stratum": "inserted" if c["min_p_entail"] >= TAU else "band",
                    "is_gold": c["url"] in row["gold_source_urls"],
                })

    rng = random.Random(SEED)
    inserted = [x for x in pool if x["stratum"] == "inserted"]
    band = [x for x in pool if x["stratum"] == "band"]
    sample = rng.sample(inserted, N_INSERTED) + rng.sample(band, N_BAND)
    rng.shuffle(sample)
    for k, s in enumerate(sample, 1):
        s["sample_id"] = k
        s["verdict"] = ""   # human fills: CORRECT | PARTIAL | WRONG

    out = Path("results/audit_v2_heldout.jsonl")
    with out.open("w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")
    print(f"pool: {len(inserted)} inserted, {len(band)} band (after excluding Ticket 4 pairs)")
    print(f"wrote {len(sample)} items to {out}; gold-url items: {sum(s['is_gold'] for s in sample)}")
    print("by arm:", {a: sum(s['arm'] == a for s in sample) for a in PIPELINES})


if __name__ == "__main__":
    main()
