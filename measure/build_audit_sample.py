"""measure/build_audit_sample.py — Build the manual-audit sample for guardrail insertions.

Ticket 4. The guardrail inserts [url] tags using the same NLI model the scorer uses,
so part of the measured CSR lift could be the model agreeing with itself. This script
draws a random sample of ACTUAL insertion decisions so a human can judge precision.

Correctness note: the (sentence, url) pairs come from ReattachResult.decisions — the
guardrail's own record of what it did. They are NOT recovered by re-parsing the patched
answer, which is unsound: reattach emits "<sentence> [url]", so splitting that on
sentence-ending punctuation makes the trailing "[url]" look like its own sentence.

Usage:
    uv run python -m measure.build_audit_sample \
        --input results/guardrail_langgraph_openai.jsonl \
        --out   results/audit_sample_50.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import diskcache
from dotenv import load_dotenv

load_dotenv()
from sentence_transformers import CrossEncoder

from guardrail.reattach import reattach_citations
from measure.score import make_nli_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/guardrail_langgraph_openai.jsonl")
    ap.add_argument("--out", default="results/audit_sample_50.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=".cache")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    cache = diskcache.Cache(args.cache_dir)
    nli_fn = make_nli_fn(CrossEncoder("cross-encoder/nli-deberta-v3-base"), cache)

    all_ins: list[dict] = []
    for row in rows:
        result = reattach_citations(
            row.get("answer_original", ""),
            row.get("retrieved_chunks", []),
            nli_fn=nli_fn,
        )
        for d in result.decisions:
            all_ins.append({
                "question_id": row.get("question_id"),
                "question": row.get("question", ""),
                "sentence": d.sentence,
                "inserted_url": d.url,
                "chunk_content": d.chunk_content,
            })

    reported = sum(r.get("citations_added", 0) for r in rows)
    print(f"Replayed insertions: {len(all_ins)}  (guardrail run reported: {reported})")
    if len(all_ins) != reported:
        print("WARNING: replay disagrees with the guardrail run — re-run the guardrail first.")

    rng = random.Random(args.seed)
    sample = rng.sample(all_ins, min(args.n, len(all_ins)))
    for k, s in enumerate(sample, 1):
        s["sample_id"] = k
        s["verdict"] = ""   # human fills: CORRECT | PARTIAL | WRONG

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(sample)} samples to {out}")


if __name__ == "__main__":
    main()
