"""guardrail/run_guardrail.py — Apply attribution guardrail to a pipeline result file.

Usage:
    python -m guardrail.run_guardrail \
        --input  results/langgraph_openai.jsonl \
        --output results/guardrail_langgraph_openai.jsonl

The script:
  1. Loads the raw pipeline result JSONL (each row has answer + retrieved_chunks).
  2. Runs reattach_citations() on each answer.
  3. Scores both the original and patched answers.
  4. Writes an augmented JSONL with before/after scores.
  5. Prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import diskcache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from sentence_transformers import CrossEncoder

from guardrail.reattach import assert_only_citations_added, reattach_citations
from measure.score import make_decompose_fn, make_nli_fn, score_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _mean(vals: list[float | None]) -> float:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply attribution guardrail")
    parser.add_argument("--input", required=True, help="Input JSONL (raw pipeline results)")
    parser.add_argument("--output", required=True, help="Output JSONL with guardrail applied")
    parser.add_argument("--cache-dir", default=".cache", help="Diskcache directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # --- Setup ---
    logger.info("Loading input: %s", input_path)
    rows = _load_jsonl(input_path)
    logger.info("  %d rows loaded", len(rows))

    cache = diskcache.Cache(args.cache_dir)
    client = OpenAI()

    logger.info("Loading NLI model cross-encoder/nli-deberta-v3-base ...")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    nli_fn = make_nli_fn(nli_model, cache)
    decompose_fn = make_decompose_fn(client, cache)

    # --- Process each row ---
    output_rows = []
    total_citations_added = 0

    for i, row in enumerate(rows):
        qid = row.get("question_id", i)
        question = row.get("question", "")
        original_answer = row.get("answer", "")
        retrieved_chunks = row.get("retrieved_chunks", [])
        gold_source_urls = row.get("gold_source_urls", [])

        logger.info("[%d/%d] q_id=%s", i + 1, len(rows), qid)

        # Apply guardrail
        result = reattach_citations(
            original_answer,
            retrieved_chunks,
            nli_fn=nli_fn,
        )
        # Enforce the "zero wording changes" claim on every single answer.
        assert_only_citations_added(result)
        total_citations_added += result.citations_added

        if result.citations_added:
            logger.info(
                "  +%d citations added (%d/%d uncited sentences covered)",
                result.citations_added,
                result.sentences_covered,
                result.sentences_checked,
            )

        # Score original answer
        orig_score = score_answer(
            answer=original_answer,
            retrieved_chunks=retrieved_chunks,
            gold_source_urls=gold_source_urls,
            question=question,
            nli_fn=nli_fn,
            decompose_fn=decompose_fn,
        )

        # Score patched answer
        patched_score = score_answer(
            answer=result.patched_answer,
            retrieved_chunks=retrieved_chunks,
            gold_source_urls=gold_source_urls,
            question=question,
            nli_fn=nli_fn,
            decompose_fn=decompose_fn,
        )

        output_rows.append({
            **{k: v for k, v in row.items() if k != "answer"},
            "answer_original": original_answer,
            "answer_patched": result.patched_answer,
            "citations_added": result.citations_added,
            "sentences_checked": result.sentences_checked,
            "sentences_covered": result.sentences_covered,
            "sentences_skipped": result.sentences_skipped,
            # Before scores
            "orig_csr": orig_score.csr,
            "orig_citation_recall": orig_score.citation_recall,
            "orig_citation_precision": orig_score.citation_precision,
            "orig_citation_f1": orig_score.citation_f1,
            "orig_error_counts": orig_score.error_counts,
            # After scores
            "patched_csr": patched_score.csr,
            "patched_citation_recall": patched_score.citation_recall,
            "patched_citation_precision": patched_score.citation_precision,
            "patched_citation_f1": patched_score.citation_f1,
            "patched_error_counts": patched_score.error_counts,
        })

    # --- Write output ---
    _write_jsonl(output_path, output_rows)
    logger.info("Wrote %d rows to %s", len(output_rows), output_path)

    # --- Summary table ---
    orig_csrs = [r["orig_csr"] for r in output_rows]
    patched_csrs = [r["patched_csr"] for r in output_rows]
    orig_f1s = [r["orig_citation_f1"] for r in output_rows]
    patched_f1s = [r["patched_citation_f1"] for r in output_rows]
    orig_recalls = [r["orig_citation_recall"] for r in output_rows]
    patched_recalls = [r["patched_citation_recall"] for r in output_rows]

    print("\n" + "=" * 60)
    print(f"GUARDRAIL SUMMARY — {input_path.name}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Before':>8} {'After':>8} {'Δ':>8}")
    print("-" * 60)

    metrics = [
        ("CSR (overall)", orig_csrs, patched_csrs),
        ("Citation Recall", orig_recalls, patched_recalls),
        ("Citation F1", orig_f1s, patched_f1s),
    ]
    for name, before, after in metrics:
        b = _mean(before)
        a = _mean(after)
        print(f"{name:<30} {b:>8.3f} {a:>8.3f} {a - b:>+8.3f}")

    print("-" * 60)
    print(f"{'Total citations inserted':<30} {total_citations_added:>8}")
    print("=" * 60)

    # Multi-source subset
    ms_rows = [r for r in output_rows if r.get("q_type") == "multi-source"]
    if ms_rows:
        ms_orig = _mean([r["orig_csr"] for r in ms_rows])
        ms_patched = _mean([r["patched_csr"] for r in ms_rows])
        print(f"\nMulti-source CSR: {ms_orig:.3f} → {ms_patched:.3f} (Δ {ms_patched - ms_orig:+.3f})")


if __name__ == "__main__":
    main()
