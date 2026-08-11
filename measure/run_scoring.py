"""measure/run_scoring.py — Score a pipeline results JSONL and write per-row + summary output.

Usage:
    uv run python -m measure.run_scoring results/baseline_openai.jsonl
    uv run python -m measure.run_scoring results/langgraph_openai.jsonl
    uv run python -m measure.run_scoring results/openai_agent_openai.jsonl
    uv run python -m measure.run_scoring results/baseline_openai.jsonl --out results/scored_baseline_openai.jsonl

For a quick sanity check on 5 rows (no real NLI/LLM cost):
    uv run python -m measure.run_scoring results/baseline_openai.jsonl --limit 5

Outputs:
    <out>.jsonl       — one row per question with all score fields appended
    <out>_summary.txt — aggregate table (citation recall/precision/F1, CSR, error breakdown)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import diskcache
from dotenv import load_dotenv

from measure.score import make_nli_fn, make_decompose_fn, score_answer, AnswerScore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = ".cache/scoring"


# ---------------------------------------------------------------------------
# Score one row
# ---------------------------------------------------------------------------

def score_row(row: dict, *, nli_fn, decompose_fn) -> dict:
    """Run score_answer on one pipeline result row and return augmented dict."""
    result: AnswerScore = score_answer(
        answer=row["answer"],
        retrieved_chunks=row.get("retrieved_chunks", []),
        gold_source_urls=row.get("gold_source_urls", []),
        question=row["question"],
        nli_fn=nli_fn,
        decompose_fn=decompose_fn,
    )
    return {
        **row,
        "citation_recall":    result.citation_recall,
        "citation_precision": result.citation_precision,
        "citation_f1":        result.citation_f1,
        "csr":                result.csr,
        "csr_numerator":      result.csr_numerator,
        "csr_denominator":    result.csr_denominator,
        "error_counts":       result.error_counts,
        "n_claims":           len(result.claims),
        "n_phantom_urls":     sum(len(c.phantom_urls) for c in result.claims),
    }


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(scored_rows: list[dict]) -> str:
    total = len(scored_rows)
    if total == 0:
        return "No rows to summarise."

    recall    = _mean([r["citation_recall"]    for r in scored_rows])
    precision = _mean([r["citation_precision"] for r in scored_rows])
    f1        = _mean([r["citation_f1"]        for r in scored_rows])

    csr_rows = [r for r in scored_rows if r["csr"] is not None]
    csr      = _mean([r["csr"] for r in csr_rows]) if csr_rows else float("nan")
    csr_eligible = len(csr_rows)

    dropped  = sum(r["error_counts"].get("dropped",        0) for r in scored_rows)
    wrong    = sum(r["error_counts"].get("wrong_attributed",0) for r in scored_rows)
    phantom  = sum(r["n_phantom_urls"]                          for r in scored_rows)

    # Per question-type breakdown
    type_groups: dict[str, list[dict]] = {}
    for r in scored_rows:
        qt = r.get("q_type", "unknown")
        type_groups.setdefault(qt, []).append(r)

    pipeline = scored_rows[0].get("pipeline", "?")
    provider = scored_rows[0].get("provider", "?")
    model    = scored_rows[0].get("model",    "?")

    lines = [
        "=" * 60,
        f"Pipeline : {pipeline}",
        f"Provider : {provider}",
        f"Model    : {model}",
        f"Questions: {total}",
        "=" * 60,
        "",
        "── Citation metrics (ALCE) ──────────────────────",
        f"  Recall    : {recall:.3f}",
        f"  Precision : {precision:.3f}",
        f"  F1        : {f1:.3f}",
        "",
        "── Citation Survival Rate (CSR) ─────────────────",
        f"  CSR       : {csr:.3f}  (over {csr_eligible}/{total} questions with ≥1 gold URL retrieved)",
        "",
        "── Failure modes (total counts across all questions) ──",
        f"  Dropped          : {dropped}   (gold URL retrieved but never cited)",
        f"  Wrong-attributed : {wrong}   (cited but NLI fails — wrong chunk for claim)",
        f"  Phantom citations: {phantom}   (cited URL never retrieved)",
        "",
        "── By question type ──────────────────────────────",
    ]
    for qt, rows in sorted(type_groups.items()):
        qt_recall = _mean([r["citation_recall"] for r in rows])
        qt_f1     = _mean([r["citation_f1"]     for r in rows])
        qt_csr_rows = [r for r in rows if r["csr"] is not None]
        qt_csr = _mean([r["csr"] for r in qt_csr_rows]) if qt_csr_rows else float("nan")
        lines.append(f"  {qt:15s}: n={len(rows):3d}  recall={qt_recall:.3f}  f1={qt_f1:.3f}  csr={qt_csr:.3f}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score a pipeline results JSONL")
    parser.add_argument("input", help="Path to results JSONL (e.g. results/baseline_openai.jsonl)")
    parser.add_argument("--out",   default=None, help="Output JSONL path (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Only score first N rows")
    args = parser.parse_args()

    in_path  = Path(args.input)
    stem     = in_path.stem
    out_path = Path(args.out) if args.out else in_path.parent / f"scored_{stem}.jsonl"
    sum_path = out_path.parent / f"scored_{stem}_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build real NLI fn + decompose fn
    cache = diskcache.Cache(CACHE_DIR)

    logger.info("Loading NLI model (cross-encoder/nli-deberta-v3-base) …")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    nli_fn = make_nli_fn(nli_model, cache)
    logger.info("NLI model loaded.")

    import openai
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    decompose_fn = make_decompose_fn(openai_client, cache)

    # Load rows
    rows = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    logger.info("Scoring %d rows from %s → %s", len(rows), in_path, out_path)

    scored = []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            try:
                s = score_row(row, nli_fn=nli_fn, decompose_fn=decompose_fn)
                scored.append(s)
                out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                out_f.flush()
                logger.info(
                    "[%d/%d] id=%s  recall=%.2f  precision=%.2f  csr=%s  claims=%d",
                    i + 1, len(rows),
                    row.get("question_id", "?"),
                    s["citation_recall"],
                    s["citation_precision"],
                    f"{s['csr']:.2f}" if s["csr"] is not None else "N/A",
                    s["n_claims"],
                )
            except Exception as exc:
                logger.error("[%d/%d] id=%s FAILED: %s", i + 1, len(rows),
                             row.get("question_id", "?"), exc)

    summary = summarise(scored)
    print("\n" + summary)
    sum_path.write_text(summary, encoding="utf-8")
    logger.info("Summary written to %s", sum_path)
    logger.info("Done. Scored %d/%d rows.", len(scored), len(rows))


if __name__ == "__main__":
    main()
