"""Filter human-verified candidates into the final eval.jsonl.

Priority order for input source (first file found wins):
  1. eval_set/final_candidates.xlsx  — exported from Google Sheets after human review.
                                       Every row in this file is treated as approved.
                                       Just delete rejected rows in Sheets before exporting.
  2. eval_set/candidates.json        — pretty-printed JSON array; filter on "verified": true.
  3. eval_set/candidates.jsonl       — newline-delimited JSON; filter on "verified": true.

Usage:
    uv run python -m eval_set.finalize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FINAL_XLSX   = Path("eval_set/final_candidates.xlsx")
CANDIDATES_JSON  = Path("eval_set/candidates.json")
CANDIDATES_JSONL = Path("eval_set/candidates.jsonl")
OUTPUT_PATH  = Path("eval_set/eval.jsonl")

FINAL_FIELDS = {"question", "reference_answer", "gold_source_urls", "difficulty", "type", "candidate_id"}


def _load_from_xlsx(path: Path) -> list[dict]:
    """Read the human-reviewed Excel export.
    Filters to rows where the 'verified' column is 'True' (case-insensitive).
    Rows left as 'False' are rejected and excluded."""
    import pandas as pd
    df = pd.read_excel(path, dtype=str).fillna("")
    approved = df[df["verified"].str.strip().str.lower() == "true"]
    logger.info(f"  {len(approved)} approved / {len(df)} total rows in {path.name}")
    records = []
    for _, row in approved.iterrows():
        item = row.to_dict()
        # gold_source_urls was stored as newline-joined string in sheets.py — reverse that
        raw_urls = item.get("gold_source_urls", "")
        item["gold_source_urls"] = [u.strip() for u in raw_urls.split("\n") if u.strip()]
        records.append(item)
    return records


def _load_from_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array at the top level")
    return [c for c in data if c.get("verified") is True]


def _load_from_jsonl(path: Path) -> list[dict]:
    candidates = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("verified") is True:
                    candidates.append(item)
            except json.JSONDecodeError as exc:
                logger.warning(f"Line {lineno}: JSON parse error — {exc}")
    return candidates


def _load_candidates() -> list[dict]:
    if FINAL_XLSX.exists():
        logger.info(f"Reading from {FINAL_XLSX} (Google Sheets export — all rows treated as approved)")
        return _load_from_xlsx(FINAL_XLSX)
    if CANDIDATES_JSON.exists():
        logger.info(f"Reading from {CANDIDATES_JSON} (filtering on verified: true)")
        return _load_from_json(CANDIDATES_JSON)
    if CANDIDATES_JSONL.exists():
        logger.info(f"Reading from {CANDIDATES_JSONL} (filtering on verified: true)")
        return _load_from_jsonl(CANDIDATES_JSONL)
    raise FileNotFoundError(
        "No candidate file found. Expected one of:\n"
        "  eval_set/final_candidates.xlsx\n"
        "  eval_set/candidates.json\n"
        "  eval_set/candidates.jsonl"
    )


def finalize() -> None:
    verified = _load_candidates()

    if not verified:
        logger.error(
            "No approved items found.\n"
            "  If using final_candidates.xlsx: make sure rejected rows were deleted before export.\n"
            "  If using candidates.json/jsonl: set 'verified': true on items you want to keep."
        )
        return

    type_counts: dict[str, int] = {}
    for c in verified:
        t = str(c.get("type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1

    single = type_counts.get("single-source", 0)
    multi  = type_counts.get("multi-source", 0)
    unans  = type_counts.get("unanswerable", 0)
    total  = len(verified)

    if total < 100:
        logger.warning(f"Only {total} approved items — aim for ~150 for a credible benchmark.")
    if multi < 30:
        logger.warning(f"Only {multi} multi-source items — these are your most informative cases; aim for ≥30.")
    if unans < 10:
        logger.warning(f"Only {unans} unanswerable items — aim for ≥10 to test abstention behaviour.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, c in enumerate(verified):
            item = {k: v for k, v in c.items() if k in FINAL_FIELDS}
            item["id"] = i
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {total} items → {OUTPUT_PATH}")
    logger.info(f"  single-source:  {single}")
    logger.info(f"  multi-source:   {multi}")
    logger.info(f"  unanswerable:   {unans}")
    logger.info("\neval.jsonl is ready. Commit it — it is the credibility anchor of the whole benchmark.")


if __name__ == "__main__":
    finalize()
