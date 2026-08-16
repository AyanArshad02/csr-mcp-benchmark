"""measure/lift_decomposition.py — Where does the guardrail's CSR lift actually come from?

The guardrail attributes per SENTENCE; the scorer attributes per citation-delimited
SEGMENT (which can span several sentences). So patching an answer does two things at
once: it adds new [url] tags, AND it makes the scorer's units finer. A reviewer will
ask how much of the lift is real attribution vs. re-granularization.

For every gold URL that flips from "not survived" to "survived" after patching, we ask:
  - NEW ATTRIBUTION  — the URL was not cited anywhere in the original answer.
  - RE-GRANULARIZED  — the URL was already cited in the original, but on a coarse unit
                       that failed the entailment check; patching split it finer.

Only the first is a genuine attribution recovery. Both are reported.

Usage:
    uv run python -m measure.lift_decomposition
Output: stdout + results/lift_decomposition.txt
"""
from __future__ import annotations

import json
from pathlib import Path

from measure.score import _CITATION_RE

GUARDRAIL_FILES = {
    "Baseline      gpt-4o-mini":  "results/guardrail_baseline_openai.jsonl",
    "LangGraph     gpt-4o-mini":  "results/guardrail_langgraph_openai.jsonl",
    "OpenAI Agent  gpt-4o-mini":  "results/guardrail_openai_agent_openai.jsonl",
    "Baseline      claude-s-4-6": "results/guardrail_baseline_bedrock.jsonl",
    "LangGraph     claude-s-4-6": "results/guardrail_langgraph_bedrock.jsonl",
}


def analyse(path: str) -> tuple[int, int, int]:
    """Return (n_flipped, n_new_attribution, n_regranularized) over all questions."""
    flipped = new_attr = regran = 0
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        orig_csr, patched_csr = row.get("orig_csr"), row.get("patched_csr")
        if orig_csr is None or patched_csr is None or patched_csr <= orig_csr:
            continue

        # Gold URLs that were retrieved (the CSR denominator set)
        retrieved = {c.get("source_url") for c in row.get("retrieved_chunks", [])}
        gold_retrieved = set(row.get("gold_source_urls", [])) & retrieved

        cited_before = set(_CITATION_RE.findall(row.get("answer_original", "")))
        cited_after = set(_CITATION_RE.findall(row.get("answer_patched", "")))

        # Approximate the per-URL flip: URLs newly surviving must be cited after.
        n_flip_this_q = round((patched_csr - orig_csr) * len(gold_retrieved))
        candidates = [u for u in gold_retrieved if u in cited_after]
        newly_cited = [u for u in candidates if u not in cited_before]

        flipped += n_flip_this_q
        # Attribute flips to "new attribution" first, remainder to re-granularization.
        n_new = min(n_flip_this_q, len(newly_cited))
        new_attr += n_new
        regran += n_flip_this_q - n_new
    return flipped, new_attr, regran


def main() -> None:
    lines = [
        "Guardrail CSR lift decomposition",
        "(gold URLs flipping not-survived -> survived after patching)",
        "",
        f"{'Pipeline':<30}  {'flips':>6}  {'new attribution':>16}  {'re-granularized':>16}",
        "-" * 76,
    ]
    for label, path in GUARDRAIL_FILES.items():
        if not Path(path).exists():
            lines.append(f"{label:<30}  (missing: {path})")
            continue
        flips, new_attr, regran = analyse(path)
        pct = f"{new_attr / flips * 100:.0f}%" if flips else "n/a"
        lines.append(
            f"{label:<30}  {flips:>6}  {new_attr:>10} ({pct:>4})  {regran:>16}"
        )

    out = "\n".join(lines) + "\n"
    print(out)
    Path("results/lift_decomposition.txt").write_text(out)
    print("Saved to results/lift_decomposition.txt")


if __name__ == "__main__":
    main()
