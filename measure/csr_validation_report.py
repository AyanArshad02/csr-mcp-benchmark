"""measure/csr_validation_report.py — Does CSR agree with a human? (Ticket 11, gate G4)

Input: results/csr_validation_verified.jsonl (sample + the human's blind first call).
Unit = one (question, pipeline, retrieved gold URL). Human: SURVIVED / NOT_SURVIVED. Each scorer's
verdict (DeBERTa, gpt-4o judge, Claude judge) maps "survived" → SURVIVED, else NOT_SURVIVED.
Reported on all 30 units and on the units where the gold URL is cited (excluding "never cited",
which every scorer gets right trivially). Cohen's kappa for two binary raters; Wilson 95% CI on
agreement.

Usage:  uv run python -m measure.csr_validation_report → stdout + results/csr_validation_report.txt
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

SCORERS = [("DeBERTa (locked scorer)", "scorer_deberta"), ("gpt-4o judge", "scorer_gpt4o"), ("Claude judge", "scorer_claude")]


def wilson(k: int, n: int, z: float = 1.96) -> str:
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.0f}% [{100 * (c - h):.0f}, {100 * (c + h):.0f}]"


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in open("results/csr_validation_verified.jsonl")]
    out: list[str] = []; p = out.append
    human = lambda r: r["human_first_verdict"] == "SURVIVED"
    scorer = lambda r, k: r[k] == "survived"
    p("CSR human validation (Ticket 11): 30 units, 6 per headline pipeline, human = blind first call\n")
    p(f"Human: SURVIVED {sum(map(human, rows))}, NOT_SURVIVED {len(rows) - sum(map(human, rows))}")
    for title, sub in (("All 30 units", rows), ("Only units where the gold URL is cited", [r for r in rows if r["citing_sentences"]])):
        p(f"\n{title} (n={len(sub)}, human-survived {sum(map(human, sub))})")
        p(f"{'Scorer':<26}{'agreement [95% CI]':<24}{'kappa':>6}   scorer-yes/human-no  scorer-no/human-yes")
        for name, k in SCORERS:
            agree = sum(human(r) == scorer(r, k) for r in sub)
            fp = sum(scorer(r, k) and not human(r) for r in sub); fn = sum(not scorer(r, k) and human(r) for r in sub)
            p(f"{name:<26}{wilson(agree, len(sub)):<24}{kappa([human(r) for r in sub], [scorer(r, k) for r in sub]):>6.2f}   {fp:>10}            {fn:>10}")
    p("\nDisagreements (any scorer vs human):")
    for r in rows:
        diff = [n for n, k in SCORERS if scorer(r, k) != human(r)]
        if diff:
            p(f"  #{r['sample_id']:<2} {r['arm']:<20} human {r['human_first_verdict']:<12} differs: {', '.join(diff)}"
              f"  | note: {r.get('human_note', '')[:140]}")
    changed = [r["sample_id"] for r in rows if r.get("human_changed_after_reveal")]
    p(f"\nHuman changed a verdict after the reveal: {changed or 'none'} (first call used throughout)")
    text = "\n".join(out); print(text)
    Path("results/csr_validation_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
