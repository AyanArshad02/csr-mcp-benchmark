"""measure/guardrail_v2_compare.py — Guardrail v1 (naive) vs v2 (Ticket 16), all 5 arms.

Reads results/guardrail_<arm>.jsonl (v1) and results/guardrail_v2_<arm>.jsonl (v2). CSR lifts use
the same paired bootstrap as measure/paired_tests.py (per-question difference, 10,000 resamples,
seed 42). These are SCORER numbers: they say how much CSR moved, not whether the inserted
citations are right. Insertion correctness comes only from the held-out human audit.

Usage:
    uv run python -m measure.guardrail_v2_compare
Output: stdout + results/guardrail_v2_compare.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from measure.paired_tests import PIPELINES, guardrail_lift, load

LABELS = {
    "baseline_openai": "Baseline      gpt-4o-mini",
    "langgraph_openai": "LangGraph     gpt-4o-mini",
    "openai_agent_openai": "OpenAI Agent  gpt-4o-mini",
    "baseline_bedrock": "Baseline      claude-s-4-6",
    "langgraph_bedrock": "LangGraph     claude-s-4-6",
}
SWEEP = (0.0, 0.5, 0.9, 0.95)   # 0.0 = every argmax-entailed candidate


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows if r.get(key) is not None]))


def main() -> None:
    V1, V2 = load("guardrail"), load("guardrail_v2")
    out: list[str] = []
    p = out.append

    p("Guardrail v1 (naive) vs v2 (Ticket 16, tau = 0.9 pre-registered)")
    p("Paired bootstrap of per-question CSR lift, 10,000 resamples, seed 42.")
    p("Scorer numbers only. Insertion correctness comes from the held-out human audit.\n")

    p(f"{'Pipeline':<28}{'CSR before':>11}{'v1 lift [95% CI]':>27}{'v2 lift [95% CI]':>27}")
    p("-" * 93)
    for arm in PIPELINES:
        before = mean(list(V2[arm].values()), "orig_csr")
        cells = []
        for G in (V1, V2):
            _, m, lo, hi = guardrail_lift(G, arm, multi=False)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        p(f"{LABELS[arm]:<28}{before:>11.3f}{cells[0]:>27}{cells[1]:>27}")

    p(f"\n{'Multi-source only':<28}{'':>11}{'v1 lift [95% CI]':>27}{'v2 lift [95% CI]':>27}")
    p("-" * 93)
    for arm in PIPELINES:
        cells = []
        for G in (V1, V2):
            n, m, lo, hi = guardrail_lift(G, arm, multi=True)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        p(f"{LABELS[arm]:<28}{'':>11}{cells[0]:>27}{cells[1]:>27}   (n={n})")

    p(f"\n{'Pipeline':<28}{'cites v1':>9}{'cites v2':>9}{'prec before':>12}{'prec v1':>9}{'prec v2':>9}"
      f"{'F1 v1':>8}{'F1 v2':>8}")
    p("-" * 92)
    for arm in PIPELINES:
        r1, r2 = list(V1[arm].values()), list(V2[arm].values())
        p(f"{LABELS[arm]:<28}{sum(r['citations_added'] for r in r1):>9}{sum(r['citations_added'] for r in r2):>9}"
          f"{mean(r2, 'orig_citation_precision'):>12.3f}{mean(r1, 'patched_citation_precision'):>9.3f}"
          f"{mean(r2, 'patched_citation_precision'):>9.3f}{mean(r1, 'patched_citation_f1'):>8.3f}"
          f"{mean(r2, 'patched_citation_f1'):>8.3f}")

    p("\nv2 candidates by threshold (best chunk argmax-entails every claim; P>=0.0 = argmax only)")
    p(f"{'Pipeline':<28}" + "".join(f"{'P>=' + str(t):>10}" for t in SWEEP))
    p("-" * 68)
    totals = [0] * len(SWEEP)
    for arm in PIPELINES:
        ps = [c["min_p_entail"] for r in V2[arm].values() for c in r["candidates"]]
        counts = [sum(x >= t for x in ps) for t in SWEEP]
        totals = [a + b for a, b in zip(totals, counts)]
        p(f"{LABELS[arm]:<28}" + "".join(f"{c:>10}" for c in counts))
    p(f"{'All arms':<28}" + "".join(f"{c:>10}" for c in totals))

    text = "\n".join(out)
    print(text)
    Path("results/guardrail_v2_compare.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
