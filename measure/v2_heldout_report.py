"""measure/v2_heldout_report.py — Guardrail v2 held-out audit results (Ticket 16).

Inputs: results/audit_v2_heldout_verified.jsonl (human verdicts, blind first call kept) and
results/audit_v2_heldout_claude.jsonl (Claude's independent labels, made before seeing the human's).
Precision = CORRECT / n (PARTIAL also reported as lenient precision). Wilson 95% intervals.
"argmax only" precision is stratum-weighted: pool had 350 inserted (P>=0.9) and 105 band (P<0.9).

Usage:  uv run python -m measure.v2_heldout_report   → stdout + results/guardrail_v2_heldout_report.txt
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

N_POOL_INSERTED, N_POOL_BAND = 350, 105
CATS = ("CORRECT", "PARTIAL", "WRONG")


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f}% [{100 * (c - h):.1f}, {100 * (c + h):.1f}]"


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in CATS)
    return (po - pe) / (1 - pe)


def main() -> None:
    rows = [json.loads(l) for l in open("results/audit_v2_heldout_verified.jsonl")]
    out: list[str] = []; p = out.append
    p("Guardrail v2 held-out human audit (65 items: 50 inserted at P>=0.9, 15 rejected band P<0.9)")
    p("Human = blind first call. Precision = CORRECT/n; lenient = (CORRECT+PARTIAL)/n. Wilson 95% CI.\n")
    groups = {
        "Inserted by v2 (tau=0.9) — HEADLINE": [r for r in rows if r["stratum"] == "inserted"],
        "  of which gold-source urls": [r for r in rows if r["stratum"] == "inserted" and r["is_gold"]],
        "  of which P>=0.95": [r for r in rows if r["stratum"] == "inserted" and r["min_p_entail"] >= 0.95],
        "Rejected by v2 (P<0.9)": [r for r in rows if r["stratum"] == "band"],
    }
    for name, g in groups.items():
        c = {k: sum(r["human_first_verdict"] == k for r in g) for k in CATS}
        p(f"{name}")
        p(f"   C/P/W = {c['CORRECT']}/{c['PARTIAL']}/{c['WRONG']}   precision {wilson(c['CORRECT'], len(g))}"
          f"   lenient {wilson(c['CORRECT'] + c['PARTIAL'], len(g))}")
    ins, band = groups["Inserted by v2 (tau=0.9) — HEADLINE"], groups["Rejected by v2 (P<0.9)"]
    pi = sum(r["human_first_verdict"] == "CORRECT" for r in ins) / len(ins)
    pb = sum(r["human_first_verdict"] == "CORRECT" for r in band) / len(band)
    pw = (N_POOL_INSERTED * pi + N_POOL_BAND * pb) / (N_POOL_INSERTED + N_POOL_BAND)
    p(f"\nArgmax-only precision (stratum-weighted, no threshold): {100 * pw:.1f}%   vs tau=0.9: {100 * pi:.1f}%")
    p("Guardrail v1 (Ticket 4, one arm, n=50): 15/50 = 30.0% [19.1, 43.8] correct")

    pairs = [(r["human_first_verdict"], r["claude_verdict"]) for r in rows]
    agree = sum(a == b for a, b in pairs)
    p(f"\nHuman (blind first call) vs Claude (independent): agreement {agree}/65, Cohen's kappa {kappa(pairs):.2f}")
    dis = [r for r in rows if r["human_first_verdict"] != r["claude_verdict"]]
    for r in dis:
        p(f"   #{r['sample_id']:<2} human {r['human_first_verdict']:<7} claude {r['claude_verdict']:<7} "
          f"{r['stratum']:<8} note: {r.get('human_note', '')}")
    changed = [r["sample_id"] for r in rows if r.get("human_changed_after_reveal")]
    p(f"Human changed after reveal: {changed or 'none'}")
    text = "\n".join(out); print(text)
    Path("results/guardrail_v2_heldout_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
