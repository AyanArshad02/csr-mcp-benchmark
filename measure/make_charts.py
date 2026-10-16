"""measure/make_charts.py — The README's headline chart, generated from results/ (Ticket 12).

CSR with 95% bootstrap CIs (same method as measure/paired_tests.py) for the gpt-4o-mini setups,
scored by the gpt-4o judge, two ways: counting only the citation format the prompt asked for, and
counting any citation format (DECISIONS.md 2026-10-14). Plain SVG, no plotting library.
Colours: reference categorical slots 1-2 (#2a78d6, #eb6834), validated (CVD ΔE 24.7, contrast ≥3:1).

Usage:  uv run python -m measure.make_charts  → results/figures/csr_by_setup.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from measure.paired_tests import boot_mean_ci

J = "rescored_gpt-4o_"
ROWS = [  # (label, requested-format file, any-format file), all gpt-4o-judge scored
    ("Single-shot RAG (baseline)", J + "baseline_openai", J + "anyformat_baseline_openai"),
    ("LangGraph agent", J + "langgraph_openai", J + "anyformat_langgraph_openai"),
    ("OpenAI Agents SDK, defaults", J + "openai_agent_openai", J + "anyformat_openai_agent_openai"),
    ("SDK + temperature 0", J + "ablation_sdk_e1_temp0", J + "anyformat_ablation_sdk_e1_temp0"),
    ("SDK + Chat Completions API", J + "ablation_sdk_e3_chat_completions", J + "anyformat_ablation_sdk_e3_chat_completions"),
    ("SDK + Kapa's recommended setup", J + "ablation_sdk_e4_kapa_full_norm", J + "anyformat_ablation_sdk_e4_kapa_full"),
]
SERIES = [("Only the requested citation format counted", "#2a78d6"), ("Any citation format counted", "#eb6834")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def csr_ci(name: str) -> tuple[float, float, float]:
    rows = [json.loads(l) for l in open(f"results/{name}.jsonl")]
    return boot_mean_ci(np.array([r["csr"] for r in rows if r.get("csr") is not None]))


def main() -> None:
    W, left, right, top, row_h = 760, 250, 40, 78, 46
    H = top + row_h * len(ROWS) + 46
    plot_w = W - left - right
    x = lambda v: left + v * plot_w
    s: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="CSR by setup with 95% confidence intervals, counting the requested citation format only versus any format">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="16" y="26" font-size="15" font-weight="600" fill="{INK}">Citation Survival Rate (gpt-4o judge), gpt-4o-mini, 127 questions, 95% CI</text>',
    ]
    # legend: swatch + ink text (text never wears the series colour)
    lx = 16
    for label, colour in SERIES:
        s.append(f'<circle cx="{lx + 5}" cy="49" r="5" fill="{colour}"/>')
        s.append(f'<text x="{lx + 15}" y="53" font-size="12.5" fill="{MUTED}">{label}</text>')
        lx += 15 + len(label) * 6.6 + 22
    # recessive grid + axis labels
    base = top + row_h * len(ROWS)
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.append(f'<line x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x(t):.1f}" y="{base + 18}" font-size="11.5" fill="{MUTED}" text-anchor="middle">{t:.1f}</text>')
    s.append(f'<text x="{left + plot_w / 2:.1f}" y="{base + 38}" font-size="12" fill="{MUTED}" text-anchor="middle">CSR (share of retrieved gold sources cited on a supported claim)</text>')
    for i, (label, deb, jud) in enumerate(ROWS):
        cy = top + row_h * i + row_h / 2
        if i == 2:   # separate the SDK variants from the reference rows
            s.append(f'<line x1="16" y1="{top + row_h * i:.1f}" x2="{W - right}" y2="{top + row_h * i:.1f}" stroke="{GRID}" stroke-width="1"/>')
        weight = "600" if i == 2 else "400"
        s.append(f'<text x="{left - 12}" y="{cy + 4:.1f}" font-size="13" font-weight="{weight}" fill="{INK}" text-anchor="end">{label}</text>')
        for (name, colour), off in zip(((deb, SERIES[0][1]), (jud, SERIES[1][1])), (-7, 7)):
            m, lo, hi = csr_ci(name)
            y = cy + off
            s.append(f'<line x1="{x(lo):.1f}" y1="{y:.1f}" x2="{x(hi):.1f}" y2="{y:.1f}" stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
            s.append(f'<circle cx="{x(m):.1f}" cy="{y:.1f}" r="5" fill="{colour}" stroke="{SURFACE}" stroke-width="2">'
                     f'<title>{label}: {m:.3f} [{lo:.3f}, {hi:.3f}]</title></circle>')
    s.append("</svg>")
    out = Path("results/figures/csr_by_setup.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
