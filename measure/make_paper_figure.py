"""measure/make_paper_figure.py — The paper's CSR figure as TikZ, generated from results/.

Per setup: CSR of each of the 3 runs (small dots) and their mean (large dot), counting only the
requested citation format vs any format, scored by the reference DeBERTa scorer (the only scorer
run on both views for every repeat). Same files as measure/repeats_report.py.

Usage:  uv run python -m measure.make_paper_figure → paper/fig_csr.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from measure.repeats_report import SETUPS, mean_csr

ROWS = ["Single-shot", "LangGraph", "SDK defaults", "SDK E3 Chat Completions", "SDK E4 Kapa setup",
        "v3a SDK (source IDs)", "v3a LangGraph (source IDs)"]
LABELS = {"SDK defaults": "Agents SDK, defaults", "SDK E3 Chat Completions": "SDK + Chat Completions",
          "SDK E4 Kapa setup": "SDK + provider setup", "v3a SDK (source IDs)": "SDK + source IDs (v3a)",
          "v3a LangGraph (source IDs)": "LangGraph + source IDs (v3a)"}
VIEWS = [("strict", "reqcol", "requested format only", -0.14), ("any", "anycol", "any link format", 0.14)]
W = 9.0   # cm of x axis for CSR 0.25 to 0.80


def x(v: float) -> float:
    return (v - 0.25) / 0.55 * W


def main() -> None:
    out = [r"\begin{tikzpicture}[x=1cm, y=-0.62cm]",   # y unit, not yscale: dots stay round
           r"\definecolor{reqcol}{HTML}{2A78D6}\definecolor{anycol}{HTML}{EB6834}"]
    n = len(ROWS)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        out.append(rf"\draw[black!12] ({x(t):.2f},-0.4) -- ({x(t):.2f},{n - 0.4:.2f});")
        out.append(rf"\node[font=\scriptsize,text=black!60] at ({x(t):.2f},{n - 0.05:.2f}) {{{t:.1f}}};")
    out.append(rf"\node[font=\scriptsize,text=black!60] at ({x(0.55):.2f},{n + 0.55:.2f}) "
               r"{CSR (DeBERTa scorer); small dots = runs, large dot = mean of 3};")
    for i, s in enumerate(ROWS):
        out.append(rf"\node[anchor=east,font=\footnotesize] at (-0.15,{i}) {{{LABELS.get(s, s)}}};")
        if i in (2, 5):
            out.append(rf"\draw[black!25] (-5.2,{i - 0.5}) -- ({W:.2f},{i - 0.5});")
        for key, col, _, off in VIEWS:
            vals = [mean_csr(r[key]) for r in SETUPS[s][1]]
            vals = [v for v in vals if v is not None]
            for v in vals:
                out.append(rf"\fill[{col},opacity=0.45] ({x(v):.2f},{i + off:.2f}) circle (1.3pt);")
            out.append(rf"\fill[{col}] ({x(np.mean(vals)):.2f},{i + off:.2f}) circle (2.6pt);")
    lx = 0.0
    for _, col, label, _ in VIEWS:
        out.append(rf"\fill[{col}] ({lx + 0.1:.2f},-1.0) circle (2.6pt);")
        out.append(rf"\node[anchor=west,font=\footnotesize] at ({lx + 0.2:.2f},-1.0) {{{label}}};")
        lx += 4.0
    out.append(r"\end{tikzpicture}")
    Path("paper/fig_csr.tex").write_text("\n".join(out) + "\n")
    print("wrote paper/fig_csr.tex")


if __name__ == "__main__":
    main()
