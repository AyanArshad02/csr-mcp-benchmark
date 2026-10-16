"""measure/normalize_kapa_citations.py — Convert Kapa-format citations to the benchmark's [url].

Kapa's recommended prompt asks for `[[short title](URL)]`, or `[[t1](u1); [t2](u2)]` for two
sources. The scorer parses only `[url]` (locked citation format), so answers produced with the
Kapa prompt are normalised before scoring. Only this exact Kapa bracket form is converted
(pre-registered 2026-10-13); plain inline links `[text](url)` are left alone and counted, so a
sensitivity run can convert them too (`--also-inline-links`).

Usage:
    uv run python -m measure.normalize_kapa_citations IN.jsonl OUT.jsonl [--also-inline-links]
The original answer is kept as `answer_raw`.
"""
from __future__ import annotations

import argparse
import json
import re

_LINK = r"\[[^\[\]]*\]\((https?://[^)\s]+)\)"
_KAPA_BRACKET = re.compile(r"\[\s*" + _LINK + r"(?:\s*;\s*" + _LINK + r")*\s*\]")
_INLINE_LINK = re.compile(_LINK)


def normalize(answer: str, also_inline_links: bool = False) -> tuple[str, int, int]:
    """Return (normalised answer, Kapa brackets converted, inline links seen/converted)."""
    n_kapa = 0

    def _kapa(m: re.Match) -> str:
        nonlocal n_kapa
        n_kapa += 1
        return "".join(f"[{u}]" for u in re.findall(r"\((https?://[^)\s]+)\)", m.group(0)))

    out = _KAPA_BRACKET.sub(_kapa, answer)
    n_inline = len(_INLINE_LINK.findall(out))
    if also_inline_links:
        out = _INLINE_LINK.sub(lambda m: f"[{m.group(1)}]", out)
    return out, n_kapa, n_inline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--also-inline-links", action="store_true")
    a = ap.parse_args()
    tot_k = tot_i = 0
    with open(a.inp) as f, open(a.out, "w") as g:
        for line in f:
            row = json.loads(line)
            row.setdefault("answer_raw", row["answer"])   # keep an earlier raw answer (v3a IDs)
            row["answer"], k, i = normalize(row["answer"], a.also_inline_links)
            tot_k += k; tot_i += i
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Kapa brackets converted: {tot_k} | inline links {'converted' if a.also_inline_links else 'left as-is'}: {tot_i}")


if __name__ == "__main__":
    main()
