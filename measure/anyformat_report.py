"""measure/anyformat_report.py — CSR with citation FORMAT separated from citation PRESENCE.

Pre-registered in DECISIONS.md (2026-10-14). "Strict" = the original numbers: only `[url]`, the
format the prompt asked for, counts as a citation. "Any format" = `[url]`, Markdown links
`[text](url)` and Kapa brackets `[[title](url)]` all count (converted by
measure/normalize_kapa_citations.py --also-inline-links); bare URLs are not counted. Same CSR
definition and scorers. Paired bootstrap as in measure/paired_tests.py (10,000 resamples, seed 42).

Usage:  uv run python -m measure.anyformat_report → stdout + results/anyformat_report.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from measure.paired_tests import boot_mean_ci, verdict

RUNS = {
    "baseline_openai": "Single-shot · gpt-4o-mini",
    "langgraph_openai": "LangGraph · gpt-4o-mini",
    "openai_agent_openai": "OpenAI SDK (published) · gpt-4o-mini",
    "baseline_bedrock": "Single-shot · Claude",
    "langgraph_bedrock": "LangGraph · Claude",
    "ablation_sdk_e0_original": "SDK E0 defaults, re-run",
    "ablation_sdk_e1_temp0": "SDK E1 temperature 0",
    "ablation_sdk_e2_kapa_tooldesc": "SDK E2 + Kapa tool description",
    "ablation_sdk_e3_chat_completions": "SDK E3 + Chat Completions",
    "ablation_sdk_e4_kapa_full": "SDK E4 Kapa recommended setup",
}
PAIRS = [("baseline_openai", "openai_agent_openai"), ("baseline_openai", "langgraph_openai"),
         ("baseline_bedrock", "langgraph_bedrock"), ("ablation_sdk_e0_original", "ablation_sdk_e1_temp0"),
         ("ablation_sdk_e1_temp0", "ablation_sdk_e2_kapa_tooldesc"),
         ("ablation_sdk_e1_temp0", "ablation_sdk_e3_chat_completions"),
         ("ablation_sdk_e1_temp0", "ablation_sdk_e4_kapa_full"), ("baseline_openai", "ablation_sdk_e1_temp0")]
STRICT = re.compile(r"\[(https?://[^\]\s]+)\]")


def strict_file(run: str, scorer: str) -> str:
    if run == "ablation_sdk_e4_kapa_full":   # strict for E4 = Kapa brackets only (its own requested format)
        run = "ablation_sdk_e4_kapa_full_norm"
    return f"results/scored_{run}.jsonl" if scorer == "deberta" else f"results/rescored_gpt-4o_{run}.jsonl"


def any_file(run: str, scorer: str) -> str:
    return (f"results/scored_anyformat_{run}.jsonl" if scorer == "deberta"
            else f"results/rescored_gpt-4o_anyformat_{run}.jsonl")


def load(path: str) -> dict[int, dict] | None:
    p = Path(path)
    return {r["question_id"]: r for r in map(json.loads, p.open())} if p.exists() else None


def csr(S: dict | None) -> str:
    if not S:
        return "—"
    m, lo, hi = boot_mean_ci(np.array([r["csr"] for r in S.values() if r.get("csr") is not None]))
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    out: list[str] = []; p = out.append
    p("CSR: strict [url] format vs any citation format (pre-registered, DECISIONS.md 2026-10-14)")
    p("Any format = [url], Markdown links, Kapa brackets. Bare URLs not counted.\n")
    p(f"{'Run':<38}{'DeBERTa strict':<22}{'DeBERTa any-format':<22}{'gpt-4o judge strict':<22}{'gpt-4o judge any-format':<22}{'no URL at all':>14}")
    p("-" * 140)
    for run, label in RUNS.items():
        raw = [json.loads(l) for l in open(f"results/anyformat_{run}.jsonl")]
        none = sum(not re.search(r"https?://", r["answer_raw"]) for r in raw)
        p(f"{label:<38}{csr(load(strict_file(run, 'deberta'))):<22}{csr(load(any_file(run, 'deberta'))):<22}"
          f"{csr(load(strict_file(run, 'judge'))):<22}{csr(load(any_file(run, 'judge'))):<22}{none:>10}/{len(raw)}")
    for view, fn in (("strict format", strict_file), ("ANY format", any_file)):
        p(f"\nPaired differences (B − A), {view}")
        p(f"{'Comparison':<72}{'DeBERTa Δ [CI]':<34}{'gpt-4o judge Δ [CI]':<34}")
        p("-" * 140)
        for a, b in PAIRS:
            cells = []
            for scorer in ("deberta", "judge"):
                A, B = load(fn(a, scorer)), load(fn(b, scorer))
                if not A or not B:
                    cells.append("—"); continue
                qs = [q for q in A if q in B and A[q].get("csr") is not None and B[q].get("csr") is not None]
                m, lo, hi = boot_mean_ci(np.array([B[q]["csr"] - A[q]["csr"] for q in qs]))
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] {'REAL' if verdict(lo, hi).startswith('REAL') else 'n.s.'}")
            p(f"{RUNS[b] + '  −  ' + RUNS[a]:<72}{cells[0]:<34}{cells[1]:<34}")
    text = "\n".join(out); print(text)
    Path("results/anyformat_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
