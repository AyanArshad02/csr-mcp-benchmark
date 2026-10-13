"""measure/rescore_compare.py — CSR under the locked DeBERTa scorer vs the independent judge (Ticket 18).

Both columns use the same answers, decompositions, premises and CSR definition; only the support
check differs. Paired bootstrap (measure/paired_tests.py: 10,000 resamples, seed 42) for the
pipeline comparisons, run separately under each scorer.

Usage:  uv run python -m measure.rescore_compare [--judge gpt-4o] → stdout + results/rescore_compare_<judge>.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from measure.paired_tests import boot_mean_ci, verdict

ARMS = {"baseline_openai": "Baseline · gpt-4o-mini", "langgraph_openai": "LangGraph · gpt-4o-mini",
        "openai_agent_openai": "OpenAI SDK · gpt-4o-mini", "baseline_bedrock": "Baseline · Claude",
        "langgraph_bedrock": "LangGraph · Claude",
        "ablation_sdk_e1_temp0": "SDK E1 temperature 0", "ablation_sdk_e3_chat_completions": "SDK E3 Chat Completions",
        "ablation_sdk_e4_kapa_full_norm": "SDK E4 Kapa full setup"}
PAIRS = [("baseline_openai", "openai_agent_openai"), ("baseline_openai", "langgraph_openai"),
         ("baseline_bedrock", "langgraph_bedrock"), ("ablation_sdk_e1_temp0", "ablation_sdk_e3_chat_completions"),
         ("ablation_sdk_e1_temp0", "ablation_sdk_e4_kapa_full_norm")]


def load(path: str) -> dict[int, dict] | None:
    p = Path(path)
    return {r["question_id"]: r for r in map(json.loads, p.open())} if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", default="gpt-4o"); judge = ap.parse_args().judge
    old = {a: load(f"results/scored_{a}.jsonl") for a in ARMS}
    new = {a: load(f"results/rescored_{judge}_{a}.jsonl") for a in ARMS}
    out: list[str] = []; p = out.append
    p(f"CSR: locked DeBERTa scorer vs independent {judge} judge (Ticket 18). Same answers, claims, premises.\n")
    p(f"{'Pipeline':<30}{'DeBERTa CSR [95% CI]':<26}{judge + ' CSR [95% CI]':<26}{'gold-URL agreement':>18}")
    p("-" * 100)
    for a, label in ARMS.items():
        if not old[a] or not new[a]:
            continue
        cells = []
        for S in (old[a], new[a]):
            m, lo, hi = boot_mean_ci(np.array([r["csr"] for r in S.values() if r.get("csr") is not None]))
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        # per-question numerator agreement (same denominator by construction)
        qs = [q for q in new[a] if new[a][q]["csr_denominator"]]
        agree = sum(old[a][q]["csr_numerator"] == new[a][q]["csr_numerator"] for q in qs)
        p(f"{label:<30}{cells[0]:<26}{cells[1]:<26}{agree:>10}/{len(qs)} Qs")
    p(f"\n{'Paired comparison (B − A)':<58}{'DeBERTa Δ [CI]':<30}{judge + ' Δ [CI]':<30}")
    p("-" * 118)
    for a, b in PAIRS:
        if not all((old[a], old[b], new[a], new[b])):
            continue
        cells = []
        for S in (old, new):
            qs = [q for q in S[a] if q in S[b] and S[a][q].get("csr") is not None and S[b][q].get("csr") is not None]
            m, lo, hi = boot_mean_ci(np.array([S[b][q]["csr"] - S[a][q]["csr"] for q in qs]))
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] {'REAL' if verdict(lo, hi).startswith('REAL') else 'n.s.'}")
        p(f"{ARMS[b] + '  −  ' + ARMS[a]:<58}{cells[0]:<30}{cells[1]:<30}")
    text = "\n".join(out); print(text)
    Path(f"results/rescore_compare_{judge}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
