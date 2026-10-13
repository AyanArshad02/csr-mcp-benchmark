"""measure/sdk_ablation_report.py — Results of the OpenAI Agents SDK mechanism ablation (Ticket 9, G2).

Comparisons are exactly those pre-registered in DECISIONS.md (2026-10-12, addendum 2026-10-13):
  E0 vs published run (replication) · E1 vs E0 · E2 vs E1 · E3 vs E1 · E4 vs E1 (and vs E0)
"Changed" = paired bootstrap 95% CI of the per-question CSR difference excludes 0 (same method,
resamples and seed as measure/paired_tests.py). Also reported, not used as the criterion:
zero-citation answers and tool calls per answer. Baseline and LangGraph (gpt-4o-mini) are shown
for context only.

Usage:  uv run python -m measure.sdk_ablation_report  → stdout + results/sdk_ablation_report.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from measure.paired_tests import boot_mean_ci, verdict

CITE = re.compile(r"\[(https?://[^\]\s]+)\]")
RUNS = {
    "published":  ("OpenAI SDK, published run (Aug)", "results/scored_openai_agent_openai.jsonl"),
    "E0":         ("E0 original config, re-run", "results/scored_ablation_sdk_e0_original.jsonl"),
    "E1":         ("E1 temperature 0", "results/scored_ablation_sdk_e1_temp0.jsonl"),
    "E2":         ("E2 E1 + Kapa tool description", "results/scored_ablation_sdk_e2_kapa_tooldesc.jsonl"),
    "E3":         ("E3 E1 + Chat Completions API", "results/scored_ablation_sdk_e3_chat_completions.jsonl"),
    "E4":         ("E4 Kapa full setup (primary)", "results/scored_ablation_sdk_e4_kapa_full_norm.jsonl"),
    "E4_inline":  ("E4 sensitivity: + inline links", "results/scored_ablation_sdk_e4_kapa_full_norm_inline.jsonl"),
    "baseline":   ("context: Baseline gpt-4o-mini", "results/scored_baseline_openai.jsonl"),
    "langgraph":  ("context: LangGraph gpt-4o-mini", "results/scored_langgraph_openai.jsonl"),
}
COMPARISONS = [("E0", "published"), ("E1", "E0"), ("E2", "E1"), ("E3", "E1"), ("E4", "E1"), ("E4", "E0"),
               ("E4_inline", "E1"), ("E1", "baseline"), ("E4", "baseline")]


def load(path: str) -> dict[int, dict]:
    return {r["question_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def main() -> None:
    S = {k: load(p) for k, (_, p) in RUNS.items() if Path(p).exists()}
    out: list[str] = []; p = out.append
    p("OpenAI Agents SDK × gpt-4o-mini — mechanism ablation (pre-registered, DECISIONS.md 2026-10-12/13)")
    p("Scorer: locked DeBERTa argmax. Paired bootstrap 10,000 resamples, seed 42.\n")
    p(f"{'Run':<36}{'n':>5}{'CSR':>7}  {'95% CI':<17}{'zero-cite':>10}{'tools/ans':>10}")
    p("-" * 85)
    for k, (label, _) in RUNS.items():
        if k not in S:
            p(f"{label:<36}  (missing)"); continue
        rows = list(S[k].values())
        csr = np.array([r["csr"] for r in rows if r.get("csr") is not None])
        m, lo, hi = boot_mean_ci(csr)
        zero = sum(not CITE.search(r.get("answer", "")) for r in rows)
        tools = np.mean([r.get("tool_call_count", 0) for r in rows]) if "tool_call_count" in rows[0] else float("nan")
        p(f"{label:<36}{len(rows):>5}{m:>7.3f}  [{lo:.3f}, {hi:.3f}]  {zero:>6}/{len(rows):<3}{tools:>9.2f}")

    p(f"\n{'Paired comparison (A − B)':<52}{'n':>5}{'Δ CSR':>8}  {'95% CI':<19}verdict")
    p("-" * 100)
    for a, b in COMPARISONS:
        if a not in S or b not in S:
            p(f"{a} − {b}: missing"); continue
        qs = [q for q in S[a] if q in S[b] and S[a][q].get("csr") is not None and S[b][q].get("csr") is not None]
        d = np.array([S[a][q]["csr"] - S[b][q]["csr"] for q in qs])
        m, lo, hi = boot_mean_ci(d)
        p(f"{RUNS[a][0] + '  −  ' + b:<52}{len(qs):>5}{m:>+8.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}")

    if "E4" in S:
        raw = [json.loads(l) for l in open("results/ablation_sdk_e4_kapa_full.jsonl")]
        kapa = sum(len(re.findall(r"\[\s*\[[^\]]*\]\(https?://", r["answer"])) for r in raw)
        p(f"\nE4 raw answers: {kapa} Kapa-format citation brackets across {len(raw)} answers")
    text = "\n".join(out); print(text)
    Path("results/sdk_ablation_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
