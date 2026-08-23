"""measure/paired_tests.py — Is the CSR drop real? Paired bootstrap tests.

All pipelines answer the SAME 127 questions, so comparing two arms by checking whether their
marginal 95% CIs overlap throws away the pairing and is badly under-powered. The correct test
resamples the per-question DIFFERENCE. A difference CI that excludes zero is a real effect.

Both are reported: marginal CIs (what a casual reader will eyeball) and paired CIs (the actual
test). Where they disagree, the paired result is the correct one.

Usage:
    uv run python -m measure.paired_tests
Output: stdout + results/paired_tests.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PIPELINES = [
    "baseline_openai", "langgraph_openai", "openai_agent_openai",
    "baseline_bedrock", "langgraph_bedrock",
]
N_BOOT, SEED = 10_000, 42


def load(prefix: str) -> dict[str, dict]:
    return {
        p: {r["question_id"]: r for r in
            (json.loads(l) for l in open(f"results/{prefix}_{p}.jsonl") if l.strip())}
        for p in PIPELINES
    }


def boot_mean_ci(d: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def verdict(lo: float, hi: float) -> str:
    return "REAL (CI excludes 0)" if (lo > 0 or hi < 0) else "not significant"


def cross_pipeline(S: dict, a: str, b: str, multi: bool) -> tuple:
    qs = [q for q in S[a]
          if S[a][q].get("csr") is not None and S[b][q].get("csr") is not None
          and (not multi or S[a][q].get("q_type") == "multi-source")]
    d = np.array([S[b][q]["csr"] - S[a][q]["csr"] for q in qs])
    return (len(qs), *boot_mean_ci(d))


def guardrail_lift(G: dict, p: str, multi: bool) -> tuple:
    qs = [q for q, r in G[p].items()
          if r.get("orig_csr") is not None and r.get("patched_csr") is not None
          and (not multi or r.get("q_type") == "multi-source")]
    d = np.array([G[p][q]["patched_csr"] - G[p][q]["orig_csr"] for q in qs])
    return (len(qs), *boot_mean_ci(d))


def main() -> None:
    S, G = load("scored"), load("guardrail")
    out: list[str] = [
        "Paired bootstrap tests — is the CSR difference real?",
        f"n_boot={N_BOOT}  seed={SEED}  percentile method, paired on question_id",
        "",
        "=== A — CSR drop: baseline vs agent pipelines ===",
        f"{'Comparison':<40}{'n':>5}{'mean d':>9}{'95% CI':>21}  verdict",
        "-" * 96,
    ]
    comparisons = [
        ("Baseline -> LangGraph    gpt-4o-mini", "baseline_openai", "langgraph_openai"),
        ("Baseline -> OpenAI Agent gpt-4o-mini", "baseline_openai", "openai_agent_openai"),
        ("Baseline -> LangGraph    claude-s-4-6", "baseline_bedrock", "langgraph_bedrock"),
    ]
    for multi, tag in [(False, ""), (True, "  [multi-source]")]:
        for label, a, b in comparisons:
            n, m, lo, hi = cross_pipeline(S, a, b, multi)
            out.append(f"{label + tag:<40}{n:>5}{m:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}")
        out.append("")

    out += [
        "=== B — Guardrail lift: before vs after, same answers ===",
        f"{'Pipeline':<40}{'n':>5}{'mean d':>9}{'95% CI':>21}  verdict",
        "-" * 96,
    ]
    labels = ["Baseline      gpt-4o-mini", "LangGraph     gpt-4o-mini",
              "OpenAI Agent  gpt-4o-mini", "Baseline      claude-s-4-6",
              "LangGraph     claude-s-4-6"]
    for multi, tag in [(False, ""), (True, "  [multi-source]")]:
        for p, label in zip(PIPELINES, labels):
            n, m, lo, hi = guardrail_lift(G, p, multi)
            out.append(f"{label + tag:<40}{n:>5}{m:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}")
        out.append("")

    text = "\n".join(out)
    print(text)
    Path("results/paired_tests.txt").write_text(text)
    print("Saved to results/paired_tests.txt")


if __name__ == "__main__":
    main()
