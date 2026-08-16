"""Bootstrap 95% CI for per-question CSR across pipeline result files.

Usage:
    uv run python -m measure.bootstrap_ci
    uv run python -m measure.bootstrap_ci --n-boot 20000

Output is printed to stdout and saved to results/confidence_intervals.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    vals: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI over vals."""
    arr = np.array(vals)
    rng = np.random.default_rng(seed)
    boots = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(arr.mean()), lo, hi


def extract_csr(rows: list[dict], key: str = "csr") -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f}  [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

PHASE2_FILES = {
    "Baseline      gpt-4o-mini":  "results/scored_baseline_openai.jsonl",
    "LangGraph     gpt-4o-mini":  "results/scored_langgraph_openai.jsonl",
    "OpenAI Agent  gpt-4o-mini":  "results/scored_openai_agent_openai.jsonl",
    "Baseline      claude-s-4-6": "results/scored_baseline_bedrock.jsonl",
    "LangGraph     claude-s-4-6": "results/scored_langgraph_bedrock.jsonl",
}

GUARDRAIL_FILES = {
    "Baseline      gpt-4o-mini":  "results/guardrail_baseline_openai.jsonl",
    "LangGraph     gpt-4o-mini":  "results/guardrail_langgraph_openai.jsonl",
    "OpenAI Agent  gpt-4o-mini":  "results/guardrail_openai_agent_openai.jsonl",
    "Baseline      claude-s-4-6": "results/guardrail_baseline_bedrock.jsonl",
    "LangGraph     claude-s-4-6": "results/guardrail_langgraph_bedrock.jsonl",
}


def table_phase2(n_boot: int) -> str:
    hdr = f"{'Pipeline':<30}  {'CSR':>5}  {'95% CI (overall)':>18}  {'n':>4}  {'CSR multi':>9}  {'95% CI (multi-source, n≈29)':>28}"
    sep = "-" * len(hdr)
    lines = ["", "=== TABLE 1 — Phase 2 CSR (canonical, post-Ticket-2) ===", hdr, sep]
    for label, path in PHASE2_FILES.items():
        rows = load(path)
        ms   = [r for r in rows if r.get("q_type") == "multi-source"]
        vals     = extract_csr(rows)
        vals_ms  = extract_csr(ms)
        m, lo, hi       = bootstrap_ci(vals,    n_boot=n_boot)
        mm, lom, him    = bootstrap_ci(vals_ms, n_boot=n_boot)
        n    = len(vals)
        n_ms = len(vals_ms)
        lines.append(
            f"{label:<30}  {m:>5.3f}  {fmt(m,lo,hi):>18}  {n:>4}"
            f"  {mm:>9.3f}  {fmt(mm,lom,him):>28}  (n={n_ms})"
        )
    return "\n".join(lines)


def table_guardrail(n_boot: int) -> str:
    hdr = f"{'Pipeline':<30}  {'Before':>6}  {'Before 95% CI':>16}  {'After':>6}  {'After 95% CI':>16}  {'Δ CSR':>6}"
    sep = "-" * len(hdr)
    lines = ["", "=== TABLE 2 — Guardrail before/after CSR (overall) ===", hdr, sep]
    for label, path in GUARDRAIL_FILES.items():
        rows  = load(path)
        orig  = extract_csr(rows, "orig_csr")
        pat   = extract_csr(rows, "patched_csr")
        mo, lo_o, hi_o = bootstrap_ci(orig, n_boot=n_boot)
        mp, lo_p, hi_p = bootstrap_ci(pat,  n_boot=n_boot)
        lines.append(
            f"{label:<30}  {mo:>6.3f}  [{lo_o:.3f}, {hi_o:.3f}]  "
            f"{mp:>6.3f}  [{lo_p:.3f}, {hi_p:.3f}]  {mp-mo:>+6.3f}"
        )
    return "\n".join(lines)


def table_guardrail_multi(n_boot: int) -> str:
    hdr = f"{'Pipeline':<30}  {'Before':>6}  {'Before 95% CI':>16}  {'After':>6}  {'After 95% CI':>16}  {'Δ CSR':>6}"
    sep = "-" * len(hdr)
    lines = [
        "",
        "=== TABLE 3 — Guardrail before/after CSR (multi-source only, n≈29) ===",
        "NOTE: Wide CIs expected with n≈29 — this is honest, not a bug.",
        hdr, sep,
    ]
    for label, path in GUARDRAIL_FILES.items():
        rows = [r for r in load(path) if r.get("q_type") == "multi-source"]
        orig = extract_csr(rows, "orig_csr")
        pat  = extract_csr(rows, "patched_csr")
        mo, lo_o, hi_o = bootstrap_ci(orig, n_boot=n_boot)
        mp, lo_p, hi_p = bootstrap_ci(pat,  n_boot=n_boot)
        n = len(orig)
        lines.append(
            f"{label:<30}  {mo:>6.3f}  [{lo_o:.3f}, {hi_o:.3f}]  "
            f"{mp:>6.3f}  [{lo_p:.3f}, {hi_p:.3f}]  {mp-mo:>+6.3f}  (n={n})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs for CSR")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap iterations")
    args = parser.parse_args()

    output = "\n".join([
        "Bootstrap 95% Confidence Intervals — CSR Benchmark",
        f"n_boot={args.n_boot}  seed=42  (percentile method)",
        table_phase2(args.n_boot),
        table_guardrail(args.n_boot),
        table_guardrail_multi(args.n_boot),
        "",
    ])

    print(output)

    out_path = Path("results/confidence_intervals.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
