"""measure/repeats_report.py — Repeat runs (Ticket 15) and guardrail v3a source IDs (Ticket 19).

Pre-registered in DECISIONS.md (2026-10-16). Run 1 of each setup is the published run in results/;
runs 2-3 (and all v3a runs) are in results/repeats/ (scripts/run_repeats_and_v3a.sh).

Per run: format buckets (the rules of measure/format_compliance.py; for v3a the requested format is
a bracketed source ID), CSR strict (DeBERTa), CSR any format (DeBERTa and gpt-4o judge).
Comparisons: per run (run k of A vs run k of B) and pooled (per-question CSR averaged over the runs
of each setup), paired bootstrap as in measure/paired_tests.py.
Decision rule: a CSR difference is "robust" only if it has the same sign in every run AND the pooled
CI excludes 0. A format difference is "robust" only if it has the same direction in every run.

Usage:  uv run python -m measure.repeats_report → stdout + results/repeats_report.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from measure.paired_tests import boot_mean_ci

R, P = "results", "results/repeats"
STRICT = re.compile(r"\[(https?://[^\]\s]+)\]")
MD = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
KAPA = re.compile(r"\[\s*\[[^\]]*\]\(https?://")
IDS = re.compile(r"\[\s*S[0-9a-f]{6}(?:\s*[,;]\s*S[0-9a-f]{6})*\s*\]")
URL = re.compile(r"https?://")


def published(run: str, strict_run: str | None = None) -> dict:
    """File paths of a published run (run 1)."""
    return {"raw": f"{R}/anyformat_{run}.jsonl",
            "strict": f"{R}/scored_{strict_run or run}.jsonl",
            "any": f"{R}/scored_anyformat_{run}.jsonl",
            "judge": f"{R}/rescored_gpt-4o_anyformat_{run}.jsonl"}


def repeat(name: str, strict_suffix: str = "") -> dict:
    return {"raw": f"{P}/anyformat_{name}.jsonl",
            "strict": f"{P}/scored_{name}{strict_suffix}.jsonl",
            "any": f"{P}/scored_anyformat_{name}.jsonl",
            "judge": f"{P}/rescored_gpt-4o_anyformat_{name}.jsonl"}


# setup -> (requested format, [run 1, run 2, run 3])
SETUPS = {
    "Single-shot": ("url", [published("baseline_openai"), repeat("baseline_r2"), repeat("baseline_r3")]),
    "LangGraph": ("url", [published("langgraph_openai"), repeat("langgraph_r2"), repeat("langgraph_r3")]),
    "SDK defaults": ("url", [published("openai_agent_openai"), repeat("sdk_defaults_r2"),
                             repeat("sdk_defaults_r3")]),
    "SDK E1 temperature 0": ("url", [published("ablation_sdk_e1_temp0"), repeat("sdk_e1_temp0_r2"),
                                     repeat("sdk_e1_temp0_r3")]),
    "SDK E3 Chat Completions": ("url", [published("ablation_sdk_e3_chat_completions"),
                                        repeat("sdk_e3_chat_completions_r2"),
                                        repeat("sdk_e3_chat_completions_r3")]),
    "SDK E4 Kapa setup": ("kapa", [published("ablation_sdk_e4_kapa_full", "ablation_sdk_e4_kapa_full_norm"),
                                   repeat("sdk_e4_kapa_full_r2", "_norm"),
                                   repeat("sdk_e4_kapa_full_r3", "_norm")]),
    "v3a SDK (source IDs)": ("ids", [repeat(f"v3a_sdk_r{k}", "_mapped") for k in (1, 2, 3)]),
    "v3a LangGraph (source IDs)": ("ids", [repeat(f"v3a_langgraph_r{k}", "_mapped") for k in (1, 2, 3)]),
}
EXTRA_RUNS = {"SDK defaults": ("E0 (earlier re-run)", published("ablation_sdk_e0_original"))}

CSR_PAIRS = [("Single-shot", "SDK defaults"), ("Single-shot", "LangGraph"),
             ("Single-shot", "SDK E1 temperature 0"), ("SDK defaults", "v3a SDK (source IDs)"),
             ("LangGraph", "v3a LangGraph (source IDs)"), ("Single-shot", "v3a SDK (source IDs)")]
FORMAT_PAIRS = [("Single-shot", "SDK defaults"), ("LangGraph", "SDK defaults"),
                ("SDK E1 temperature 0", "SDK E3 Chat Completions"),
                ("SDK E1 temperature 0", "SDK E4 Kapa setup"), ("SDK defaults", "v3a SDK (source IDs)"),
                ("LangGraph", "v3a LangGraph (source IDs)")]
VIEWS = [("strict", "strict format, DeBERTa"), ("any", "any format, DeBERTa"),
         ("judge", "any format, gpt-4o judge")]


def rows(path: str) -> dict[int, dict] | None:
    p = Path(path)
    return {r["question_id"]: r for r in map(json.loads, p.open())} if p.exists() else None


def bucket(answer: str, requested: str) -> str:
    asked = {"url": STRICT, "kapa": KAPA, "ids": IDS}[requested]
    if asked.search(answer):
        return "requested"
    if MD.search(answer) or STRICT.search(answer):
        return "other link format"
    if URL.search(answer):
        return "bare URL only"
    return "no citation"


def count(run: dict, requested: str, which: str) -> int | None:
    """Answers in one bucket. 'other link format' is the published format-drift count (51, 41, 9, 20)."""
    r = rows(run["raw"])
    return None if r is None else sum(bucket(x["answer_raw"], requested) == which for x in r.values())


def mean_csr(path: str) -> float | None:
    r = rows(path)
    if r is None:
        return None
    v = [x["csr"] for x in r.values() if x.get("csr") is not None]
    return float(np.mean(v))


def diff(a: dict[int, float | None], b: dict[int, float | None]) -> np.ndarray:
    return np.array([b[q] - a[q] for q in a if a[q] is not None and b.get(q) is not None])


def per_question(path: str) -> dict[int, float | None] | None:
    r = rows(path)
    return None if r is None else {q: x.get("csr") for q, x in r.items()}


def pooled(paths: list[str]) -> dict[int, float | None]:
    runs = [per_question(p) for p in paths]
    runs = [r for r in runs if r is not None]
    out = {}
    for q in runs[0]:
        v = [r[q] for r in runs if r.get(q) is not None]
        out[q] = float(np.mean(v)) if v else None
    return out


def fmt(v: float | None) -> str:
    return "  —  " if v is None else f"{v:.3f}"


def main() -> None:
    out = ["Repeat runs (Ticket 15) and guardrail v3a source IDs (Ticket 19). DECISIONS.md 2026-10-16.",
           "gpt-4o-mini, 127 questions. Run 1 = the published run; runs 2-3 fresh (results/repeats/).", ""]

    wf: dict[str, list[int | None]] = {}
    for title, which in (("1a. Format drift: answers citing ONLY in another link format, per run",
                          "other link format"),
                         ("1b. Answers with no citation at all (includes 'the docs don't say' answers)",
                          "no citation"),
                         ("1c. Answers with bare URLs only", "bare URL only")):
        out += [title, f"{'Setup':<30}{'run 1':>7}{'run 2':>7}{'run 3':>7}   extra", "-" * 70]
        for s, (req, runs) in SETUPS.items():
            vals = [count(r, req, which) for r in runs]
            if which == "other link format":
                wf[s] = vals
            extra = ""
            if s in EXTRA_RUNS:
                label, run = EXTRA_RUNS[s]
                extra = f"{label}: {count(run, req, which)}"
            out.append(f"{s:<30}" + "".join(f"{'—' if v is None else v:>7}" for v in vals) + f"   {extra}")
        out.append("")
    out.append("Requested format: [url]; Kapa's [[title](url)] for E4; a bracketed source ID for v3a.")

    unres = []
    for s in ("v3a SDK (source IDs)", "v3a LangGraph (source IDs)"):
        for k, run in enumerate(SETUPS[s][1], 1):
            m = rows(run["strict"].replace("scored_", ""))
            if m:
                tot = {key: sum(x["id_mapping"][key] for x in m.values()) for key in ("mapped", "unresolved", "bare")}
                unres.append(f"  {s} run {k}: IDs mapped {tot['mapped']}, unresolved (never retrieved) "
                             f"{tot['unresolved']}, outside brackets {tot['bare']}")
    if unres:
        out += ["", "v3a ID mapping:"] + unres

    out += ["", "2. Format drift (1a) differences: direction in every run (robust = same direction in all runs)"]
    for a, b in FORMAT_PAIRS:
        pa, pb = wf[a], wf[b]
        ks = [k for k in range(3) if pa[k] is not None and pb[k] is not None]
        if not ks:
            continue
        signs = {np.sign(pb[k] - pa[k]) for k in ks}
        d = ", ".join(f"{pa[k]}→{pb[k]}" for k in ks)
        verdict = "robust" if len(signs) == 1 and 0 not in signs and len(ks) == 3 else "NOT robust"
        out.append(f"  {a} → {b}: {d}   [{verdict}{'' if len(ks) == 3 else ', incomplete'}]")

    out += ["", "3. Mean CSR per run"]
    for key, label in VIEWS:
        out.append(f"  {label}")
        for s, (_, runs) in SETUPS.items():
            vals = [mean_csr(r[key]) for r in runs]
            got = [v for v in vals if v is not None]
            spread = f"   mean {np.mean(got):.3f}, range {min(got):.3f}–{max(got):.3f}" if got else ""
            extra = ""
            if s in EXTRA_RUNS:
                extra = f"   E0 {fmt(mean_csr(EXTRA_RUNS[s][1][key]))}"
            out.append(f"    {s:<28}" + "".join(f"{fmt(v):>8}" for v in vals) + spread + extra)

    out += ["", "4. Paired CSR differences (B − A): per run, and pooled over runs",
            "   robust = same sign in every run AND pooled 95% CI excludes 0"]
    for key, label in VIEWS:
        out.append(f"  {label}")
        for a, b in CSR_PAIRS:
            ra, rb = SETUPS[a][1], SETUPS[b][1]
            per = []
            for k in range(3):
                qa, qb = per_question(ra[k][key]), per_question(rb[k][key])
                per.append(None if qa is None or qb is None else diff(qa, qb).mean())
            avail = [k for k in range(3) if per[k] is not None]
            if not avail:
                continue
            d = diff(pooled([ra[k][key] for k in avail]), pooled([rb[k][key] for k in avail]))
            m, lo, hi = boot_mean_ci(d)
            signs = {np.sign(per[k]) for k in avail}
            robust = len(avail) == 3 and len(signs) == 1 and 0 not in signs and (lo > 0 or hi < 0)
            runs_s = " ".join(f"{per[k]:+.3f}" if per[k] is not None else "  —  " for k in range(3))
            out.append(f"    {b:<28} − {a:<22} runs [{runs_s}]  pooled {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                       f"  {'ROBUST' if robust else 'not robust'}{'' if len(avail) == 3 else ' (incomplete)'}")

    text = "\n".join(out)
    print(text)
    Path("results/repeats_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
