"""measure/format_compliance.py — How often does each pipeline cite in the format it was asked for?

Counts answers, no scorer involved. Requested format: `[url]` (Kapa's `[[title](url)]` for SDK E4).
Each answer falls in exactly one bucket: uses the requested format at least once; cites only in
another link format (Markdown `[text](url)`, or `[url]` for E4); only bare URLs; no URL at all.

Usage:  uv run python -m measure.format_compliance → stdout + results/format_compliance.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from measure.anyformat_report import RUNS

STRICT = re.compile(r"\[(https?://[^\]\s]+)\]")
MD = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
KAPA = re.compile(r"\[\s*\[[^\]]*\]\(https?://")
URL = re.compile(r"https?://")


def bucket(answer: str, kapa: bool) -> str:
    asked = KAPA if kapa else STRICT
    if asked.search(answer):
        return "requested"
    if MD.search(answer) or (kapa and STRICT.search(answer)):
        return "other link format"
    if URL.search(answer):
        return "bare URL only"
    return "no URL"


def main() -> None:
    out = ["Citation format compliance per answer (127 answers per run; no scorer involved)\n",
           f"{'Run':<38}{'requested format':>18}{'other link format':>19}{'bare URL only':>15}{'no URL':>8}"]
    out.append("-" * 98)
    for run, label in RUNS.items():
        rows = [json.loads(l) for l in open(f"results/anyformat_{run}.jsonl")]
        counts = {k: 0 for k in ("requested", "other link format", "bare URL only", "no URL")}
        for r in rows:
            counts[bucket(r["answer_raw"], run.endswith("e4_kapa_full"))] += 1
        out.append(f"{label:<38}{counts['requested']:>18}{counts['other link format']:>19}"
                   f"{counts['bare URL only']:>15}{counts['no URL']:>8}")
    out.append("\nE4's requested format is Kapa's [[title](url)]; for every other run it is [url].")
    # Where do the SDK's Markdown links point? (answers with no [url] citation, published run)
    rows = [json.loads(l) for l in open("results/anyformat_openai_agent_openai.jsonl")]
    total = to_retrieved = 0
    for r in rows:
        if STRICT.search(r["answer_raw"]):
            continue
        retrieved = {c["source_url"] for c in r["retrieved_chunks"]}
        for u in MD.findall(r["answer_raw"]):
            total += 1; to_retrieved += u in retrieved
    out.append(f"OpenAI SDK (published): Markdown links in answers without [url]: {total}; "
               f"pointing to a retrieved page: {to_retrieved}")
    text = "\n".join(out); print(text)
    Path("results/format_compliance.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
