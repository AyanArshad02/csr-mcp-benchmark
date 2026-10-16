"""guardrail/map_source_ids.py — Guardrail v3a: map [source_id] citations back to [url] (Ticket 19).

Pre-registered in DECISIONS.md (2026-10-16). The MCP server (CSR_SOURCE_IDS=1) tags every chunk with
a short `source_id`; the agent cites `[S1a2b3c]`. After the answer is written, every bracketed group
of IDs — `[S1a2b3c]`, `[S1a2b3c, Sd4e5f6]`, `[S1a2b3c; Sd4e5f6]` — becomes `[url1] [url2]`.
Deterministic, no model. IDs resolve only against the answer's OWN retrieved chunks, so the trace
check still means what it meant: an ID the agent never retrieved stays as plain text (no citation)
and is counted. IDs outside brackets are not mapped; they are counted too.

Usage:
    uv run python -m guardrail.map_source_ids IN.jsonl OUT.jsonl
The original answer is kept as `answer_raw`; per-answer counts go in `id_mapping`.
"""
from __future__ import annotations

import argparse
import json
import re
import time

_ID = r"S[0-9a-f]{6}"
_GROUP = re.compile(r"\[\s*(" + _ID + r"(?:\s*[,;]\s*" + _ID + r")*)\s*\]")
_ANY_ID = re.compile(r"\b" + _ID + r"\b")


def map_ids(answer: str, retrieved_chunks: list[dict]) -> tuple[str, dict]:
    """Return (answer with ID groups replaced by [url] tokens, counts)."""
    id_to_url = {c["source_id"]: c["source_url"] for c in retrieved_chunks
                 if c.get("source_id") and c.get("source_url")}
    stats = {"mapped": 0, "unresolved": 0, "bare": 0}

    def _group(m: re.Match) -> str:
        out = []
        for sid in re.findall(_ID, m.group(1)):
            if sid in id_to_url:
                stats["mapped"] += 1
                out.append(f"[{id_to_url[sid]}]")
            else:
                stats["unresolved"] += 1
                out.append(f"[{sid}]")   # left as text; the [url] parser ignores it
        return " ".join(out)

    in_groups = sum(len(re.findall(_ID, m.group(1))) for m in _GROUP.finditer(answer))
    stats["bare"] = len(_ANY_ID.findall(answer)) - in_groups
    return _GROUP.sub(_group, answer), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    a = ap.parse_args()
    tot = {"mapped": 0, "unresolved": 0, "bare": 0}
    secs = []
    with open(a.inp) as f, open(a.out, "w") as g:
        for line in f:
            row = json.loads(line)
            t0 = time.perf_counter()
            row["answer_raw"] = row["answer"]
            row["answer"], stats = map_ids(row["answer"], row.get("retrieved_chunks", []))
            secs.append(time.perf_counter() - t0)
            row["id_mapping"] = stats
            for k in tot:
                tot[k] += stats[k]
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"IDs mapped {tot['mapped']} | unresolved (not retrieved) {tot['unresolved']} | "
          f"bare (outside brackets) {tot['bare']} | mean mapping time {1e3 * sum(secs) / max(len(secs), 1):.3f} ms/answer")


if __name__ == "__main__":
    main()
