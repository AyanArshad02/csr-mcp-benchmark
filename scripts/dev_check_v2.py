"""Dev check for guardrail v2 on the Ticket 4 audit items (DEV SET — never report as v2 accuracy).

For each human-labelled v1 insertion, re-run v2 on the same answer and report what v2 does with
that sentence: cites the same url, cites a different url, records a below-threshold candidate,
or cites nothing.
Run: uv run python -m scripts.dev_check_v2
"""
import json
from collections import Counter, defaultdict

import diskcache
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from guardrail.reattach import assert_only_citations_added
from guardrail.reattach_v2 import DEFAULT_THRESHOLD, make_nli_prob_fn, reattach_citations_v2
from measure.score import make_decompose_fn

load_dotenv()
cache = diskcache.Cache(".cache")
nli = make_nli_prob_fn(CrossEncoder("cross-encoder/nli-deberta-v3-base"), cache)
decompose = make_decompose_fn(OpenAI(), cache)

audit = [json.loads(l) for l in open("results/audit_sample_50_verified.jsonl")]
rows = {json.loads(l)["question_id"]: json.loads(l) for l in open("results/guardrail_langgraph_openai.jsonl") if l.strip()}

by_q = {}
for qid in sorted({a["question_id"] for a in audit}):
    r = rows[qid]
    res = reattach_citations_v2(r["answer_original"], r["retrieved_chunks"], r["question"],
                                nli_prob_fn=nli, decompose_fn=decompose)
    assert_only_citations_added(res)
    by_q[qid] = res

table = defaultdict(Counter)
detail = []
for a in audit:
    cands = [c for c in by_q[a["question_id"]].candidates if c.sentence == a["sentence"]]
    c = cands[0] if cands else None
    if c is None:
        outcome = "v2 cites nothing"
    elif not c.inserted:
        outcome = "below threshold (not cited)"
    elif c.url == a["inserted_url"]:
        outcome = "v2 cites SAME url"
    else:
        outcome = "v2 cites a DIFFERENT url"
    table[a["verdict"]][outcome] += 1
    detail.append((a["sample_id"], a["verdict"], outcome, round(c.min_p_entail, 3) if c else None))

print(f"Guardrail v2 dev check, tau={DEFAULT_THRESHOLD}  (human verdict on the v1 citation -> what v2 does)")
for v in ("CORRECT", "PARTIAL", "WRONG"):
    print(f"  {v:8s} n={sum(table[v].values()):2d}  " + ", ".join(f"{k}: {n}" for k, n in sorted(table[v].items())))
print("v1 vs v2 on these 32 answers: citations v1 =", sum(rows[q]["citations_added"] for q in by_q),
      "| v2 =", sum(r.citations_added for r in by_q.values()))
for d in detail: print("  #%-2d %-7s %-28s minP=%s" % d)
