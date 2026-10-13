"""measure/verifier_bench.py — Which verifier agrees with humans about citations? (Ticket 17)

Data: the 115 human citation judgements.
  dev  = results/audit_sample_50_verified.jsonl   (Ticket 4: v1 insertions, LangGraph × gpt-4o-mini)
  test = results/audit_v2_heldout_verified.jsonl  (Ticket 16: v2 candidates, 4 arms, held out)
Gold = the human's BLIND FIRST CALL. Strict: CORRECT = supported. Lenient: CORRECT or PARTIAL.
Score-based verifiers also get a threshold tuned on dev (max strict F1) and applied unchanged to test.

SELECTION BIAS (state it whenever quoting these numbers): every pair was one DeBERTa-NLI already
judged "entailment" (that is how the guardrail chose them). So this measures how well a verifier
REJECTS DeBERTa's false positives; recall is measured only within DeBERTa-positive pairs.

Usage:
    uv run python -m measure.verifier_bench [--skip llm_judge,minicheck]
Output: results/verifier_bench.txt + results/verifier_bench_items.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Both models are downloaded once; load them from the local cache without network checks.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import diskcache
from dotenv import load_dotenv

from guardrail.reattach_v2 import make_nli_prob_fn
from measure.score import make_decompose_fn
from measure.verifiers import (
    MiniCheckFT5, deberta_raw, deberta_v2, llm_judge, sentence_prose, symbols_present,
)

load_dotenv()
DEV, TEST = "results/audit_sample_50_verified.jsonl", "results/audit_v2_heldout_verified.jsonl"


def load(path: str, split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["split"], r["gold"] = split, r["human_first_verdict"]
    return rows


def dev_claims(r: dict, decompose, v2_rows: dict) -> list[str] | None:
    """Claims for a dev (Ticket 4) sentence: from the decomposition cache, else from guardrail v2's
    recorded candidate for the same sentence. None = v2 found no all-entailed chunk at all."""
    try:
        return decompose(sentence_prose(r["sentence"]), r["question"])
    except Exception:
        pass
    for c in v2_rows.get(r["question_id"], {}).get("candidates", []):
        if c["sentence"] == r["sentence"]:
            return c["claims"]
    return None


def metrics(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(not p and g for p, g in zip(pred, gold)); tn = len(pred) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    n = len(pred); po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "prec": prec, "rec": rec, "f1": f1, "acc": po, "kappa": kappa}


def tune(scores: list[float], gold: list[bool]) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in sorted(set(scores)):
        f1 = metrics([s >= t for s in scores], gold)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma list: minicheck,llm_judge")
    skip = set(filter(None, ap.parse_args().skip.split(",")))

    rows = load(DEV, "dev") + load(TEST, "test")
    cache = diskcache.Cache(".cache")
    from sentence_transformers import CrossEncoder
    # DeBERTa scores are cached from earlier runs; keeping it on CPU leaves the GPU to MiniCheck
    # (both on Apple MPS at once caused memory swapping and ~45 s per item).
    nli = make_nli_prob_fn(CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cpu"), cache)
    decompose = make_decompose_fn(None, cache)   # cache-only: a miss raises, never calls the API
    v2_lg = {json.loads(l)["question_id"]: json.loads(l) for l in open("results/guardrail_v2_langgraph_openai.jsonl")}

    mc = None if "minicheck" in skip else MiniCheckFT5(cache)
    judge = None
    if "llm_judge" not in skip:
        from pipelines.strategy.bedrock_s import BedrockStrategy
        bedrock = BedrockStrategy(cache=cache)
        judge = lambda msgs: bedrock.chat(msgs, temperature=0.0, max_tokens=300)

    for i, r in enumerate(rows, 1):
        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] scored", flush=True)
        chunk, sent = r["chunk_content"], r["sentence"]
        claims = r["claims"] if r["split"] == "test" else dev_claims(r, decompose, v2_lg)
        prose = sentence_prose(sent) or sent
        v = {"deberta_raw": deberta_raw(nli, chunk, sent),
             "deberta_v2": deberta_v2(nli, claims, chunk) if claims else None}
        if mc:
            v["minicheck_sentence"] = mc(chunk, sent)
            v["minicheck_prose"] = mc(chunk, prose)
        if judge:
            j, verdict, reason = llm_judge(judge, chunk, sent)
            v["claude_judge"] = j
            r["claude_judge_verdict"], r["claude_judge_reason"] = verdict, reason
        sym = symbols_present(chunk, sent)
        r["symbols"] = None if sym is None else sym.supported
        r["v"] = {k: (None if x is None else {"score": x.score, "supported": x.supported}) for k, x in v.items()}

    names = list(rows[0]["v"])
    combos = [n for n in names if n.startswith(("minicheck", "claude"))]
    out: list[str] = []; p = out.append
    p("Verifier benchmark vs human blind first calls (Ticket 17)")
    p("dev = Ticket 4 (50 v1 insertions, 1 arm); test = v2 held-out (65, 4 arms)")
    p("SELECTION BIAS: every pair was DeBERTa-positive when sampled; this measures rejecting its false positives.\n")
    for gold_name, gold_fn in (("STRICT (CORRECT = supported)", lambda g: g == "CORRECT"),
                               ("LENIENT (CORRECT or PARTIAL = supported)", lambda g: g in ("CORRECT", "PARTIAL"))):
        p(f"=== Gold: {gold_name}")
        for split in ("dev", "test"):
            S = [r for r in rows if r["split"] == split]
            gold = [gold_fn(r["gold"]) for r in S]
            p(f"--- {split} (n={len(S)}, gold-positive {sum(gold)})")
            p(f"{'verifier':<44}{'prec':>6}{'rec':>6}{'F1':>6}{'acc':>6}{'kappa':>7}   tp/fp/fn/tn")

            def row(label, pred):
                m = metrics(pred, gold)
                p(f"{label:<44}{m['prec']:>6.2f}{m['rec']:>6.2f}{m['f1']:>6.2f}{m['acc']:>6.2f}{m['kappa']:>7.2f}"
                  f"   {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

            for name in names:
                if any(r["v"][name] is None for r in S):
                    p(f"{name:<44}  (missing for {sum(r['v'][name] is None for r in S)} items)"); continue
                row(name + " (default rule)", [r["v"][name]["supported"] for r in S])
                if name.startswith(("deberta", "minicheck")):
                    dev_rows = [r for r in rows if r["split"] == "dev" and r["v"][name] is not None]
                    t = tune([r["v"][name]["score"] for r in dev_rows], [gold_fn(r["gold"]) for r in dev_rows])
                    row(f"{name} (dev-tuned t={t:.3f})", [r["v"][name]["score"] >= t for r in S])
            for name in combos:
                row(f"{name} + symbol veto", [r["v"][name]["supported"] and r["symbols"] is not False for r in S])
            p("")
    n_sym = sum(r["symbols"] is not None for r in rows)
    p(f"symbol check applies to {n_sym}/{len(rows)} sentences (those naming code identifiers)")
    text = "\n".join(out); print(text)
    Path("results/verifier_bench.txt").write_text(text + "\n")
    with open("results/verifier_bench_items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("split", "sample_id", "question_id", "inserted_url", "gold",
                                                      "symbols", "v", "claude_judge_verdict", "claude_judge_reason")}) + "\n")


if __name__ == "__main__":
    main()
