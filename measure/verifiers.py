"""measure/verifiers.py — Candidate citation verifiers for the benchmark in Ticket 17.

Every verifier answers one question: does this chunk support this sentence? Each returns a
`Judgement(score in [0, 1], supported: bool)` and caches on disk, so re-running is free.

Verifiers:
  deberta_raw      v1's check: NLI(chunk, raw sentence), argmax = entailment
  deberta_v2       v2's check: cleaned chunk vs decomposed prose claims, all argmax-entailed, min P
  minicheck        MiniCheck-Flan-T5-Large (Tang et al., 2024), re-implemented from the official
                   code (github.com/Liyan06/MiniCheck @ b58b9fa, minicheck/inference.py):
                   input "predict: " + chunk + </s> + claim, one decoder step, softmax over token
                   ids [3, 209] (no / yes), max over ~500-word chunks, supported if P > 0.5.
                   Deviation: sentence splitting for long chunks uses a regex, not nltk.
  llm_judge        an LLM given the same rubric the human labeller used
  symbols_present  deterministic: every code identifier named in the sentence appears in the chunk
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from guardrail.reattach_v2 import _code_mask, clean_premise, prose_of


@dataclass
class Judgement:
    score: float      # higher = more supported
    supported: bool   # the verifier's own decision at its default rule


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# DeBERTa NLI (uses guardrail v2's batched, cached probability function)
# ---------------------------------------------------------------------------

def deberta_raw(nli_prob_fn, chunk: str, sentence: str) -> Judgement:
    label, p = nli_prob_fn([(chunk, sentence)])[0]
    return Judgement(p, label == "entailment")


def deberta_v2(nli_prob_fn, claims: list[str], chunk: str, threshold: float = 0.9) -> Judgement:
    results = nli_prob_fn([(clean_premise(chunk), c) for c in claims])
    all_entailed = all(label == "entailment" for label, _ in results)
    score = min(p for _, p in results) if all_entailed else 0.0
    return Judgement(score, all_entailed and score >= threshold)


def sentence_prose(sentence: str) -> str:
    """The v2 prose of a stand-alone sentence (code and Markdown removed)."""
    return prose_of(sentence, 0, len(sentence), _code_mask(sentence))


# ---------------------------------------------------------------------------
# MiniCheck-Flan-T5-Large
# ---------------------------------------------------------------------------

class MiniCheckFT5:
    CKPT = "lytang/MiniCheck-Flan-T5-Large"
    CHUNK_WORDS, MAX_LEN = 500, 2048
    NO_ID, YES_ID = 3, 209   # as hard-coded in the official inference code

    def __init__(self, cache, device: str | None = None, dtype: str = "bfloat16"):
        """dtype: "float32" matches the official code; "bfloat16" halves memory (~1.6 GB), needed
        on an 8 GB machine where float32 swapped heavily. (float16 is avoided: T5 overflows in it.)"""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.cache, self.dtype = torch, cache, dtype
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.CKPT)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.CKPT, dtype=getattr(torch, dtype)).to(self.device).eval()

    def _doc_chunks(self, doc: str) -> list[str]:
        sents: list[str] = []
        for block in doc.split("\n"):
            sents.extend(s for s in re.split(r"(?<=[.!?])\s+", block) if s)
            sents.append("\n")
        chunks, cur, n = [], [], 0
        for s in sents:
            w = len(s.split())
            if n + w > self.CHUNK_WORDS and cur:
                chunks.append(" ".join(cur)); cur, n = [s], w
            else:
                cur.append(s); n += w
        if cur:
            chunks.append(" ".join(cur))
        chunks = [c.replace(" \n ", "\n").strip() for c in chunks]
        return [c for c in chunks if c] or [""]

    def __call__(self, chunk: str, claim: str) -> Judgement:
        # float32 keeps the original key so earlier cached scores stay valid
        prefix = "minicheck-ft5:" if self.dtype == "float32" else f"minicheck-ft5-{self.dtype}:"
        key = prefix + _key(chunk, claim)
        cached = self.cache.get(key)
        if cached is None:
            probs = []
            for piece in self._doc_chunks(chunk):
                text = "predict: " + self.tok.eos_token.join([piece, claim])
                # Pad to a fixed bucket (512/1024/1536/2048): Apple MPS recompiles for every new
                # input length (one call took 329 s); padding is masked, so scores are unchanged.
                n = len(self.tok(text, truncation=True, max_length=self.MAX_LEN)["input_ids"])
                bucket = min(self.MAX_LEN, -(-n // 512) * 512)
                enc = self.tok([text], max_length=bucket, truncation=True, padding="max_length",
                               return_tensors="pt").to(self.device)
                dec = self.torch.zeros((1, 1), dtype=self.torch.long, device=self.device)
                with self.torch.no_grad():
                    logits = self.model(**enc, decoder_input_ids=dec).logits.squeeze(1)
                pair = logits[:, [self.NO_ID, self.YES_ID]].float().cpu()
                probs.append(float(self.torch.softmax(pair, dim=-1)[0, 1]))
            cached = max(probs)
            self.cache.set(key, cached)
        return Judgement(cached, cached > 0.5)


# ---------------------------------------------------------------------------
# LLM judge (same rubric as the human labelling page)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are auditing citations in an AI-generated answer about the FastAPI framework.

Decide whether the CHUNK supports the SENTENCE it was cited for. Use only the chunk text: not
your own knowledge of FastAPI, not the live website, not other documents. Ignore whether the
sentence is a good answer to anything; judge only the citation.

- CORRECT: the chunk clearly supports everything the sentence claims.
- PARTIAL: the chunk supports some of what the sentence claims, but not all of it.
- WRONG: the chunk supports none of it. Being on the same topic is not support.

Code in the sentence counts as a claim: the chunk must show that code or that pattern. Filler
like "Here is an example:" carries no claim. If unsure between two verdicts, pick the stricter.

SENTENCE:
<<<
{sentence}
>>>

CHUNK:
<<<
{chunk}
>>>

Reply with JSON only: {{"verdict": "CORRECT" | "PARTIAL" | "WRONG", "reason": "<one sentence>"}}"""

_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


def llm_judge(chat_fn, chunk: str, sentence: str) -> tuple[Judgement, str, str]:
    """chat_fn(messages) -> str (a strategy's cached .chat at temperature 0)."""
    raw = chat_fn([{"role": "user", "content": JUDGE_PROMPT.format(sentence=sentence, chunk=chunk)}])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in _SCORE:
        verdict = "WRONG"   # unparseable = not supported (conservative), counted in the report
    return Judgement(_SCORE[verdict], verdict == "CORRECT"), verdict, str(data.get("reason", ""))


# ---------------------------------------------------------------------------
# Code-symbol presence (deterministic add-on)
# ---------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)
# identifiers worth checking: snake_case, dotted.names, CamelCase, or call-like name()
_IDENT = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Za-z_]\w*\.[A-Za-z_][\w.]*|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")


def code_symbols(sentence: str) -> set[str]:
    code = " ".join(_INLINE_CODE.findall(sentence) + _FENCE.findall(sentence))
    return set(_IDENT.findall(code))


def symbols_present(chunk: str, sentence: str) -> Judgement | None:
    """None when the sentence names no code identifiers (the check does not apply)."""
    syms = code_symbols(sentence)
    if not syms:
        return None
    found = sum(s in chunk for s in syms) / len(syms)
    return Judgement(found, found == 1.0)
