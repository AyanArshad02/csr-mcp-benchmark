"""measure/score.py — Citation Survival Rate scorer.

Pipeline per answer:
  1. parse_cited_units()     — split answer into (sentence, [cited_urls]) pairs
  2. decompose_fn()          — LLM breaks each sentence into atomic claims
  3. nli_fn()                — NLI checks (premise=cited chunk, hypothesis=claim)
  4. score_answer()          — computes citation recall/precision/F1 (ALCE) + CSR (two checks)

All I/O functions (NLI, LLM decomposition) are injected as callables so tests can mock them.
Use make_nli_fn() / make_decompose_fn() to build the production versions.

Citation format assumed throughout: [https://some.url] immediately after each claim.
Multiple consecutive citations attach to the same preceding sentence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

NLILabel = str   # "entailment" | "neutral" | "contradiction"
NLIFn = Callable[[str, str], NLILabel]          # (premise, hypothesis) -> label
DecomposeFn = Callable[[str, str], list[str]]   # (text, question) -> [atomic claims]


@dataclass
class ClaimResult:
    claim: str
    source_text: str           # sentence this claim was derived from
    cited_urls: list[str]      # URLs cited by the source sentence
    nli_label: NLILabel        # NLI result over concatenated cited chunks
    supported: bool            # nli_label == "entailment" (Check 1)
    phantom_urls: list[str]    # cited_urls not in the retrieved trace (Check 2 fails)
    precision_per_url: dict[str, bool]  # url -> was this citation necessary?


@dataclass
class AnswerScore:
    citation_recall: float     # ALCE: fraction of claims entailed by their cited chunks
    citation_precision: float  # ALCE: fraction of citations that are necessary
    citation_f1: float
    csr: float | None          # None when denominator == 0
    csr_numerator: int
    csr_denominator: int       # |gold_source_urls ∩ retrieved_urls|
    claims: list[ClaimResult]
    retrieved_urls: list[str]
    gold_source_urls: list[str]
    error_counts: dict[str, int]  # "dropped" / "wrong_attributed" / "phantom" per gold URL


# ---------------------------------------------------------------------------
# Step 1 — Citation parser
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r'\[(https?://[^\]]+)\]')


def parse_cited_units(answer: str) -> list[tuple[str, list[str]]]:
    """Split answer into (sentence_text, [cited_urls]) pairs.

    The citation format is [url] immediately after each claim; multiple
    consecutive [url] tokens all attach to the preceding sentence.

    Returns only segments where sentence_text is non-empty.
    """
    # CITATION_RE with a capture group splits into alternating [text, url, text, url, ...]
    parts = _CITATION_RE.split(answer)
    # Even indices: text segments; odd indices: captured URL strings (no brackets)
    texts = parts[0::2]
    urls  = parts[1::2]

    units: list[tuple[str, list[str]]] = []

    for i, text in enumerate(texts):
        stripped = text.strip()
        if i < len(urls):
            url = urls[i]
            if stripped:
                units.append((stripped, [url]))
            else:
                # Whitespace-only between two citations — attach url to previous unit
                if units:
                    prev_text, prev_urls = units[-1]
                    units[-1] = (prev_text, prev_urls + [url])
        else:
            # Last text segment — no trailing citation
            if stripped:
                units.append((stripped, []))

    return units


# ---------------------------------------------------------------------------
# Step 2 — Atomic claim decomposition (FActScore-style)
# ---------------------------------------------------------------------------

def make_decompose_fn(client, cache) -> DecomposeFn:
    """Return a decomposition function backed by gpt-4o-mini with diskcache."""

    def _decompose(text: str, question: str) -> list[str]:
        key = "decompose:" + hashlib.sha256(
            (text + "\x00" + question).encode()
        ).hexdigest()

        cached = cache.get(key)
        if cached is not None:
            return cached

        prompt = (
            "Break the following text into atomic, self-contained facts.\n"
            "Rules:\n"
            "- Each fact must be minimal (one piece of information).\n"
            "- Each fact must be self-contained: resolve any pronouns to their referents.\n"
            "- Each fact must be independently verifiable.\n"
            f"Context (the question being answered): {question}\n\n"
            f"Text: {text}\n\n"
            'Return JSON: {"claims": ["claim 1", "claim 2", ...]}\n'
            "If the text is already a single atomic fact, return it as a one-element list."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            claims = data.get("claims", [])
            if not isinstance(claims, list) or not claims:
                claims = [text]
        except (json.JSONDecodeError, AttributeError):
            claims = [text]

        result = [str(c).strip() for c in claims if str(c).strip()]
        if not result:
            result = [text]

        cache.set(key, result)
        return result

    return _decompose


# ---------------------------------------------------------------------------
# Step 3 — NLI entailment check
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


def make_nli_fn(model, cache) -> NLIFn:
    """Return an NLI function backed by CrossEncoder with diskcache.

    model should be a sentence_transformers.CrossEncoder loaded with
    'cross-encoder/nli-deberta-v3-base'. Label order: 0=contradiction,
    1=entailment, 2=neutral (per the model's config.json).
    """

    def _nli(premise: str, hypothesis: str) -> NLILabel:
        key = "nli:" + hashlib.sha256(
            (premise + "\x00" + hypothesis).encode()
        ).hexdigest()

        cached = cache.get(key)
        if cached is not None:
            return cached

        scores = model.predict([(premise, hypothesis)])
        # scores shape: (1, 3) or (3,) depending on batch size
        arr = scores[0] if scores.ndim == 2 else scores
        label = _LABEL_MAP[int(arr.argmax())]

        cache.set(key, label)
        return label

    return _nli


# ---------------------------------------------------------------------------
# Step 4 — Main scorer
# ---------------------------------------------------------------------------

def _concat_chunks(urls: list[str], url_to_content: dict[str, str]) -> str:
    """Concatenate chunk content for a list of URLs, skipping phantoms."""
    parts = [url_to_content[u] for u in urls if u in url_to_content]
    return "\n\n".join(parts)


def score_answer(
    answer: str,
    retrieved_chunks: list[dict],   # [{"source_url": str, "content": str}, ...]
    gold_source_urls: list[str],
    question: str,
    *,
    nli_fn: NLIFn,
    decompose_fn: DecomposeFn,
) -> AnswerScore:
    """Score one agent answer against its retrieved chunks and gold URLs.

    retrieved_chunks: the full list of chunks the agent retrieved across ALL
        tool calls (from the logged trace). Each dict must have keys
        "source_url" and "content".
    gold_source_urls: the correct URLs from eval.jsonl for this question.
    """
    # Build url -> content lookup from trace
    url_to_content: dict[str, str] = {}
    for chunk in retrieved_chunks:
        url = chunk.get("source_url", "")
        content = chunk.get("content", "")
        if url and url not in url_to_content:
            url_to_content[url] = content

    retrieved_urls = set(url_to_content.keys())
    gold_set = set(gold_source_urls)

    # Gold URLs the agent actually retrieved (CSR denominator)
    retrieved_gold = gold_set & retrieved_urls
    csr_denominator = len(retrieved_gold)

    # --- Parse answer into (sentence, [cited_urls]) units ---
    units = parse_cited_units(answer)

    # --- Decompose each unit into atomic claims ---
    claim_results: list[ClaimResult] = []

    for sentence, cited_urls in units:
        # Decompose into atomic claims; each inherits the sentence's citations
        try:
            atomic_claims = decompose_fn(sentence, question)
        except Exception as exc:
            logger.warning("Decomposition failed for sentence %r: %s", sentence[:60], exc)
            atomic_claims = [sentence]

        # Identify phantoms (cited URLs not in retrieved trace) — Check 2
        phantom_urls = [u for u in cited_urls if u not in retrieved_urls]

        for claim in atomic_claims:
            # Check 1 — Support (NLI)
            if cited_urls:
                # Only run NLI on non-phantom citations (we need content)
                valid_urls = [u for u in cited_urls if u in url_to_content]
                if valid_urls:
                    premise = _concat_chunks(valid_urls, url_to_content)
                    try:
                        nli_label = nli_fn(premise, claim)
                    except Exception as exc:
                        logger.warning("NLI failed for claim %r: %s", claim[:60], exc)
                        nli_label = "neutral"
                else:
                    # All citations are phantoms — can't check support
                    nli_label = "neutral"
            else:
                # No citation — uncited claim
                nli_label = "neutral"

            supported = nli_label == "entailment"

            # Citation precision — leave-one-out test per cited URL
            precision_per_url: dict[str, bool] = {}
            valid_cited = [u for u in cited_urls if u in url_to_content]

            if len(valid_cited) == 1:
                # Single citation: precise iff it supports the claim
                precision_per_url[valid_cited[0]] = supported
            elif len(valid_cited) > 1:
                for url in valid_cited:
                    rest = [u for u in valid_cited if u != url]
                    rest_premise = _concat_chunks(rest, url_to_content)
                    try:
                        rest_label = nli_fn(rest_premise, claim)
                    except Exception:
                        rest_label = "neutral"
                    # Necessary if removing this URL breaks support
                    necessary = rest_label != "entailment"
                    # OR independently sufficient on its own
                    try:
                        solo_label = nli_fn(url_to_content[url], claim)
                    except Exception:
                        solo_label = "neutral"
                    precision_per_url[url] = necessary or (solo_label == "entailment")

            # Mark phantom citations as imprecise
            for url in phantom_urls:
                precision_per_url[url] = False

            claim_results.append(ClaimResult(
                claim=claim,
                source_text=sentence,
                cited_urls=cited_urls,
                nli_label=nli_label,
                supported=supported,
                phantom_urls=phantom_urls,
                precision_per_url=precision_per_url,
            ))

    # --- ALCE citation recall ---
    # Recall per claim: 1 if supported (entailed by cited chunks), else 0
    # Uncited claims (cited_urls == []) always get recall 0
    recall_scores = []
    for cr in claim_results:
        recall_scores.append(1.0 if cr.supported else 0.0)

    citation_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    # --- ALCE citation precision ---
    # Precision is defined over cited claims only (those with >= 1 citation)
    precision_scores = []
    for cr in claim_results:
        if cr.precision_per_url:
            claim_precision = sum(cr.precision_per_url.values()) / len(cr.precision_per_url)
            precision_scores.append(claim_precision)

    citation_precision = (
        sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    )

    # --- Citation F1 ---
    if citation_recall + citation_precision > 0:
        citation_f1 = (
            2 * citation_recall * citation_precision
            / (citation_recall + citation_precision)
        )
    else:
        citation_f1 = 0.0

    # --- CSR ---
    # For each gold URL that was retrieved, check if it was correctly cited
    # on at least one NLI-supported claim (Check 1 + Check 2)
    csr_numerator = 0
    for gold_url in retrieved_gold:
        # Check 2 is satisfied (gold_url is in retrieved_urls by definition)
        # Check 1: find any claim where cited_urls includes gold_url AND supported
        correctly_cited = any(
            gold_url in cr.cited_urls and cr.supported
            for cr in claim_results
        )
        if correctly_cited:
            csr_numerator += 1

    csr = csr_numerator / csr_denominator if csr_denominator > 0 else None

    # --- Error analysis per gold URL ---
    error_counts: dict[str, int] = {"dropped": 0, "wrong_attributed": 0, "phantom": 0}

    for gold_url in retrieved_gold:
        claims_citing_gold = [cr for cr in claim_results if gold_url in cr.cited_urls]
        if not claims_citing_gold:
            error_counts["dropped"] += 1
        elif not any(cr.supported for cr in claims_citing_gold):
            error_counts["wrong_attributed"] += 1
        # else: survived (counted in numerator)

    # Count phantom citations from ALL claims (any cited URL not in trace)
    for cr in claim_results:
        error_counts["phantom"] += len(cr.phantom_urls)

    return AnswerScore(
        citation_recall=citation_recall,
        citation_precision=citation_precision,
        citation_f1=citation_f1,
        csr=csr,
        csr_numerator=csr_numerator,
        csr_denominator=csr_denominator,
        claims=claim_results,
        retrieved_urls=list(retrieved_urls),
        gold_source_urls=gold_source_urls,
        error_counts=error_counts,
    )


# ---------------------------------------------------------------------------
# Batch scorer (convenience)
# ---------------------------------------------------------------------------

def score_dataset(
    rows: list[dict],
    *,
    nli_fn: NLIFn,
    decompose_fn: DecomposeFn,
) -> list[dict]:
    """Score a list of result dicts and return augmented rows.

    Each row must have:
      - "answer": str              — the agent's generated answer with [url] citations
      - "retrieved_chunks": list   — [{source_url, content}, ...]
      - "gold_source_urls": list   — from eval.jsonl
      - "question": str
    Returns the input rows extended with AnswerScore fields.
    """
    results = []
    for i, row in enumerate(rows):
        try:
            score = score_answer(
                answer=row["answer"],
                retrieved_chunks=row["retrieved_chunks"],
                gold_source_urls=row["gold_source_urls"],
                question=row["question"],
                nli_fn=nli_fn,
                decompose_fn=decompose_fn,
            )
            results.append({
                **row,
                "citation_recall": score.citation_recall,
                "citation_precision": score.citation_precision,
                "citation_f1": score.citation_f1,
                "csr": score.csr,
                "csr_numerator": score.csr_numerator,
                "csr_denominator": score.csr_denominator,
                "error_counts": score.error_counts,
            })
        except Exception as exc:
            logger.error("Scoring failed for row %d: %s", i, exc)
            results.append({**row, "citation_recall": None, "csr": None, "error": str(exc)})

    return results
