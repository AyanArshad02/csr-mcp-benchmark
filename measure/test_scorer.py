"""measure/test_scorer.py — 5 hand-built unit tests for the CSR scorer.

Each test uses deterministic mock NLI and decomposition functions (no model downloads).
Run with: uv run pytest measure/test_scorer.py -v

Test cases:
  T1 — Perfect answer: citation correct + grounded → recall=1, precision=1, CSR=1
  T2 — Missing citation: claim made with no source → recall=0, CSR=0
  T3 — Wrong source: URL cited but chunk doesn't entail claim → recall=0, CSR=0
  T4 — Phantom citation: URL cited but never retrieved → Check 2 fails, CSR=0
  T5 — Over-citing: two citations where one suffices → precision < 1, recall=1, CSR=1
"""

from __future__ import annotations

import pytest

from measure.score import parse_cited_units, score_answer, AnswerScore

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

URL_A = "https://fastapi.tiangolo.com/tutorial/body/"
URL_B = "https://fastapi.tiangolo.com/tutorial/query-params/"
URL_C = "https://fastapi.tiangolo.com/tutorial/path-params/"

CHUNK_A = "FastAPI uses Pydantic models to declare request bodies."
CHUNK_B = "Query parameters are declared as function arguments with default values."
CHUNK_C = "Path parameters are declared using Python type annotations in the path."


def _nli_real(supported_pairs: set[tuple[str, str]]):
    """Mock NLI: returns 'entailment' iff (premise, hypothesis) in supported_pairs."""
    def _fn(premise: str, hypothesis: str) -> str:
        return "entailment" if (premise, hypothesis) in supported_pairs else "neutral"
    return _fn


def _decompose_identity(text: str, question: str) -> list[str]:
    """Mock decomposition: treat each sentence as already atomic (identity)."""
    return [text]


QUESTION = "How does FastAPI handle request bodies?"


# ---------------------------------------------------------------------------
# parse_cited_units — parser unit tests (sanity checks)
# ---------------------------------------------------------------------------

def test_parse_single_citation():
    answer = f"FastAPI uses Pydantic. [{URL_A}]"
    units = parse_cited_units(answer)
    assert len(units) == 1
    text, urls = units[0]
    assert "Pydantic" in text
    assert urls == [URL_A]


def test_parse_multi_citation_on_one_sentence():
    answer = f"FastAPI uses Pydantic and Starlette. [{URL_A}] [{URL_B}]"
    units = parse_cited_units(answer)
    assert len(units) == 1
    _, urls = units[0]
    assert URL_A in urls and URL_B in urls


def test_parse_uncited_sentence():
    answer = f"FastAPI uses Pydantic. [{URL_A}] This sentence has no citation."
    units = parse_cited_units(answer)
    assert len(units) == 2
    _, urls_second = units[1]
    assert urls_second == []


# ---------------------------------------------------------------------------
# T1 — Perfect answer
# ---------------------------------------------------------------------------

def test_t1_perfect_answer():
    """All citations correct + grounded. Expect recall=1, precision=1, F1=1, CSR=1."""
    answer = f"FastAPI uses Pydantic models to declare request bodies. [{URL_A}]"

    retrieved_chunks = [{"source_url": URL_A, "content": CHUNK_A}]
    gold_source_urls = [URL_A]

    nli = _nli_real({(CHUNK_A, "FastAPI uses Pydantic models to declare request bodies.")})

    result = score_answer(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        gold_source_urls=gold_source_urls,
        question=QUESTION,
        nli_fn=nli,
        decompose_fn=_decompose_identity,
    )

    assert result.citation_recall == pytest.approx(1.0), f"recall={result.citation_recall}"
    assert result.citation_precision == pytest.approx(1.0), f"precision={result.citation_precision}"
    assert result.citation_f1 == pytest.approx(1.0), f"f1={result.citation_f1}"
    assert result.csr == pytest.approx(1.0), f"csr={result.csr}"
    assert result.csr_numerator == 1
    assert result.csr_denominator == 1
    assert result.error_counts["dropped"] == 0
    assert result.error_counts["phantom"] == 0


# ---------------------------------------------------------------------------
# T2 — Missing citation (uncited claim)
# ---------------------------------------------------------------------------

def test_t2_missing_citation():
    """Claim made with no [url]. recall=0 (nothing to entail), CSR=0 (gold URL never cited)."""
    answer = "FastAPI uses Pydantic models to declare request bodies."  # no [url]

    retrieved_chunks = [{"source_url": URL_A, "content": CHUNK_A}]
    gold_source_urls = [URL_A]

    # NLI shouldn't even be called since there's no citation
    nli = _nli_real(set())

    result = score_answer(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        gold_source_urls=gold_source_urls,
        question=QUESTION,
        nli_fn=nli,
        decompose_fn=_decompose_identity,
    )

    assert result.citation_recall == pytest.approx(0.0), f"recall={result.citation_recall}"
    assert result.csr == pytest.approx(0.0), f"csr={result.csr}"
    assert result.csr_denominator == 1  # URL_A was retrieved
    assert result.csr_numerator == 0
    assert result.error_counts["dropped"] == 1  # gold URL retrieved but never cited


# ---------------------------------------------------------------------------
# T3 — Wrong source (URL cited but chunk doesn't entail claim)
# ---------------------------------------------------------------------------

def test_t3_wrong_source():
    """Agent cites URL_B for a claim that URL_B doesn't support. recall=0, CSR=0."""
    claim_text = "FastAPI uses Pydantic models to declare request bodies."
    answer = f"{claim_text} [{URL_B}]"  # wrong URL for this claim

    retrieved_chunks = [
        {"source_url": URL_A, "content": CHUNK_A},
        {"source_url": URL_B, "content": CHUNK_B},
    ]
    gold_source_urls = [URL_A]

    # URL_B chunk does NOT entail the claim (NLI returns neutral)
    nli = _nli_real(set())  # no pairs → always neutral

    result = score_answer(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        gold_source_urls=gold_source_urls,
        question=QUESTION,
        nli_fn=nli,
        decompose_fn=_decompose_identity,
    )

    assert result.citation_recall == pytest.approx(0.0), f"recall={result.citation_recall}"
    # gold URL_A was retrieved but never cited → dropped
    assert result.error_counts["dropped"] == 1
    # URL_B was cited and retrieved, but NLI failed → wrong_attributed for URL_A
    assert result.csr == pytest.approx(0.0), f"csr={result.csr}"
    assert result.csr_denominator == 1  # only URL_A is gold and retrieved
    assert result.csr_numerator == 0


# ---------------------------------------------------------------------------
# T4 — Phantom citation (URL cited but never retrieved)
# ---------------------------------------------------------------------------

def test_t4_phantom_citation():
    """Agent cites URL_C which was never in the tool-call trace. Check 2 fails, CSR=0."""
    claim_text = "FastAPI uses Pydantic models to declare request bodies."
    answer = f"{claim_text} [{URL_C}]"  # URL_C was never retrieved

    # Trace only has URL_A — URL_C is phantom
    retrieved_chunks = [{"source_url": URL_A, "content": CHUNK_A}]
    gold_source_urls = [URL_A]

    nli = _nli_real(set())  # doesn't matter — URL_C has no content to check

    result = score_answer(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        gold_source_urls=gold_source_urls,
        question=QUESTION,
        nli_fn=nli,
        decompose_fn=_decompose_identity,
    )

    # The single claim has only a phantom citation — NLI can't run, so neutral
    assert result.citation_recall == pytest.approx(0.0)

    # URL_C should be in phantom_urls for the claim
    assert len(result.claims) == 1
    assert URL_C in result.claims[0].phantom_urls

    # Phantom counter should be 1
    assert result.error_counts["phantom"] == 1

    # Gold URL_A was retrieved but cited with wrong URL (URL_C) → dropped
    assert result.error_counts["dropped"] == 1
    assert result.csr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T5 — Over-citing (two citations where one alone suffices)
# ---------------------------------------------------------------------------

def test_t5_over_citing():
    """Agent cites both URL_A and URL_B for a claim; URL_A alone suffices.
    Precision < 1 (URL_B is unnecessary), but recall = 1 and CSR = 1."""
    claim_text = "FastAPI uses Pydantic models to declare request bodies."
    answer = f"{claim_text} [{URL_A}] [{URL_B}]"  # URL_B is unnecessary

    retrieved_chunks = [
        {"source_url": URL_A, "content": CHUNK_A},
        {"source_url": URL_B, "content": CHUNK_B},
    ]
    gold_source_urls = [URL_A]

    # URL_A alone entails the claim; URL_B alone does NOT
    # Full set (A+B concatenated) also entails
    full_premise = CHUNK_A + "\n\n" + CHUNK_B
    supported = {
        (CHUNK_A, claim_text),        # URL_A alone → entailment
        (full_premise, claim_text),   # full set → entailment
    }
    nli = _nli_real(supported)
    # URL_B alone is neutral (not in supported), rest-without-B = CHUNK_A which IS supported
    # → URL_B is NOT necessary (removing it, CHUNK_A still entails) → URL_B is imprecise
    # But URL_A alone entails → URL_A is independently sufficient → URL_A is precise

    result = score_answer(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        gold_source_urls=gold_source_urls,
        question=QUESTION,
        nli_fn=nli,
        decompose_fn=_decompose_identity,
    )

    assert result.citation_recall == pytest.approx(1.0), f"recall={result.citation_recall}"

    # precision_per_url: URL_A=True (sufficient), URL_B=False (not necessary, not sufficient)
    cr = result.claims[0]
    assert cr.precision_per_url.get(URL_A) is True
    assert cr.precision_per_url.get(URL_B) is False

    # Overall precision = (1 + 0) / 2 = 0.5
    assert result.citation_precision == pytest.approx(0.5), f"precision={result.citation_precision}"

    # CSR: gold URL_A was retrieved and correctly cited on a supported claim → CSR=1
    assert result.csr == pytest.approx(1.0), f"csr={result.csr}"
    assert result.csr_numerator == 1
    assert result.csr_denominator == 1
