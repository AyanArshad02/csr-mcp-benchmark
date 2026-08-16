"""guardrail/test_reattach.py — hand-built cases for the attribution guardrail.

Run: uv run pytest guardrail/test_reattach.py -q
"""
from __future__ import annotations

import pytest

from guardrail.reattach import (
    _is_citable,
    _sentences_from_text,
    assert_only_citations_added,
    reattach_citations,
)

URL_A = "https://fastapi.tiangolo.com/tutorial/testing/"
URL_B = "https://fastapi.tiangolo.com/reference/exceptions/"

CHUNKS = [
    {"source_url": URL_A, "content": "FastAPI testing uses Starlette's TestClient."},
    {"source_url": URL_B, "content": "HTTPException returns HTTP error responses."},
]


def _nli_only(*entailed_urls: str):
    """NLI stub: entailment iff the premise is one of the given URLs' content."""
    wanted = {c["content"] for c in CHUNKS if c["source_url"] in entailed_urls}
    return lambda premise, hypothesis: "entailment" if premise in wanted else "neutral"


# --- Segmentation -----------------------------------------------------------

def test_list_marker_is_not_a_sentence():
    """Regression: "2. Foo" must not split into the fragment "2."."""
    assert _sentences_from_text("2. WebSocketException is used here.") == [
        "2. WebSocketException is used here."
    ]


def test_ordinary_sentences_still_split():
    assert _sentences_from_text("One thing here. Another thing there.") == [
        "One thing here.",
        "Another thing there.",
    ]


def test_abbreviation_does_not_split():
    assert len(_sentences_from_text("Use a client, e.g. TestClient, for tests.")) == 1


@pytest.mark.parametrize("frag", ["2.", "-", "###", "1)", "   ", "```"])
def test_fragments_are_not_citable(frag):
    assert not _is_citable(frag)


def test_prose_is_citable():
    assert _is_citable("FastAPI uses Pydantic for validation.")


# --- Reattachment -----------------------------------------------------------

def test_inserts_citation_after_uncited_sentence():
    answer = "FastAPI testing uses Starlette's TestClient. Something unrelated happens."

    # Entail only the first sentence, so the count is unambiguous.
    def nli(premise: str, hypothesis: str) -> str:
        supports = premise == CHUNKS[0]["content"] and "TestClient" in hypothesis
        return "entailment" if supports else "neutral"

    result = reattach_citations(answer, CHUNKS, nli_fn=nli)

    assert result.citations_added == 1
    assert f"TestClient. [{URL_A}]" in result.patched_answer
    assert result.decisions[0].url == URL_A


def test_does_not_recite_an_already_cited_sentence():
    answer = f"FastAPI testing uses Starlette's TestClient. [{URL_A}]"
    result = reattach_citations(answer, CHUNKS, nli_fn=_nli_only(URL_A))

    assert result.citations_added == 0
    assert result.patched_answer == answer


def test_list_markers_receive_no_citations():
    """Regression: the bare "1."/"2." markers must never be cited."""
    answer = (
        "Exceptions you can raise:\n\n"
        "1. FastAPI testing uses Starlette's TestClient.\n\n"
        "2. HTTPException returns HTTP error responses.\n"
    )
    result = reattach_citations(answer, CHUNKS, nli_fn=_nli_only(URL_A, URL_B))

    assert f"1. [{URL_A}]" not in result.patched_answer
    assert f"2. [{URL_B}]" not in result.patched_answer
    assert result.sentences_skipped == 0  # markers were never separate sentences


def test_markdown_structure_is_preserved_exactly():
    answer = (
        "Steps:\n\n"
        "1. FastAPI testing uses Starlette's TestClient.\n\n"
        "   ```python\n   from fastapi.testclient import TestClient\n   ```\n\n"
        "2. HTTPException returns HTTP error responses.\n"
    )
    result = reattach_citations(answer, CHUNKS, nli_fn=_nli_only(URL_A, URL_B))

    assert result.citations_added > 0
    assert_only_citations_added(result)          # no wording/whitespace loss
    assert "\n\n" in result.patched_answer       # blank lines survived
    assert "```python" in result.patched_answer  # code fence survived


def test_multiple_supporting_chunks_all_cited():
    answer = "Some uncited claim about the framework here."
    result = reattach_citations(answer, CHUNKS, nli_fn=_nli_only(URL_A, URL_B))

    assert result.citations_added == 2
    assert result.sentences_covered == 1
    assert {d.url for d in result.decisions} == {URL_A, URL_B}


def test_no_chunks_is_a_noop():
    answer = "Nothing to cite against."
    result = reattach_citations(answer, [], nli_fn=_nli_only(URL_A))

    assert result.patched_answer == answer
    assert result.citations_added == 0
