"""guardrail/test_reattach_v2.py — hand-built cases for guardrail v2.

Each case pins one failure pattern from the Ticket 4 audit of v1.
Run: uv run pytest guardrail/test_reattach_v2.py -q
"""
from __future__ import annotations

from guardrail.reattach import assert_only_citations_added
from guardrail.reattach_v2 import clean_premise, reattach_citations_v2

URL_A = "https://fastapi.tiangolo.com/tutorial/testing/"
URL_B = "https://fastapi.tiangolo.com/reference/exceptions/"
URL_C = "https://fastapi.tiangolo.com/fastapi-cli/"

CHUNKS = [
    {"source_url": URL_A, "content": "FastAPI testing uses Starlette's TestClient."},
    {"source_url": URL_B, "content": "HTTPException returns HTTP error responses."},
]


def _one_claim(text: str, question: str) -> list[str]:
    return [text]


def _nli(table: dict[tuple[str, str], float]):
    """NLI stub: P(entailment) looked up by (premise substring, claim substring); else neutral."""
    seen: list[tuple[str, str]] = []

    def fn(pairs):
        seen.extend(pairs)
        out = []
        for premise, claim in pairs:
            p = next((v for (pk, ck), v in table.items() if pk in premise and ck in claim), 0.0)
            out.append(("entailment", p) if p > 0.5 else ("neutral", 0.01))
        return out

    fn.seen = seen
    return fn


def test_cites_a_fully_supported_sentence():
    answer = "FastAPI testing uses Starlette's TestClient. Something unrelated happens."
    nli = _nli({("TestClient", "TestClient"): 0.97})
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim)

    assert r.citations_added == 1
    assert f"TestClient. [{URL_A}]" in r.patched_answer
    assert_only_citations_added(r)


def test_code_block_alone_is_never_sent_to_nli():
    """v1 cited whole code blocks (#3, #4, #24); v2 drops code before NLI."""
    answer = "Example:\n\n```python\nfrom prometheus_fastapi_instrumentator import Instrumentator\n```\n"
    nli = _nli({("", ""): 0.99})   # would entail anything it is shown
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim)

    assert r.citations_added == 0
    assert not any("Instrumentator" in claim for _, claim in nli.seen)


def test_list_heading_markup_is_stripped_before_decomposition():
    """#47: '2. **Title**:' made v1's NLI say 0.997 entailment."""
    received: list[str] = []

    def decompose(text, question):
        received.append(text)
        return [text]

    answer = "2. **Create a Dependency**: You can define a dependency that handles exceptions."
    reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=_nli({}), decompose_fn=decompose)

    assert received == ["Create a Dependency: You can define a dependency that handles exceptions."]


def test_every_claim_must_be_entailed():
    answer = "FastAPI testing uses TestClient and caches every dependency forever."
    decompose = lambda t, q: ["FastAPI testing uses TestClient.", "FastAPI caches every dependency forever."]
    nli = _nli({("TestClient", "TestClient"): 0.98})   # second claim unsupported
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=decompose)

    assert r.citations_added == 0
    assert r.candidates == []


def test_below_threshold_is_recorded_but_not_inserted():
    answer = "FastAPI testing uses Starlette's TestClient."
    nli = _nli({("TestClient", "TestClient"): 0.80})
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim, threshold=0.9)

    assert r.citations_added == 0
    assert r.patched_answer == answer
    assert len(r.candidates) == 1 and not r.candidates[0].inserted
    assert r.candidates[0].min_p_entail == 0.80


def test_only_the_best_chunk_is_cited():
    """v1 attached 5 urls to one sentence (q24); v2 cites the single best."""
    answer = "Some claim about the framework here."
    nli = _nli({("TestClient", "claim"): 0.93, ("HTTPException", "claim"): 0.99})
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim)

    assert r.citations_added == 1
    assert r.patched_answer.endswith(f"[{URL_B}]")


def test_premise_is_cleaned_of_html_and_anchors():
    """Console-output chunks were HTML tag soup; v2 strips tags before NLI."""
    raw = '# FastAPI CLI { #fastapi-cli }\n\n<font color="#4E9A06">fastapi</font> dev &apos;x&apos;'
    assert clean_premise(raw) == "# FastAPI CLI\n\nfastapi dev 'x'"

    nli = _nli({})
    reattach_citations_v2("Run the app with the CLI tool.", [{"source_url": URL_C, "content": raw}], "q",
                          nli_prob_fn=nli, decompose_fn=_one_claim)
    assert all("<font" not in premise for premise, _ in nli.seen)


def test_already_cited_sentence_is_left_alone():
    answer = f"FastAPI testing uses Starlette's TestClient. [{URL_A}]"
    nli = _nli({("", ""): 0.99})
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim)

    assert r.patched_answer == answer
    assert r.sentences_checked == 0


def test_markdown_structure_is_preserved_exactly():
    answer = (
        "Steps:\n\n"
        "1. FastAPI testing uses Starlette's TestClient.\n\n"
        "   ```python\n   from fastapi.testclient import TestClient\n   ```\n\n"
        "2. HTTPException returns HTTP error responses.\n"
    )
    nli = _nli({("TestClient", "TestClient"): 0.97, ("HTTPException", "HTTPException"): 0.97})
    r = reattach_citations_v2(answer, CHUNKS, "q", nli_prob_fn=nli, decompose_fn=_one_claim)

    assert r.citations_added == 2
    assert_only_citations_added(r)
    assert "```python" in r.patched_answer and "\n\n" in r.patched_answer


def test_no_chunks_is_a_noop():
    r = reattach_citations_v2("Nothing to cite.", [], "q", nli_prob_fn=_nli({}), decompose_fn=_one_claim)
    assert r.patched_answer == "Nothing to cite." and r.citations_added == 0
