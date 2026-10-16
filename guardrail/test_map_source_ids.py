"""guardrail/test_map_source_ids.py — hand-built cases for guardrail v3a's ID mapping.

Run: uv run pytest guardrail/test_map_source_ids.py -q
"""
from __future__ import annotations

from guardrail.map_source_ids import map_ids
from measure.score import parse_cited_units

URL_A = "https://fastapi.tiangolo.com/tutorial/testing/"
URL_B = "https://fastapi.tiangolo.com/reference/exceptions/"
CHUNKS = [
    {"source_id": "S0a1b2c", "source_url": URL_A, "content": "TestClient."},
    {"source_id": "Sd3e4f5", "source_url": URL_B, "content": "HTTPException."},
]


def test_single_id_becomes_url():
    out, st = map_ids("Use TestClient. [S0a1b2c]", CHUNKS)
    assert out == f"Use TestClient. [{URL_A}]"
    assert st == {"mapped": 1, "unresolved": 0, "bare": 0}


def test_comma_and_semicolon_groups_split():
    for sep in (", ", "; ", ","):
        out, st = map_ids(f"Both. [S0a1b2c{sep}Sd3e4f5]", CHUNKS)
        assert out == f"Both. [{URL_A}] [{URL_B}]"
        assert st["mapped"] == 2


def test_unretrieved_id_is_not_a_citation():
    out, st = map_ids("Made up. [Sffffff]", CHUNKS)
    assert out == "Made up. [Sffffff]"
    assert st == {"mapped": 0, "unresolved": 1, "bare": 0}
    assert all(not urls for _, urls in parse_cited_units(out))   # the scorer sees no citation


def test_bare_id_is_counted_not_mapped():
    out, st = map_ids("See S0a1b2c for details.", CHUNKS)
    assert out == "See S0a1b2c for details."
    assert st == {"mapped": 0, "unresolved": 0, "bare": 1}


def test_mapped_answer_parses_like_a_url_citation():
    out, _ = map_ids("Use TestClient. [S0a1b2c] [Sd3e4f5]", CHUNKS)
    assert parse_cited_units(out) == [("Use TestClient.", [URL_A, URL_B])]


def test_urls_and_markdown_links_untouched():
    text = f"A. [{URL_A}] B. [docs]({URL_B})"
    assert map_ids(text, CHUNKS) == (text, {"mapped": 0, "unresolved": 0, "bare": 0})
