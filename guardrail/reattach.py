"""guardrail/reattach.py — Attribution guardrail: reattach missing citations.

Given an agent answer and the chunks the agent actually retrieved, for each
sentence that lacks a [url] citation, try to find a retrieved chunk that NLI-
supports the sentence and insert [url] immediately after it.

Zero wording changes — only [url] tags are inserted. This is enforced
structurally: sentences are located as (start, end) offsets into the original
string and citations are spliced in at those offsets, so every original
character (including newlines and Markdown indentation) is preserved verbatim.
`assert_only_citations_added()` checks this property.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from measure.score import NLIFn, _CITATION_RE


@dataclass
class InsertionDecision:
    """One (sentence, url) pair the guardrail decided to cite. Feeds the manual audit."""
    sentence: str
    url: str
    chunk_content: str


@dataclass
class ReattachResult:
    original_answer: str
    patched_answer: str
    citations_added: int        # number of [url] insertions made
    sentences_checked: int      # uncited sentences we ran NLI on
    sentences_covered: int      # uncited sentences that found at least one supporting chunk
    sentences_skipped: int      # candidates rejected as non-sentence fragments
    decisions: list[InsertionDecision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------
#
# Split on sentence-ending punctuation followed by whitespace, but NOT when the
# period belongs to a Markdown ordered-list marker ("1.", "23.") or a common
# abbreviation. Without these guards, "2. WebSocketException is ..." is split
# into the fragment "2." and the rest — and the bare "2." then gets NLI'd
# against every retrieved chunk, which spuriously entails ~21% of the time.
_SENT_BOUNDARY = re.compile(
    r"""(?<=[.!?])          # a sentence-ending mark
        (?<!\b\d[.])        # ...not "1."
        (?<!\b\d\d[.])      # ...not "12."
        (?<!\be\.g[.])
        (?<!\bi\.e[.])
        (?<!\betc[.])
        (?<!\bvs[.])
        \s+                 # followed by whitespace
    """,
    re.VERBOSE,
)

# A citable sentence needs real prose. Markdown scaffolding ("2.", "-", "###",
# "```python") carries no claim, so citing it is always a false positive.
_ALPHA_TOKEN = re.compile(r"[A-Za-z]{2,}")
_MIN_ALPHA_TOKENS = 3


def _is_citable(sentence: str) -> bool:
    """True if the sentence contains enough prose to carry a factual claim."""
    return len(_ALPHA_TOKEN.findall(sentence)) >= _MIN_ALPHA_TOKENS


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each sentence's content within `text`.

    Offsets exclude surrounding whitespace, so splicing a citation at `end`
    places it immediately after the sentence's final character.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENT_BOUNDARY.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        s, e = start + lead, end - trail
        if e > s:
            trimmed.append((s, e))
    return trimmed


def _sentences_from_text(text: str) -> list[str]:
    """Split a block of text into individual sentences (kept for callers/tests)."""
    return [text[s:e] for s, e in _sentence_spans(text)]


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

def reattach_citations(
    answer: str,
    retrieved_chunks: list[dict],   # [{"source_url": str, "content": str}, ...]
    *,
    nli_fn: NLIFn,
) -> ReattachResult:
    """Insert missing [url] citations into answer without changing wording.

    Algorithm per uncited sentence:
      1. Skip it unless it is real prose (see _is_citable).
      2. For each retrieved chunk, run NLI(chunk_content, sentence).
      3. If any chunk entails the sentence, insert [source_url] after the sentence.
      4. If multiple chunks entail the sentence, insert all their [url] tags.
    """
    # Build unique url -> content map (first occurrence wins, same as scorer)
    url_to_content: dict[str, str] = {}
    for chunk in retrieved_chunks:
        url = chunk.get("source_url", "")
        content = chunk.get("content", "")
        if url and url not in url_to_content:
            url_to_content[url] = content

    if not url_to_content:
        return ReattachResult(
            original_answer=answer,
            patched_answer=answer,
            citations_added=0,
            sentences_checked=0,
            sentences_covered=0,
            sentences_skipped=0,
        )

    # Split the answer on existing [url] citations. `texts` are the uncited
    # stretches; `urls[i]` (when present) is the citation that already covers
    # the LAST sentence of texts[i].
    parts = _CITATION_RE.split(answer)
    texts = parts[0::2]
    urls = parts[1::2]

    citations_added = 0
    sentences_checked = 0
    sentences_covered = 0
    sentences_skipped = 0
    decisions: list[InsertionDecision] = []

    output_parts: list[str] = []

    for i, text in enumerate(texts):
        has_following_citation = i < len(urls)
        spans = _sentence_spans(text)

        # (span_end_offset -> citation string) splices for this text segment
        splices: dict[int, str] = {}
        n = len(spans)

        for j, (start, end) in enumerate(spans):
            # The last sentence is already covered by the citation that follows.
            if j == n - 1 and has_following_citation:
                continue

            sentence = text[start:end]
            if not _is_citable(sentence):
                sentences_skipped += 1
                continue

            sentences_checked += 1
            supporting_urls: list[str] = []
            for url, content in url_to_content.items():
                try:
                    label = nli_fn(content, sentence)
                except Exception:
                    label = "neutral"
                if label == "entailment":
                    supporting_urls.append(url)

            if supporting_urls:
                sentences_covered += 1
                citations_added += len(supporting_urls)
                splices[end] = "".join(f"[{u}]" for u in supporting_urls)
                for url in supporting_urls:
                    decisions.append(InsertionDecision(
                        sentence=sentence,
                        url=url,
                        chunk_content=url_to_content[url],
                    ))

        # Rebuild this segment verbatim, splicing citations at sentence ends.
        if splices:
            rebuilt: list[str] = []
            cursor = 0
            for offset in sorted(splices):
                rebuilt.append(text[cursor:offset])
                rebuilt.append(" " + splices[offset])
                cursor = offset
            rebuilt.append(text[cursor:])
            output_parts.append("".join(rebuilt))
        else:
            output_parts.append(text)

        if has_following_citation:
            output_parts.append(f"[{urls[i]}]")

    return ReattachResult(
        original_answer=answer,
        patched_answer="".join(output_parts),
        citations_added=citations_added,
        sentences_checked=sentences_checked,
        sentences_covered=sentences_covered,
        sentences_skipped=sentences_skipped,
        decisions=decisions,
    )


def assert_only_citations_added(result: ReattachResult) -> None:
    """Raise AssertionError if patching changed anything but [url] insertions.

    Stripping every citation tag and all whitespace from both answers must yield
    identical strings — this is the "zero wording changes" claim, checked rather
    than asserted in prose.
    """
    def normalize(text: str) -> str:
        return re.sub(r"\s+", "", _CITATION_RE.sub("", text))

    before, after = normalize(result.original_answer), normalize(result.patched_answer)
    if before != after:
        raise AssertionError(
            "guardrail altered answer text beyond citation insertion "
            f"({len(before)} vs {len(after)} non-citation chars)"
        )
