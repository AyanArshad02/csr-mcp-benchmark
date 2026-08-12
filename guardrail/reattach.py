"""guardrail/reattach.py — Attribution guardrail: reattach missing citations.

Given an agent answer and the chunks the agent actually retrieved, for each
sentence that lacks a [url] citation, try to find a retrieved chunk that NLI-
supports the sentence and insert [url] immediately after it.

Zero wording changes — only [url] tags are inserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from measure.score import NLIFn, _CITATION_RE


@dataclass
class ReattachResult:
    original_answer: str
    patched_answer: str
    citations_added: int        # number of [url] insertions made
    sentences_checked: int      # uncited sentences we ran NLI on
    sentences_covered: int      # uncited sentences that found at least one supporting chunk


def _sentences_from_text(text: str) -> list[str]:
    """Split a block of text into individual sentences (rough but consistent)."""
    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def reattach_citations(
    answer: str,
    retrieved_chunks: list[dict],   # [{"source_url": str, "content": str}, ...]
    *,
    nli_fn: NLIFn,
) -> ReattachResult:
    """Insert missing [url] citations into answer without changing wording.

    Algorithm per uncited sentence:
      1. For each retrieved chunk, run NLI(chunk_content, sentence).
      2. If any chunk entails the sentence, insert [source_url] after the sentence.
      3. If multiple chunks entail the sentence, insert all their [url] tags.
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
        )

    # Split answer on existing [url] citations to find uncited segments
    # Strategy: work segment by segment; segments between citations that
    # contain text are candidates for reattachment.
    parts = _CITATION_RE.split(answer)
    # parts alternates: text, url, text, url, ..., text
    texts = parts[0::2]
    urls = parts[1::2]

    citations_added = 0
    sentences_checked = 0
    sentences_covered = 0

    output_parts: list[str] = []

    for i, text in enumerate(texts):
        # Determine if this text segment already has a following citation
        has_following_citation = i < len(urls)

        # If text segment ends immediately before a citation, the last sentence
        # is already cited. But text may also contain multiple sentences, the
        # first of which are uncited.
        # For simplicity: if no following citation, ALL sentences in this segment
        # are uncited and eligible. If there IS a following citation, only
        # sentences before the last are uncited (the last is covered by the url).

        sentences = _sentences_from_text(text)

        if not sentences:
            output_parts.append(text)
            if has_following_citation:
                output_parts.append(f"[{urls[i]}]")
            continue

        rebuilt = []
        n = len(sentences)

        for j, sentence in enumerate(sentences):
            is_last = j == n - 1
            # If this is the last sentence AND there's a citation following the
            # whole text block, it's already cited — don't reattach.
            if is_last and has_following_citation:
                rebuilt.append(sentence)
                continue

            # This sentence is uncited — try every retrieved chunk
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
                citation_str = "".join(f"[{u}]" for u in supporting_urls)
                rebuilt.append(f"{sentence} {citation_str}")
            else:
                rebuilt.append(sentence)

        # Re-join sentences with a space, preserving rough structure
        rebuilt_text = " ".join(rebuilt)

        # Preserve leading/trailing whitespace from original segment
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        output_parts.append(lead + rebuilt_text + trail)

        if has_following_citation:
            output_parts.append(f"[{urls[i]}]")

    patched = "".join(output_parts)

    return ReattachResult(
        original_answer=answer,
        patched_answer=patched,
        citations_added=citations_added,
        sentences_checked=sentences_checked,
        sentences_covered=sentences_covered,
    )
