"""guardrail/reattach_v2.py — Attribution guardrail v2 (Ticket 16).

v1 (`reattach.py`) is kept unchanged as the naive baseline. Its human audit
(Ticket 4) found most insertions wrong, because it ran NLI on raw Markdown
"sentences" (whole code blocks, `2. **Title**:` headings) against noisy chunks,
and cited every chunk that cleared argmax. v2 changes only how a citation is
chosen; segmentation and splicing are v1's, so the "zero wording changes"
guarantee and `assert_only_citations_added()` are unchanged.

Per uncited sentence, v2:
  1. keeps prose only (drops fenced code, list/heading markers, emphasis),
  2. decomposes the prose into atomic claims (same decompose_fn as the scorer),
  3. cleans each chunk (HTML tags, `{ #anchor }` suffixes, link syntax),
  4. accepts a chunk only if EVERY claim is argmax-entailment with
     P(entailment) >= threshold,
  5. cites only the single best chunk (highest min-over-claims P(entailment)).

Every sentence whose best chunk is all-claims-argmax-entailment is recorded as
a candidate with its score, whether or not it clears the threshold, so one run
supports a threshold sensitivity sweep.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from typing import Callable

from guardrail.reattach import ReattachResult, _is_citable, _sentence_spans
from measure.score import _CITATION_RE, _LABEL_MAP, DecomposeFn

# (premise, hypothesis) pairs -> [(label, P(entailment)), ...], batched.
NLIProbFn = Callable[[list[tuple[str, str]]], list[tuple[str, float]]]

DEFAULT_THRESHOLD = 0.9  # pre-registered 2026-10-10, before any held-out labels (DECISIONS.md)


@dataclass
class Candidate:
    """The best-supported chunk for one uncited sentence, inserted or not."""
    sentence: str
    prose: str               # what was decomposed (code and Markdown removed)
    claims: list[str]
    url: str
    chunk_content: str       # raw chunk, as the agent retrieved it
    min_p_entail: float      # weakest claim's P(entailment); the chunk's support score
    inserted: bool           # min_p_entail >= threshold


@dataclass
class ReattachResultV2(ReattachResult):
    candidates: list[Candidate] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Batched NLI with probabilities
# ---------------------------------------------------------------------------

def make_nli_prob_fn(model, cache, batch_size: int = 8) -> NLIProbFn:
    """NLI over many pairs at once, returning (label, P(entailment)) per pair.

    Cached per pair under its own key prefix, so it never collides with the
    scorer's label-only cache. Label order is the model's: 0=contradiction,
    1=entailment, 2=neutral (`_LABEL_MAP`).
    """

    def _key(premise: str, hypothesis: str) -> str:
        return "nliprob:" + hashlib.sha256((premise + "\x00" + hypothesis).encode()).hexdigest()

    def _probs(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        out: list[tuple[str, float] | None] = [None] * len(pairs)
        todo: list[tuple[int, str]] = []
        for i, (p, h) in enumerate(pairs):
            key = _key(p, h)
            cached = cache.get(key)
            if cached is None:
                todo.append((i, key))
            else:
                out[i] = (cached[0], cached[1])
        if todo:
            scores = model.predict(
                [pairs[i] for i, _ in todo], apply_softmax=True, batch_size=batch_size,
                show_progress_bar=False,
            )
            for (i, key), s in zip(todo, scores):
                result = (_LABEL_MAP[int(s.argmax())], float(s[1]))
                cache.set(key, list(result))
                out[i] = result
        return out  # type: ignore[return-value]

    return _probs


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
_LINE_MARKER = re.compile(r"^\s*(?:#{1,6}\s+|\d+[.)]\s+|[-*+]\s+|>\s*)", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[\w-]+\s*\}")
_DOC_DIRECTIVE = re.compile(r"^\s*(?:///.*|:::.*)$", re.MULTILINE)


def _code_mask(text: str) -> list[bool]:
    """Per-character flag: True inside a fenced code block (fence lines included).

    Computed over the whole text segment, because v1's punctuation splitter can
    cut a span in the middle of a code block.
    """
    mask = [False] * len(text)
    in_code, pos = False, 0
    for line in text.splitlines(keepends=True):
        is_fence = bool(_FENCE_LINE.match(line))
        if in_code or is_fence:
            for k in range(pos, pos + len(line)):
                mask[k] = True
        if is_fence:
            in_code = not in_code
        pos += len(line)
    return mask


def prose_of(text: str, start: int, end: int, mask: list[bool]) -> str:
    """The claim-bearing prose of text[start:end]: code and Markdown scaffolding removed."""
    kept = "".join(ch if not mask[k] else "\n" for k, ch in enumerate(text[start:end], start))
    kept = _LINE_MARKER.sub("", kept)
    kept = _EMPHASIS.sub(r"\2", kept)
    kept = _MD_LINK.sub(r"\1", kept)
    kept = kept.replace("`", "")
    return re.sub(r"\s+", " ", kept).strip()


def clean_premise(chunk: str) -> str:
    """Chunk text as NLI should see it: no HTML tags, anchors, link syntax or doc directives.

    Code inside the chunk is kept: a chunk's code can genuinely support a claim
    such as "use exclude_unset in .model_dump()".
    """
    c = _HTML_TAG.sub("", chunk)
    c = html.unescape(c)
    c = _HEADING_ANCHOR.sub("", c)
    c = _MD_LINK.sub(r"\1", c)
    c = _DOC_DIRECTIVE.sub("", c)
    c = re.sub(r"[ \t]+", " ", c)
    return re.sub(r"\n{3,}", "\n\n", c).strip()


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

def reattach_citations_v2(
    answer: str,
    retrieved_chunks: list[dict],   # [{"source_url": str, "content": str}, ...]
    question: str,
    *,
    nli_prob_fn: NLIProbFn,
    decompose_fn: DecomposeFn,
    threshold: float = DEFAULT_THRESHOLD,
) -> ReattachResultV2:
    """Insert at most one [url] after each uncited sentence that a chunk fully supports."""
    # url -> content, first occurrence wins (same as v1 and the scorer)
    url_to_content: dict[str, str] = {}
    for chunk in retrieved_chunks:
        url = chunk.get("source_url", "")
        if url and url not in url_to_content:
            url_to_content[url] = chunk.get("content", "")

    if not url_to_content:
        return ReattachResultV2(answer, answer, 0, 0, 0, 0, threshold=threshold)

    urls_in_order = list(url_to_content)
    premises = {u: clean_premise(c) for u, c in url_to_content.items()}

    parts = _CITATION_RE.split(answer)
    texts, cited = parts[0::2], parts[1::2]

    citations_added = sentences_checked = sentences_covered = sentences_skipped = 0
    candidates: list[Candidate] = []
    output_parts: list[str] = []

    for i, text in enumerate(texts):
        has_following_citation = i < len(cited)
        spans = _sentence_spans(text)
        mask = _code_mask(text)
        splices: dict[int, str] = {}

        for j, (start, end) in enumerate(spans):
            # The last sentence is already covered by the citation that follows.
            if j == len(spans) - 1 and has_following_citation:
                continue

            prose = prose_of(text, start, end, mask)
            if not _is_citable(prose):
                sentences_skipped += 1
                continue
            sentences_checked += 1

            claims = decompose_fn(prose, question) or [prose]
            pairs = [(premises[u], c) for u in urls_in_order for c in claims]
            results = nli_prob_fn(pairs)

            best_url, best_score = None, -1.0
            for k, url in enumerate(urls_in_order):
                per_claim = results[k * len(claims):(k + 1) * len(claims)]
                if all(label == "entailment" for label, _ in per_claim):
                    score = min(p for _, p in per_claim)
                    if score > best_score:   # strict: ties keep retrieval order
                        best_url, best_score = url, score

            if best_url is None:
                continue
            inserted = best_score >= threshold
            candidates.append(Candidate(
                sentence=text[start:end], prose=prose, claims=list(claims), url=best_url,
                chunk_content=url_to_content[best_url], min_p_entail=best_score, inserted=inserted,
            ))
            if inserted:
                splices[end] = f"[{best_url}]"
                citations_added += 1
                sentences_covered += 1

        if splices:
            rebuilt, cursor = [], 0
            for offset in sorted(splices):
                rebuilt.append(text[cursor:offset])
                rebuilt.append(" " + splices[offset])
                cursor = offset
            rebuilt.append(text[cursor:])
            output_parts.append("".join(rebuilt))
        else:
            output_parts.append(text)

        if has_following_citation:
            output_parts.append(f"[{cited[i]}]")

    return ReattachResultV2(
        original_answer=answer,
        patched_answer="".join(output_parts),
        citations_added=citations_added,
        sentences_checked=sentences_checked,
        sentences_covered=sentences_covered,
        sentences_skipped=sentences_skipped,
        candidates=candidates,
        threshold=threshold,
    )
