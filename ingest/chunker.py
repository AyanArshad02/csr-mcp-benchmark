"""Heading-aware, recursive Markdown chunker.

Splits on H1-H3 headings (the citable-unit boundary), then recursively sub-splits
any section over `max_chars` on paragraph boundaries, and merges any section under
`min_chars` into its neighbor rather than dropping it. Never splits inside a fenced
code block, and never mistakes a `#`-prefixed line inside a code block for a heading.

Each chunk's content is prefixed with its ancestor heading chain (e.g. a chunk under
an H3 also carries its parent H1/H2 lines) so it reads as self-contained even when
returned in isolation via the MCP tool, without that chain leaking into Kapa's minimal
{source_url, content} contract shape (the ancestors are just more Markdown in `content`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^`{3,}.*?\n.*?^`{3,}[ \t]*$", re.MULTILINE | re.DOTALL)
_ANCHOR_SUFFIX_RE = re.compile(r"\s*\{[^}]*\}\s*$")

MAX_CHARS = 2000
MIN_CHARS = 50

Section = tuple[tuple[str, ...], str, str]  # (ancestor_heading_lines, own_heading_line, body)


@dataclass(frozen=True)
class Chunk:
    source_url: str
    content: str
    heading: str
    heading_path: str
    chunk_index: int
    source_file: str


def _code_spans(content: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CODE_FENCE_RE.finditer(content)]


def _inside_any_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _clean_heading_text(heading_line: str) -> str:
    """Strip leading '#'s and a trailing MkDocs anchor suffix, e.g.
    '## Create a middleware { #create-a-middleware }' -> 'Create a middleware'."""
    text = heading_line.lstrip("#").strip()
    return _ANCHOR_SUFFIX_RE.sub("", text).strip()


def _split_on_headings(content: str) -> list[Section]:
    """Return (ancestor_heading_lines, own_heading_line, body) triples, tracking
    heading nesting depth so each section also knows its ancestor chain. Ignores
    '#' lines inside code fences."""
    code_spans = _code_spans(content)
    matches = [
        m for m in _HEADING_RE.finditer(content) if not _inside_any_span(m.start(), code_spans)
    ]

    if not matches:
        return [((), "", content)]

    sections: list[Section] = []

    if matches[0].start() > 0:
        sections.append(((), "", content[: matches[0].start()]))

    stack: list[tuple[int, str]] = []  # (level, heading_line), outermost first
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_line = match.group(0)

        while stack and stack[-1][0] >= level:
            stack.pop()
        ancestors = tuple(line for _, line in stack)
        stack.append((level, heading_line))

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((ancestors, heading_line, content[body_start:body_end]))

    return sections


_LIST_ITEM_RE = re.compile(r"^(\s*([*\-+]|\d+[.)])\s)")


def _split_list_items(paragraph: str) -> list[str]:
    """Split a paragraph at list-item boundaries (handles blank-line-free lists,
    e.g. release-notes.md's hundreds of consecutive `* ...` lines with no blank
    lines between them, which a pure blank-line split would treat as one blob)."""
    lines = paragraph.split("\n")
    if sum(1 for line in lines if _LIST_ITEM_RE.match(line)) < 2:
        return [paragraph]

    units: list[str] = []
    current: list[str] = []
    for line in lines:
        if _LIST_ITEM_RE.match(line) and current:
            units.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        units.append("\n".join(current))
    return units


def _atomic_paragraphs(body: str) -> list[str]:
    """Split body into paragraph units, keeping each fenced code block intact."""
    code_spans = _code_spans(body)
    boundaries = sorted({0, len(body)} | {s for span in code_spans for s in span})

    units: list[str] = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = body[start:end]
        if _inside_any_span(start, code_spans):
            units.append(segment)
        else:
            for para in re.split(r"\n\s*\n", segment):
                if para.strip():
                    units.extend(_split_list_items(para))
    return units


def _pack_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedily pack paragraph units into sub-chunks no larger than max_chars."""
    packed: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > max_chars and current:
            packed.append(current)
            current = para
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _assemble_text(ancestors: tuple[str, ...], heading: str, body: str) -> str:
    parts = [*ancestors, heading, body] if heading else [body]
    return "\n\n".join(p for p in parts if p.strip()).strip()


def chunk_markdown(
    content: str,
    source_url: str,
    source_file: str,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    sections = _split_on_headings(content)

    # Recursive fallback: sub-split any section over max_chars on paragraph boundaries.
    expanded: list[Section] = []
    for ancestors, heading, body in sections:
        text = _assemble_text(ancestors, heading, body)
        if len(text) <= max_chars:
            expanded.append((ancestors, heading, body))
            continue
        for sub_body in _pack_paragraphs(_atomic_paragraphs(body), max_chars):
            expanded.append((ancestors, heading, sub_body))

    # Merge any section under min_chars into the next one (or the previous, if last).
    merged: list[Section] = []
    pending: Section | None = None
    for ancestors, heading, body in expanded:
        text = _assemble_text(ancestors, heading, body)
        if pending is not None:
            pending_ancestors, pending_heading, pending_body = pending
            if pending_heading:
                ancestors, heading = pending_ancestors, pending_heading
            body = f"{pending_body}\n\n{body}".strip()
            pending = None
        if len(text) < min_chars:
            pending = (ancestors, heading, body)
        else:
            merged.append((ancestors, heading, body))
    if pending is not None:
        if merged:
            last_ancestors, last_heading, last_body = merged[-1]
            merged[-1] = (last_ancestors, last_heading, f"{last_body}\n\n{pending[2]}".strip())
        else:
            merged.append(pending)

    chunks: list[Chunk] = []
    for i, (ancestors, heading, body) in enumerate(merged):
        text = _assemble_text(ancestors, heading, body)
        if not text:
            continue
        heading_path = " > ".join(
            _clean_heading_text(h) for h in (*ancestors, heading) if h
        )
        chunks.append(
            Chunk(
                source_url=source_url,
                content=text,
                heading=_clean_heading_text(heading),
                heading_path=heading_path,
                chunk_index=i,
                source_file=source_file,
            )
        )
    return chunks
