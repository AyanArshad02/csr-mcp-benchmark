"""Generate ~250 candidate Q&A pairs for human verification.

Outputs eval_set/candidates.jsonl.  Do NOT treat these as ground truth —
LLM-drafted gold attributions are frequently wrong.  Every item must be
human-verified before it enters eval.jsonl.  See CLAUDE.md hard rule #3.

Usage:
    uv run python -m eval_set.build_candidates

What it produces:
    eval_set/candidates.jsonl  — ~250 draft items, each with "verified": false
    (human sets "verified": true and trims bad items, then runs finalize.py)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import diskcache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "fastapi_docs"
CACHE_DIR = Path(".cache/eval_candidates")
OUTPUT_PATH = Path("eval_set/candidates.jsonl")

# gpt-4o-mini: cheap, structured-output-capable, good at following instructions
DRAFT_MODEL = "gpt-4o-mini"

# Over-generate so human can prune to ~150 good items
TARGET_SINGLE = 165
TARGET_MULTI = 65
TARGET_UNANSWERABLE = 20

# Minimum chunk length to consider for question generation
MIN_CHUNK_CHARS = 200

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_SINGLE_PROMPT = """\
You are building an evaluation dataset for a FastAPI documentation RAG system.

Given the FastAPI documentation chunk below, write ONE specific question that:
1. A real FastAPI developer would actually ask.
2. Can only be correctly answered by reading this specific chunk.
3. CANNOT be answered from general Python / web-framework knowledge alone.
4. Has a concise, factually correct answer based ONLY on this chunk.

Source URL: {source_url}

Documentation chunk:
---
{content}
---

Respond with a JSON object only, no markdown or extra text:
{{
  "question": "<specific, concrete question>",
  "reference_answer": "<concise answer based only on the chunk above>",
  "gold_source_urls": ["{source_url}"],
  "difficulty": "<easy|medium|hard>",
  "type": "single-source"
}}

Difficulty guide:
- easy: direct fact lookup (what parameter does X accept? what is the return type of Y?)
- medium: requires understanding a concept, not just locating a fact
- hard: requires synthesising multiple parts of the chunk or grasping a non-obvious implication"""

_MULTI_PROMPT = """\
You are building an evaluation dataset for a FastAPI documentation RAG system.

Given TWO FastAPI documentation chunks from DIFFERENT pages, write ONE question that:
1. A real FastAPI developer would actually ask.
2. Requires information from BOTH chunks to answer fully — one chunk alone is not enough.
3. Has a concise, factually correct answer drawing on both chunks.

Chunk A — Source URL: {url_a}
---
{content_a}
---

Chunk B — Source URL: {url_b}
---
{content_b}
---

Respond with a JSON object only, no markdown or extra text.

If you CAN write a genuine multi-source question (one that truly needs BOTH chunks):
{{
  "question": "<question requiring both chunks>",
  "reference_answer": "<concise answer drawing on both chunks>",
  "gold_source_urls": ["{url_a}", "{url_b}"],
  "difficulty": "medium",
  "type": "multi-source"
}}

If one chunk alone is sufficient to answer any natural question about this pair:
{{"skip": true}}"""

# Hardcoded unanswerable items — realistic FastAPI developer questions that are NOT
# covered by the FastAPI docs corpus (they concern third-party integrations, ops tooling,
# or ecosystem libraries that FastAPI's own docs do not document).
# These test whether pipelines correctly abstain instead of fabricating an answer.
_UNANSWERABLE = [
    "How do I integrate FastAPI with Celery for background task queuing?",
    "Can I use FastAPI with Django ORM directly without SQLAlchemy?",
    "How do I deploy FastAPI to AWS Lambda using the Mangum adapter?",
    "How do I add a GraphQL endpoint to FastAPI using the Strawberry library?",
    "How do I add rate limiting to FastAPI using the slowapi package?",
    "How do I use FastAPI with Beanie ODM for MongoDB?",
    "How do I add Prometheus metrics to FastAPI with prometheus-fastapi-instrumentator?",
    "How do I implement soft deletes in FastAPI with SQLAlchemy?",
    "How do I use FastAPI with Redis for server-side session management?",
    "How do I configure FastAPI to send transactional emails with fastapi-mail?",
    "How do I add request-level tracing to FastAPI with OpenTelemetry?",
    "How do I run FastAPI with Gunicorn and multiple workers behind Nginx in production?",
    "How do I configure Google OAuth2 in FastAPI using the Authlib library?",
    "How do I serve a trained scikit-learn model through FastAPI and cache predictions?",
    "How do I use FastAPI with SQLModel and Alembic for database migrations?",
    "How do I run database schema migrations automatically on FastAPI startup?",
    "How do I implement WebSocket authentication in FastAPI using JWT cookies?",
    "How do I stream large binary file downloads from a FastAPI endpoint?",
    "How do I use FastAPI with Tortoise ORM for async database access?",
    "How do I add full-text search to FastAPI using Elasticsearch?",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(f"{DRAFT_MODEL}\n{prompt}".encode()).hexdigest()


def _call_llm(client: OpenAI, cache: diskcache.Cache, prompt: str) -> str | None:
    key = _cache_key(prompt)
    if key in cache:
        return cache[key]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=DRAFT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            text = resp.choices[0].message.content
            cache.set(key, text)
            return text
        except Exception as exc:
            if attempt == 2:
                logger.warning(f"LLM call failed: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def _parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _topic_prefix(url: str) -> str:
    """Return the first path segment of a fastapi.tiangolo.com URL.
    e.g. https://fastapi.tiangolo.com/tutorial/dependencies/ → 'tutorial'"""
    path = url.replace("https://fastapi.tiangolo.com/", "").strip("/")
    parts = path.split("/")
    return parts[0] if parts and parts[0] else "root"


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

def _gen_single(client: OpenAI, cache: diskcache.Cache, chunk: dict) -> dict | None:
    prompt = _SINGLE_PROMPT.format(
        source_url=chunk["source_url"],
        content=chunk["content"][:2000],
    )
    data = _parse_json(_call_llm(client, cache, prompt))
    if not data or "question" not in data or "reference_answer" not in data:
        return None
    return data


def _gen_multi(client: OpenAI, cache: diskcache.Cache, a: dict, b: dict) -> dict | None:
    prompt = _MULTI_PROMPT.format(
        url_a=a["source_url"],
        content_a=a["content"][:1500],
        url_b=b["source_url"],
        content_b=b["content"][:1500],
    )
    data = _parse_json(_call_llm(client, cache, prompt))
    if not data or data.get("skip"):
        return None
    if "question" not in data or "reference_answer" not in data:
        return None
    return data


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_candidates() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(CACHE_DIR))
    client = OpenAI()
    rng = random.Random(RANDOM_SEED)

    # Load all chunks from Chroma
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(COLLECTION_NAME)
    raw = col.get(include=["documents", "metadatas"])
    all_chunks = [
        {
            "content": doc,
            "source_url": meta["source_url"],
            "heading_path": meta.get("heading_path", ""),
        }
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    logger.info(f"Loaded {len(all_chunks)} chunks from Chroma")

    usable = [c for c in all_chunks if len(c["content"]) >= MIN_CHUNK_CHARS]
    logger.info(f"{len(usable)} usable chunks (>= {MIN_CHUNK_CHARS} chars)")

    candidates: list[dict] = []
    cid = 0

    # --- single-source ------------------------------------------------------ #
    logger.info(f"Generating up to {TARGET_SINGLE} single-source candidates...")
    pool = rng.sample(usable, min(TARGET_SINGLE + 50, len(usable)))
    for chunk in pool:
        if len(candidates) >= TARGET_SINGLE:
            break
        item = _gen_single(client, cache, chunk)
        if item:
            item["candidate_id"] = f"c{cid:04d}"
            item["verified"] = False
            candidates.append(item)
            cid += 1
            if cid % 25 == 0:
                logger.info(f"  {cid} total candidates generated...")

    single_count = len(candidates)
    logger.info(f"Single-source: {single_count}")

    # --- multi-source ------------------------------------------------------- #
    logger.info(f"Generating up to {TARGET_MULTI} multi-source candidates...")
    # Group usable chunks by topic prefix, collect pairs across different source_urls
    topic_map: dict[str, list[dict]] = defaultdict(list)
    for c in usable:
        topic_map[_topic_prefix(c["source_url"])].append(c)

    pairs: list[tuple[dict, dict]] = []
    for topic, group in topic_map.items():
        urls = list({c["source_url"] for c in group})
        if len(urls) < 2:
            continue
        rng.shuffle(urls)
        for i in range(len(urls) - 1):
            a = rng.choice([c for c in group if c["source_url"] == urls[i]])
            b = rng.choice([c for c in group if c["source_url"] == urls[i + 1]])
            pairs.append((a, b))
    rng.shuffle(pairs)

    multi_count = 0
    for a, b in pairs:
        if multi_count >= TARGET_MULTI:
            break
        item = _gen_multi(client, cache, a, b)
        if item:
            item["candidate_id"] = f"c{cid:04d}"
            item["verified"] = False
            candidates.append(item)
            cid += 1
            multi_count += 1
            if multi_count % 15 == 0:
                logger.info(f"  {multi_count} multi-source so far...")

    logger.info(f"Multi-source: {multi_count}")

    # --- unanswerable ------------------------------------------------------- #
    logger.info(f"Adding {TARGET_UNANSWERABLE} unanswerable items...")
    for question in _UNANSWERABLE[:TARGET_UNANSWERABLE]:
        candidates.append({
            "candidate_id": f"c{cid:04d}",
            "question": question,
            "reference_answer": "This question cannot be answered from the FastAPI documentation corpus.",
            "gold_source_urls": [],
            "difficulty": "medium",
            "type": "unanswerable",
            "verified": False,
        })
        cid += 1

    # Shuffle so the human review isn't one big block of each type
    rng.shuffle(candidates)
    # Re-assign IDs after shuffle for stable sequential numbering
    for i, c in enumerate(candidates):
        c["candidate_id"] = f"c{i:04d}"

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary
    type_counts: dict[str, int] = {}
    for c in candidates:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1

    logger.info(f"\nWrote {len(candidates)} candidates → {OUTPUT_PATH}")
    for t, n in sorted(type_counts.items()):
        logger.info(f"  {t}: {n}")
    logger.info(
        "\nNext steps:\n"
        "  1. Open eval_set/candidates.jsonl and review every item.\n"
        "  2. For each item: set 'verified': true if it passes, or delete the line if it fails.\n"
        "     Verification criteria per item:\n"
        "       - Is the question something a real FastAPI developer would ask?\n"
        "       - Is the reference_answer factually correct based on the docs?\n"
        "       - Do the gold_source_urls actually support the answer? (click and check)\n"
        "       - Would the question be answerable from general Python knowledge alone? (prune if yes)\n"
        "  3. Once you have ~150 verified items, run: uv run python -m eval_set.finalize\n"
        "  Budget ~2–4 min per item. ~150 items ≈ 5–10 hours total."
    )


if __name__ == "__main__":
    build_candidates()
