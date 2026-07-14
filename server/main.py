"""Self-hosted MCP retrieval server mirroring Kapa's `search_<product>_knowledge_sources`
contract over the FastAPI docs corpus.

Usage: uv run python -m server.main
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "fastapi_docs"

# Defaults mirror Kapa's documented bounds (docs.kapa.ai/integrations/mcp/overview);
# verify against the live page before relying on the exact numbers.
DEFAULT_TOP_K = 15
DEFAULT_MAX_CHARS = 35_000
CANDIDATE_POOL_SIZE = 30

mcp = FastMCP(name="csr-benchmark-fastapi-docs")

_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_collection(COLLECTION_NAME)
_openai_client = OpenAI()


def _read_meta_int(ctx: Context | None, key: str, default: int) -> int:
    if ctx is None:
        return default
    request_context = ctx.request_context
    if request_context is None or request_context.meta is None:
        return default
    value = getattr(request_context.meta, key, None)
    return int(value) if isinstance(value, (int, float)) else default


def _apply_max_chars(results: list[dict], max_chars: int) -> list[dict]:
    """Walk relevance-ordered results, keeping whole chunks within max_chars.
    Never truncates a chunk mid-string; if even the first chunk alone exceeds
    max_chars, returns an empty list (matches Kapa's documented behavior)."""
    kept: list[dict] = []
    total = 0
    for item in results:
        size = len(item["content"])
        if total + size > max_chars:
            break
        kept.append(item)
        total += size
    return kept


@mcp.tool(name="search_fastapi_knowledge_sources")
def search_fastapi_knowledge_sources(query: str, ctx: Context | None = None) -> list[dict]:
    """Search the FastAPI documentation knowledge base for content relevant to the query.

    Returns a list of {"source_url": str, "content": str} objects, ordered by
    relevance (most relevant first).
    """
    top_k = _read_meta_int(ctx, "top_k", DEFAULT_TOP_K)
    max_chars = _read_meta_int(ctx, "max_chars", DEFAULT_MAX_CHARS)

    query_embedding = (
        _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    )

    raw = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(CANDIDATE_POOL_SIZE, _collection.count()),
        include=["documents", "metadatas"],
    )

    candidates = [
        {"source_url": meta["source_url"], "content": doc}
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0])
    ]

    truncated = candidates[:top_k]
    return _apply_max_chars(truncated, max_chars)


if __name__ == "__main__":
    mcp.run(transport="http")
