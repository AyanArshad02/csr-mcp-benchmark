"""Chunk the downloaded FastAPI docs, embed each chunk, and persist to Chroma.

Usage: uv run python -m ingest.build_index
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from pathlib import Path

import chromadb
import diskcache
from dotenv import load_dotenv
from openai import OpenAI

from ingest.chunker import Chunk, chunk_markdown
from ingest.download import RAW_DIR, download_fastapi_docs, path_to_url

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "fastapi_docs"
CACHE_DIR = Path(".cache/embeddings")


def _load_or_download(raw_dir: Path = RAW_DIR) -> list[Path]:
    md_files = sorted(raw_dir.rglob("*.md"))
    if not md_files:
        logger.info(f"No files found in {raw_dir}/ — downloading first.")
        download_fastapi_docs(raw_dir)
        md_files = sorted(raw_dir.rglob("*.md"))
    return md_files


def _chunk_all(md_files: list[Path], raw_dir: Path = RAW_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in md_files:
        relative_path = str(path.relative_to(raw_dir))
        content = path.read_text(encoding="utf-8")
        source_url = path_to_url(relative_path)
        chunks.extend(chunk_markdown(content, source_url=source_url, source_file=relative_path))
    return chunks


def _embed_chunks(chunks: list[Chunk], client: OpenAI, cache: diskcache.Cache) -> list[list[float]]:
    embeddings: list[list[float] | None] = [None] * len(chunks)
    to_fetch: list[tuple[int, str]] = []

    for i, chunk in enumerate(chunks):
        key = hashlib.sha256(f"{EMBEDDING_MODEL}:{chunk.content}".encode()).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            embeddings[i] = cached
        else:
            to_fetch.append((i, key))

    logger.info(f"Embedding {len(to_fetch)} new chunks ({len(chunks) - len(to_fetch)} cached)")

    batch_size = 100
    for batch_start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[batch_start : batch_start + batch_size]
        inputs = [chunks[i].content for i, _ in batch]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=inputs)
        for (i, key), item in zip(batch, resp.data):
            embeddings[i] = item.embedding
            cache.set(key, item.embedding)

    assert all(e is not None for e in embeddings)
    return embeddings  # type: ignore[return-value]


def build_index() -> None:
    md_files = _load_or_download()
    logger.info(f"{len(md_files)} Markdown files found.")

    chunks = _chunk_all(md_files)
    logger.info(f"{len(chunks)} chunks produced.")

    lengths = [len(c.content) for c in chunks]
    logger.info(
        f"Chunk length stats — min: {min(lengths)}, max: {max(lengths)}, "
        f"mean: {statistics.mean(lengths):.0f}, median: {statistics.median(lengths):.0f}"
    )
    missing_url = sum(1 for c in chunks if not c.source_url)
    if missing_url:
        raise ValueError(f"{missing_url} chunks have no source_url — aborting.")

    client = OpenAI()
    cache = diskcache.Cache(str(CACHE_DIR))
    embeddings = _embed_chunks(chunks, client, cache)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in chroma_client.list_collections()
    ] else None
    collection = chroma_client.create_collection(COLLECTION_NAME)

    ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
    documents = [c.content for c in chunks]
    metadatas = [
        {
            "source_url": c.source_url,
            "heading": c.heading,
            "heading_path": c.heading_path,
            "chunk_index": c.chunk_index,
            "source_file": c.source_file,
        }
        for c in chunks
    ]

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"Indexed {len(chunks)} chunks into Chroma at {CHROMA_DIR}/ (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    build_index()
