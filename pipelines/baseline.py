"""pipelines/baseline.py — Single-shot baseline pipeline (control condition).

One retrieve → one generate. No agentic loop.

This is the strongest possible single-shot pipeline:
- Retrieves top-15 chunks, passes ALL of them as context
- System prompt explicitly instructs inline [url] citations per claim
- No second call, no reasoning loop

Output shape (same across all 3 pipelines so the scorer can treat them uniformly):
{
    "pipeline":          "baseline",
    "provider":          "openai" | "openrouter",
    "model":             str,
    "question_id":       int,
    "question":          str,
    "answer":            str,           # with inline [url] citations
    "retrieved_chunks":  list[dict],    # [{"source_url": str, "content": str}]
    "tool_call_count":   int,           # always 1 for baseline
    "latency_s":         float,
}

Usage:
    uv run python -m pipelines.baseline --provider openai --limit 5
    uv run python -m pipelines.baseline --provider openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import diskcache
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from pipelines import mcp_config

from pipelines.strategy import get_strategy

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_PATH    = Path("eval_set/eval.jsonl")
RESULTS_DIR  = Path("results")
TOP_K        = 15
CACHE_DIR    = ".cache/llm"

# ---------------------------------------------------------------------------
# System prompt — IDENTICAL wording used across all 3 pipelines
# (Changing this mid-experiment invalidates comparisons.)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise technical assistant answering questions about FastAPI.

Answer using ONLY the provided context passages. After each claim in your answer,
cite the source URL in square brackets immediately after the claim, like this:
  FastAPI uses Pydantic for validation. [https://fastapi.tiangolo.com/tutorial/body/]

Rules:
- Every factual claim must be followed by [source_url] from the provided passages.
- Do not cite a URL unless it directly supports that specific claim.
- If a claim is supported by multiple passages, cite all of them: [url1] [url2]
- If the context does not contain enough information to answer, say so explicitly
  and do not fabricate an answer.
- Do not add citations that are not in the provided passages.
"""

# ---------------------------------------------------------------------------
# MCP retrieval (via fastmcp.Client — required by the streamable HTTP transport)
# ---------------------------------------------------------------------------

async def _retrieve_async(query: str, top_k: int = TOP_K) -> list[dict]:
    """Call the MCP server and return [{source_url, content}] chunks."""
    # Server, auth and tool name come from pipelines/mcp_config.py (.env); defaults = local mirror.
    transport = StreamableHttpTransport(mcp_config.mcp_url(), headers=mcp_config.mcp_headers())
    async with Client(transport) as client:
        result = await client.call_tool(
            mcp_config.search_tool(),
            {"query": query},
            meta={"top_k": top_k},
        )
    # CallToolResult.data contains the already-parsed return value from the tool
    # (FastMCP deserialises the JSON for us when the tool returns a list/dict).
    if result.data is not None:
        return result.data if isinstance(result.data, list) else []
    # Fallback: parse from the raw text content
    if result.content:
        raw = result.content[0].text
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


def _retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    return asyncio.run(_retrieve_async(query, top_k))


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered passages for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        url = chunk.get("source_url", "unknown")
        content = chunk.get("content", "").strip()
        parts.append(f"[Passage {i}] Source: {url}\n{content}")
    return "\n\n---\n\n".join(parts)


def run_baseline(
    question: str,
    question_id: int,
    strategy,
) -> dict:
    """Run one question through the baseline pipeline."""
    t0 = time.perf_counter()

    # Step 1: retrieve
    chunks = _retrieve(question, top_k=TOP_K)

    # Step 2: generate
    context = _build_context(chunks)
    user_msg = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite each claim with [source_url] immediately after it):"
    )
    answer = strategy.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
    )

    latency = time.perf_counter() - t0

    return {
        "pipeline":         "baseline",
        "provider":         strategy.provider,
        "model":            strategy.model_id,
        "question_id":      question_id,
        "question":         question,
        "answer":           answer,
        "retrieved_chunks": chunks,
        "tool_call_count":  1,
        "latency_s":        round(latency, 3),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline pipeline over the eval set")
    parser.add_argument("--provider", choices=["openai", "openrouter", "bedrock"], default="openai")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only the first N questions (for smoke tests)")
    parser.add_argument("--out",      type=str, default=None,
                        help="Output JSONL path (default: results/baseline_<provider>.jsonl)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the LLM response cache: a fresh generation (Ticket 15 repeat runs)")
    args = parser.parse_args()

    cache   = None if args.no_cache else diskcache.Cache(CACHE_DIR)
    strategy = get_strategy(args.provider, cache=cache)

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"baseline_{args.provider}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[: args.limit]

    logger.info(mcp_config.describe())
    logger.info("Baseline | provider=%s model=%s | %d questions → %s",
                args.provider, strategy.model_id, len(questions), out_path)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            try:
                result = run_baseline(
                    question=q["question"],
                    question_id=q["id"],
                    strategy=strategy,
                )
                # Attach gold URLs for downstream scoring
                result["gold_source_urls"] = q.get("gold_source_urls", [])
                result["q_type"]           = q.get("type", "")
                result["difficulty"]       = q.get("difficulty", "")
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                logger.info("[%d/%d] id=%d latency=%.2fs tool_calls=%d",
                            i + 1, len(questions),
                            q["id"], result["latency_s"], result["tool_call_count"])
            except Exception as exc:
                logger.error("[%d/%d] id=%d FAILED: %s", i + 1, len(questions), q["id"], exc)

    logger.info("Done. Results written to %s", out_path)


if __name__ == "__main__":
    main()
