"""pipelines/openai_agent.py — ReAct agent via OpenAI Agents SDK + MCPServerStreamableHttp.

Follows OpenAI's MCP documentation (no Kapa template exists for this SDK).
Citation prompting and output shape are kept identical to langgraph_agent.py
for a fair cross-framework comparison.

Note: The OpenAI Agents SDK is most natural with OpenAI models. OpenRouter support
is included via a custom base_url client but is secondary — the primary cross-LLM
comparison runs through langgraph_agent.py which has first-class support for any
LangChain-compatible chat model.

Output shape: identical to baseline.py (pipeline="openai_agent").

Usage:
    uv run python -m pipelines.openai_agent --provider openai --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import diskcache
from dotenv import load_dotenv

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from agents.items import ToolCallItem, ToolCallOutputItem

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_PATH   = Path("eval_set/eval.jsonl")
RESULTS_DIR = Path("results")
MCP_URL     = "http://127.0.0.1:8000/mcp"
MAX_TURNS   = 10
CACHE_DIR   = ".cache/llm"

# ---------------------------------------------------------------------------
# System prompt — same wording as langgraph_agent.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise technical assistant answering questions about FastAPI.
You have access to a search tool that retrieves relevant passages from the FastAPI documentation.

Use the search tool as many times as needed to find the information required to answer the question.
After each factual claim in your final answer, cite the source URL in square brackets immediately
after the claim, like this:
  FastAPI uses Pydantic for validation. [https://fastapi.tiangolo.com/tutorial/body/]

Rules:
- Every factual claim must be followed by [source_url] from the retrieved passages.
- Do not cite a URL unless it directly supports that specific claim.
- If a claim is supported by multiple passages, cite all of them: [url1] [url2]
- If you cannot find enough information to answer, say so explicitly — do not fabricate.
- Do not add citations that are not from retrieved passages.
"""


# ---------------------------------------------------------------------------
# Trace extraction from RunResult.new_items
# ---------------------------------------------------------------------------

def _extract_trace(new_items) -> list[dict]:
    """Parse all ToolCallOutputItems to extract retrieved {source_url, content} chunks."""
    all_chunks: list[dict] = []
    seen_urls: set[str] = set()

    for item in new_items:
        if isinstance(item, ToolCallOutputItem):
            output = item.output
            # output can be: str, dict {"type": "text", "text": "..."}, or list thereof
            if isinstance(output, dict):
                raw = output.get("text", "")
            elif isinstance(output, list):
                raw = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in output
                )
            else:
                raw = str(output) if output is not None else ""
            try:
                chunks = json.loads(raw)
                if isinstance(chunks, list):
                    for chunk in chunks:
                        url = chunk.get("source_url", "")
                        if url and url not in seen_urls:
                            all_chunks.append(chunk)
                            seen_urls.add(url)
            except (json.JSONDecodeError, AttributeError):
                pass

    return all_chunks


def _count_tool_calls(new_items) -> int:
    return sum(1 for item in new_items if isinstance(item, ToolCallItem))


# ---------------------------------------------------------------------------
# Run one question
# ---------------------------------------------------------------------------

def _make_openai_client(provider: str):
    """Return an AsyncOpenAI client pointed at the right base URL."""
    import openai as oai
    if provider == "openai":
        return oai.AsyncOpenAI()
    if provider == "openrouter":
        return oai.AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown provider: {provider!r}")


async def _run_one_async(
    question: str,
    question_id: int,
    provider: str,
    model_id: str,
) -> dict:
    t0 = time.perf_counter()

    mcp_server = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(url=MCP_URL),
        cache_tools_list=True,   # avoid repeated list_tools() calls within a run
        name="fastapi-docs",
    )

    openai_client = _make_openai_client(provider)

    agent = Agent(
        name="fastapi-assistant",
        instructions=SYSTEM_PROMPT,
        mcp_servers=[mcp_server],
        model=model_id,
    )

    async with mcp_server:
        result = await Runner.run(
            starting_agent=agent,
            input=question,
            max_turns=MAX_TURNS,
        )

    answer = str(result.final_output) if result.final_output is not None else ""
    retrieved_chunks = _extract_trace(result.new_items)
    tool_call_count  = _count_tool_calls(result.new_items)
    latency = time.perf_counter() - t0

    return {
        "pipeline":         "openai_agent",
        "provider":         provider,
        "model":            model_id,
        "question_id":      question_id,
        "question":         question,
        "answer":           answer,
        "retrieved_chunks": retrieved_chunks,
        "tool_call_count":  tool_call_count,
        "latency_s":        round(latency, 3),
    }


def run_openai_agent(
    question: str,
    question_id: int,
    provider: str,
    model_id: str,
) -> dict:
    return asyncio.run(_run_one_async(question, question_id, provider, model_id))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI Agents SDK pipeline over the eval set"
    )
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    from pipelines.strategy import get_strategy
    cache    = diskcache.Cache(CACHE_DIR)
    strat    = get_strategy(args.provider, cache=cache)
    model_id = strat.model_id

    out_path = (
        Path(args.out) if args.out
        else RESULTS_DIR / f"openai_agent_{args.provider}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[: args.limit]

    logger.info("OpenAI Agent | provider=%s model=%s | %d questions → %s",
                args.provider, model_id, len(questions), out_path)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            try:
                result = run_openai_agent(
                    question=q["question"],
                    question_id=q["id"],
                    provider=args.provider,
                    model_id=model_id,
                )
                result["gold_source_urls"] = q.get("gold_source_urls", [])
                result["q_type"]           = q.get("type", "")
                result["difficulty"]       = q.get("difficulty", "")
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                logger.info("[%d/%d] id=%d latency=%.2fs tool_calls=%d chunks=%d",
                            i + 1, len(questions), q["id"],
                            result["latency_s"], result["tool_call_count"],
                            len(result["retrieved_chunks"]))
            except Exception as exc:
                logger.error("[%d/%d] id=%d FAILED: %s", i + 1, len(questions), q["id"], exc)

    logger.info("Done. Results written to %s", out_path)


if __name__ == "__main__":
    main()
