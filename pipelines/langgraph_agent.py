"""pipelines/langgraph_agent.py — ReAct agent via LangGraph + langchain-mcp-adapters.

Follows Kapa's published reference integration pattern (kapa-ai/langchain-agent-example):
  - MultiServerMCPClient with streamable_http transport
  - Tools discovered via get_tools()
  - Agent constructed via create_react_agent

Output shape: identical to baseline.py (same fields, pipeline="langgraph").

Usage:
    uv run python -m pipelines.langgraph_agent --provider openai --limit 5
    uv run python -m pipelines.langgraph_agent --provider openrouter
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
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

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
# System prompt — same wording as baseline, adapted for agent framing
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
# Trace extraction
# ---------------------------------------------------------------------------

def _parse_chunks(text: str, all_chunks: list, seen_urls: set) -> None:
    try:
        chunks = json.loads(text)
        if isinstance(chunks, list):
            for chunk in chunks:
                url = chunk.get("source_url", "")
                if url and url not in seen_urls:
                    all_chunks.append(chunk)
                    seen_urls.add(url)
    except (json.JSONDecodeError, AttributeError):
        pass


def _extract_trace(messages: list) -> list[dict]:
    """Extract all retrieved chunks from the agent's message history.

    Walks the message list and collects ToolMessage content (the MCP tool results),
    parsing each as a JSON list of {source_url, content} dicts.
    """
    all_chunks: list[dict] = []
    seen_urls: set[str] = set()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            raw = msg.content
            # LangGraph MCP adapter returns content as list[dict] with type/text keys
            if isinstance(raw, list):
                for block in raw:
                    text = block.get("text", "") if isinstance(block, dict) else str(block)
                    _parse_chunks(text, all_chunks, seen_urls)
            elif isinstance(raw, str):
                _parse_chunks(raw, all_chunks, seen_urls)
    return all_chunks


def _count_tool_calls(messages: list) -> int:
    """Count how many times the agent called a tool."""
    return sum(
        1 for msg in messages
        if isinstance(msg, AIMessage) and msg.tool_calls
    )


def _get_final_answer(messages: list) -> str:
    """Return the last AIMessage content that is NOT a tool call (the final answer)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Some models return a list of content blocks
                return " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
    return ""


# ---------------------------------------------------------------------------
# Build the LangGraph agent (provider-aware)
# ---------------------------------------------------------------------------

def _make_chat_model(provider: str, model_id: str):
    """Return a LangChain chat model for the given provider."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=0)

    if provider == "openrouter":
        import os
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return ChatOpenAI(
            model=model_id,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    if provider == "bedrock":
        import os
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            model_kwargs={"temperature": 0},
        )

    raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Run one question
# ---------------------------------------------------------------------------

async def _run_one_async(
    question: str,
    question_id: int,
    provider: str,
    model_id: str,
) -> dict:
    t0 = time.perf_counter()

    mcp_client = MultiServerMCPClient(
        connections={
            "fastapi-docs": {
                "url": MCP_URL,
                "transport": "streamable_http",
            }
        }
    )
    tools = await mcp_client.get_tools()
    chat_model = _make_chat_model(provider, model_id)
    agent = create_react_agent(
        model=chat_model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {"messages": [("human", question)]},
        config={"recursion_limit": MAX_TURNS * 2},
    )

    messages = result.get("messages", [])
    answer = _get_final_answer(messages)
    retrieved_chunks = _extract_trace(messages)
    tool_call_count = _count_tool_calls(messages)
    latency = time.perf_counter() - t0

    return {
        "pipeline":         "langgraph",
        "provider":         provider,
        "model":            model_id,
        "question_id":      question_id,
        "question":         question,
        "answer":           answer,
        "retrieved_chunks": retrieved_chunks,
        "tool_call_count":  tool_call_count,
        "latency_s":        round(latency, 3),
    }


def run_langgraph(
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
    parser = argparse.ArgumentParser(description="Run the LangGraph agent over the eval set")
    parser.add_argument("--provider", choices=["openai", "openrouter", "bedrock"], default="openai")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    # Resolve model ID from provider
    from pipelines.strategy import get_strategy
    cache   = diskcache.Cache(CACHE_DIR)
    strat   = get_strategy(args.provider, cache=cache)
    model_id = strat.model_id

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"langgraph_{args.provider}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[: args.limit]

    logger.info("LangGraph | provider=%s model=%s | %d questions → %s",
                args.provider, model_id, len(questions), out_path)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            try:
                result = run_langgraph(
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
