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

Ablation flags (Ticket 9 / G2). Defaults reproduce the original published run exactly:
    --temperature 0                    default: unset (the API's own default)
    --api chat_completions             default: responses (the SDK default)
    --mcp-url http://127.0.0.1:8001/mcp  e.g. a server started with CSR_TOOL_DESCRIPTION=kapa
    --dump-inputs FILE.jsonl           record exactly what the model receives on every LLM call
    --system-prompt kapa               Kapa's recommended system prompt (their citation format;
                                       normalise with measure.normalize_kapa_citations before scoring)
    --system-prompt ids                cite by source_id (needs a CSR_SOURCE_IDS=1 server; map with
                                       guardrail.map_source_ids before scoring)
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

from pipelines import mcp_config
from pipelines.source_ids import IDS_SYSTEM_PROMPT

from agents import Agent, ModelSettings, RunHooks, Runner, set_default_openai_api
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


# Kapa's recommended system prompt for agents using its search tool, from
# docs.kapa.ai/retrieval/guides/prompt-your-agent (fetched 2026-10-12). Adaptations, and only these:
# [Product Name] -> FastAPI; [product description] filled in; tool name search_knowledge_base ->
# search_fastapi_knowledge_sources; the "...your other instructions..." placeholder removed.
# Its citation format is [[short title](URL)], NOT the benchmark's [url] (locked); answers from
# this prompt are normalised to [url] before scoring (approved by the user 2026-10-13).
KAPA_SYSTEM_PROMPT = """\
You are an AI assistant for FastAPI, a Python web framework for building APIs.

You have different tools to interact with the platform on behalf of the
user. You also have access to a knowledge base search tool to answer the
user's questions and fill in your own understanding of the product.

## Knowledge Base Search

Use the search_fastapi_knowledge_sources tool whenever you need information about
FastAPI, whether to answer a user's question directly, to
understand parts of the product in connection with the user's task, or
to interpret results and inputs from other tools. Never rely on your
inherent knowledge or any other tool for product-specific information.

When using the results, follow these rules:

1. Review the content of each knowledge source document carefully and
   assess its relevance to the user query before using it in your answer.
   Some documents may appear relevant at first but are not. If you do not
   find enough information in the knowledge sources to answer the query,
   clearly state that the knowledge sources do not contain enough
   information. Never try to make up an answer.
2. Answer the user query solely based on the relevant knowledge sources.
   Explicitly state your uncertainty with phrases like "I'm sorry, but
   there's no information about", "The knowledge sources do not explicitly
   mention" or similar, if you cannot find enough information in the
   knowledge sources to provide a confident answer. State your uncertainty
   at the beginning of your answer — the user must be made aware of any
   limitations upfront.
3. Cite documents that include a source URL. Format each citation as
   [[short title](URL)] using a 1-4 word title. Place the citation
   immediately after the sentence it supports. If two or more documents
   support the same sentence, include at most two citations inside the
   same brackets separated by a semicolon.
"""
SYSTEM_PROMPTS = {"default": SYSTEM_PROMPT, "kapa": KAPA_SYSTEM_PROMPT, "ids": IDS_SYSTEM_PROMPT}


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


class _DumpInputsHooks(RunHooks):
    """Append the exact system prompt + input items of every LLM call to a JSONL file."""

    def __init__(self, path: Path, question_id: int):
        self.path, self.question_id, self.call = path, question_id, 0

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self.call += 1
        items = [i.model_dump() if hasattr(i, "model_dump") else i for i in input_items]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": self.question_id, "llm_call": self.call,
                                "system_prompt": system_prompt, "input_items": items},
                               ensure_ascii=False, default=str) + "\n")


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
    temperature: float | None = None,
    mcp_url: str | None = None,
    dump_inputs: Path | None = None,
    system_prompt: str = "default",
) -> dict:
    t0 = time.perf_counter()

    mcp_server = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(   # server + auth from pipelines/mcp_config.py (.env)
            url=mcp_url or mcp_config.mcp_url(),
            **({"headers": h} if (h := mcp_config.mcp_headers()) else {}),
        ),
        cache_tools_list=True,   # avoid repeated list_tools() calls within a run
        name="fastapi-docs",
    )

    openai_client = _make_openai_client(provider)

    agent = Agent(
        name="fastapi-assistant",
        instructions=SYSTEM_PROMPTS[system_prompt],
        mcp_servers=[mcp_server],
        model=model_id,
        # temperature=None leaves it unset, exactly as in the original run
        model_settings=ModelSettings(temperature=temperature),
    )

    async with mcp_server:
        result = await Runner.run(
            starting_agent=agent,
            input=question,
            max_turns=MAX_TURNS,
            hooks=_DumpInputsHooks(dump_inputs, question_id) if dump_inputs else None,
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
    **ablation,
) -> dict:
    return asyncio.run(_run_one_async(question, question_id, provider, model_id, **ablation))


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
    parser.add_argument("--temperature", type=float, default=None,
                        help="default: unset (as in the original run)")
    parser.add_argument("--api", choices=["responses", "chat_completions"], default="responses",
                        help="OpenAI API the SDK calls (default: responses, the SDK default)")
    parser.add_argument("--mcp-url", default=None, help="default: MCP_SERVER_URL in .env, else the local mirror")
    parser.add_argument("--system-prompt", choices=sorted(SYSTEM_PROMPTS), default="default")
    parser.add_argument("--dump-inputs", type=Path, default=None,
                        help="JSONL file recording what the model receives on each LLM call")
    args = parser.parse_args()
    set_default_openai_api(args.api)
    config = {"temperature": args.temperature, "api": args.api, "mcp_url": args.mcp_url or mcp_config.mcp_url(),
              "system_prompt": args.system_prompt}

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

    logger.info(mcp_config.describe())
    logger.info("OpenAI Agent | provider=%s model=%s | %d questions → %s | config=%s",
                args.provider, model_id, len(questions), out_path, config)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            try:
                result = run_openai_agent(
                    question=q["question"],
                    question_id=q["id"],
                    provider=args.provider,
                    model_id=model_id,
                    temperature=args.temperature,
                    mcp_url=args.mcp_url,
                    dump_inputs=args.dump_inputs,
                    system_prompt=args.system_prompt,
                )
                result["config"]           = config
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
