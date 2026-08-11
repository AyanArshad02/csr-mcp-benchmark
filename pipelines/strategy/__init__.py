"""pipelines/strategy — provider-agnostic LLM strategy layer.

Usage:
    from pipelines.strategy import get_strategy
    import diskcache

    cache = diskcache.Cache(".cache/llm")
    strategy = get_strategy("openai", cache=cache)          # gpt-4o-mini
    strategy = get_strategy("openrouter", cache=cache)      # nvidia nemotron free
    reply = strategy.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

from .base import LLMStrategy
from .openai_s import OpenAIStrategy
from .openrouter_s import OpenRouterStrategy
from .bedrock_s import BedrockStrategy

_OPENAI_DEFAULT     = "gpt-4o-mini"
_OPENROUTER_DEFAULT = "nvidia/nemotron-3-ultra-550b-a55b:free"
_BEDROCK_DEFAULT    = "us.anthropic.claude-sonnet-4-6"


def get_strategy(provider: str, *, model: str | None = None, cache=None) -> LLMStrategy:
    """Factory: return the right LLMStrategy for the given provider.

    provider: "openai" | "openrouter" | "bedrock"
    model:    optional override (defaults to the locked model for each provider)
    cache:    a diskcache.Cache instance (pass None to disable caching)
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIStrategy(model=model or _OPENAI_DEFAULT, cache=cache)
    if provider == "openrouter":
        return OpenRouterStrategy(model=model or _OPENROUTER_DEFAULT, cache=cache)
    if provider == "bedrock":
        return BedrockStrategy(model=model or _BEDROCK_DEFAULT, cache=cache)
    raise ValueError(
        f"Unknown provider {provider!r}. Expected 'openai', 'openrouter', or 'bedrock'."
    )


__all__ = ["LLMStrategy", "OpenAIStrategy", "OpenRouterStrategy", "BedrockStrategy", "get_strategy"]
