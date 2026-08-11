"""pipelines/strategy/openrouter_s.py — OpenRouter strategy (NVIDIA Nemotron free).

OpenRouter exposes an OpenAI-compatible API so we reuse the openai SDK,
just pointing it at a different base_url.
"""

from __future__ import annotations

import logging
import os
import time

import openai

from .base import LLMStrategy
from .openai_s import _cache_key

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 3
_RETRY_DELAY = 4.0   # free-tier models can be slower; give more breathing room


class OpenRouterStrategy(LLMStrategy):
    """Wraps OpenRouter's OpenAI-compatible API with diskcache + retry."""

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        cache=None,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE,
        )
        self._model = model
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openrouter"

    def chat(self, messages: list[dict], *, temperature: float = 0.0, **kwargs) -> str:
        cache_key = _cache_key("openrouter", self._model, messages, temperature, kwargs)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        delay = _RETRY_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                text = response.choices[0].message.content or ""
                if self._cache is not None:
                    self._cache.set(cache_key, text)
                return text
            except openai.RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("OpenRouter rate limit, retrying in %.1fs", delay)
                time.sleep(delay)
                delay *= 2
            except openai.APIError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("OpenRouter API error (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("OpenRouter chat: all retries exhausted")
