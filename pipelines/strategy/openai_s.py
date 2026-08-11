"""pipelines/strategy/openai_s.py — OpenAI strategy (gpt-4o-mini)."""

from __future__ import annotations

import hashlib
import json
import logging
import time

import openai

from .base import LLMStrategy

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0   # seconds; doubles on each retry


class OpenAIStrategy(LLMStrategy):
    """Wraps openai.OpenAI with diskcache + exponential-backoff retry."""

    def __init__(self, model: str = "gpt-4o-mini", cache=None):
        self._client = openai.OpenAI()   # reads OPENAI_API_KEY from env
        self._model = model
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    def chat(self, messages: list[dict], *, temperature: float = 0.0, **kwargs) -> str:
        cache_key = _cache_key("openai", self._model, messages, temperature, kwargs)

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
                logger.warning("OpenAI rate limit, retrying in %.1fs", delay)
                time.sleep(delay)
                delay *= 2
            except openai.APIError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("OpenAI API error (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("OpenAI chat: all retries exhausted")


def _cache_key(provider: str, model: str, messages: list[dict],
               temperature: float, extra: dict) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, "messages": messages,
         "temperature": temperature, "extra": extra},
        sort_keys=True,
    )
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()
