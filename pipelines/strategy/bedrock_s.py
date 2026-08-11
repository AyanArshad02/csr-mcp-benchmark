"""pipelines/strategy/bedrock_s.py — AWS Bedrock strategy (Claude Haiku 4.5)."""

from __future__ import annotations

import hashlib
import json
import logging
import time

import os

import boto3
from dotenv import load_dotenv

from .base import LLMStrategy

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


class BedrockStrategy(LLMStrategy):
    """Wraps AWS Bedrock InvokeModel for Claude, with diskcache + exponential-backoff retry.

    Uses credentials from ~/.aws/credentials (no extra env vars needed beyond AWS_REGION).
    """

    def __init__(self, model: str = "us.anthropic.claude-sonnet-4-6",
                 region: str = "us-east-1", cache=None):
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model = model
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "bedrock"

    def chat(self, messages: list[dict], *, temperature: float = 0.0, **kwargs) -> str:
        cache_key = _cache_key("bedrock", self._model, messages, temperature, kwargs)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Separate system message from the rest (Bedrock requires system at top level)
        system_content = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_content:
            body["system"] = system_content

        delay = _RETRY_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.invoke_model(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                text = result["content"][0]["text"]
                if self._cache is not None:
                    self._cache.set(cache_key, text)
                return text
            except self._client.exceptions.ThrottlingException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("Bedrock throttled, retrying in %.1fs", delay)
                time.sleep(delay)
                delay *= 2
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("Bedrock error (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("Bedrock chat: all retries exhausted")


def _cache_key(provider: str, model: str, messages: list[dict],
               temperature: float, extra: dict) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, "messages": messages,
         "temperature": temperature, "extra": extra},
        sort_keys=True,
    )
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()
