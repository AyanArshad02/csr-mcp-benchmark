"""pipelines/strategy/base.py — Abstract LLM strategy interface.

All pipeline code calls strategy.chat(messages) — never a provider SDK directly.
This keeps provider-specific details (keys, base URLs, retry logic) in one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMStrategy(ABC):
    """Provider-agnostic LLM interface used by all three pipelines."""

    @abstractmethod
    def chat(self, messages: list[dict], *, temperature: float = 0.0, **kwargs) -> str:
        """Send messages and return the assistant's text reply."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The exact model string sent to the API (used in result logs for reproducibility)."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Short provider name, e.g. 'openai' or 'openrouter'."""
        ...
