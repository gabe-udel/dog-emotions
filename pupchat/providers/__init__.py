"""Provider registry. `cfg.provider` selects; nothing else in Pupchat branches on it."""
from __future__ import annotations

from .base import MEDIA_FRAMES, MEDIA_VIDEO, Provider, ProviderResult
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

_REGISTRY: dict[str, type[Provider]] = {
    OpenAIProvider.name: OpenAIProvider,
    GeminiProvider.name: GeminiProvider,
}

__all__ = ["Provider", "ProviderResult", "MEDIA_FRAMES", "MEDIA_VIDEO",
           "get_provider", "available"]


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str) -> Provider:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise SystemExit(
            f"unknown provider {name!r}; expected one of {available()}. "
            f"Set PUPCHAT_PROVIDER in .env or pass --provider.") from None
