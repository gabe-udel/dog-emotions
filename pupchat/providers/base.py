"""What Pupchat needs from an AI provider, and the one way they genuinely differ.

The difference is not the model name - that is config. It is **what media the provider
can accept**:

    OpenAI  MEDIA_FRAMES   no video input exists, so frames are sampled client-side
    Gemini  MEDIA_VIDEO    takes the file itself and samples frames server-side

Everything above this line is shared: the same system prompt, the same evidence summary,
the same question-last ordering. Swapping providers must change who interprets the
video, never what they are told about it - otherwise a comparison between them measures
two different prompts rather than two different models.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

MEDIA_FRAMES = "frames"
"""Media is a list of (timestamp_seconds, base64_jpeg)."""
MEDIA_VIDEO = "video"
"""Media is a Path to the video file."""


@dataclass
class ProviderResult:
    answer: str
    usage: dict = field(default_factory=dict)
    """Normalised across providers: input_tokens, output_tokens, cached_tokens.
    Each provider reports a different shape; cost accounting needs one."""
    raw: object = None


class Provider(abc.ABC):
    name: str = ""
    media_kind: str = MEDIA_FRAMES

    @abc.abstractmethod
    def build_request(self, evidence_text: str, question: str, media,
                      cfg, history: list | None = None) -> dict:
        """Assemble the provider's payload. Question LAST, always."""

    @abc.abstractmethod
    def send(self, payload: dict, cfg, client=None) -> ProviderResult:
        """Call the API. `client` is injectable so tests never need a key."""

    @abc.abstractmethod
    def redact(self, payload: dict) -> dict:
        """The payload with media shortened, safe to print and to log."""

    def extend_history(self, history: list | None, question: str,
                       answer: str) -> list:
        """Provider-neutral transcript. Overridden where the wire format differs."""
        out = list(history or [])
        out.append({"role": "user", "text": question})
        out.append({"role": "assistant", "text": answer})
        return out


def normalise_usage(input_tokens=0, output_tokens=0, cached_tokens=0) -> dict:
    return {"input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cached_tokens": int(cached_tokens or 0)}
