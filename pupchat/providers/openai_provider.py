"""OpenAI: sampled frames, because the API has no video input.

Checked against the official docs on 2026-10-11.
https://developers.openai.com/api/docs/guides/images-vision documents exactly two input
content types, `input_text` and `input_image` (URL, base64 data URL, or a Files API id,
up to 1,500 images per request). There is no `video` content type, and
https://developers.openai.com/changelog/ has no entry adding one - its video coverage is
all Sora generation. Client-side frame sampling is the documented path, not a workaround.

Follow-ups reuse the frames rather than re-sending new ones. They sit in a stable prefix
at the front of every request, so the provider's prompt cache covers them: on
gpt-6.1-sol cached input is $0.10/Mtok against $2.00, so a follow-up costs a fraction of
the first question.
"""
from __future__ import annotations

import copy

from ..config import system_prompt
from .base import (MEDIA_FRAMES, Provider, ProviderResult,
                   normalise_usage)


class OpenAIProvider(Provider):
    name = "openai"
    media_kind = MEDIA_FRAMES

    def build_request(self, evidence_text, question, media, cfg,
                      history=None, profile=None) -> dict:
        settings = cfg.openai
        content: list[dict] = []
        if profile:
            content.append({"type": "input_text", "text": profile})
        content.append({
            "type": "input_text",
            "text": "Automated measurements from this video:\n\n" + evidence_text,
        })
        for t, b64 in media:
            content.append({"type": "input_text", "text": f"Frame at {t:g}s:"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            })
        # The question goes last, after the frames.
        content.append({"type": "input_text", "text": f"The owner asks: {question}"})

        messages: list[dict] = [{"role": "user", "content": content}]
        if history:
            # Drop the question from the cached prefix and replay the transcript, so
            # the frames stay at a byte-identical offset and the cache still hits.
            messages[0]["content"] = content[:-1]
            for turn in history:
                kind = ("output_text" if turn["role"] == "assistant" else "input_text")
                messages.append({"role": turn["role"],
                                 "content": [{"type": kind, "text": turn["text"]}]})
            messages.append({"role": "user",
                             "content": [{"type": "input_text", "text": question}]})

        return {
            "model": settings.model,
            "instructions": system_prompt(),
            "input": messages,
            "max_output_tokens": settings.max_output_tokens,
        }

    def send(self, payload, cfg, client=None) -> ProviderResult:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=cfg.openai.key(), timeout=cfg.openai.timeout_s)
        resp = client.responses.create(**payload)

        answer = getattr(resp, "output_text", None) or str(resp)
        u = getattr(resp, "usage", None)
        u = (u.model_dump() if hasattr(u, "model_dump") else dict(u) if u else {})
        # OpenAI's output_tokens ALREADY includes reasoning tokens, so unlike Gemini
        # they must not be added again - reasoning is reported here for transparency.
        return ProviderResult(
            answer=answer,
            usage=normalise_usage(
                u.get("input_tokens"), u.get("output_tokens"),
                (u.get("input_tokens_details") or {}).get("cached_tokens"),
                (u.get("output_tokens_details") or {}).get("reasoning_tokens")),
            raw=resp,
            finish_reason=(getattr(resp, "incomplete_details", None)
                           and getattr(resp.incomplete_details, "reason", "")) or "")

    def redact(self, payload, keep_chars: int = 24) -> dict:
        p = copy.deepcopy(payload)
        for msg in p.get("input", []):
            for part in msg.get("content", []):
                if part.get("type") == "input_image":
                    url = part.get("image_url", "")
                    part["image_url"] = url[:keep_chars] + f"...[{len(url)} chars]"
        return p

