"""Gemini: the video file itself, no client-side frame sampling.

Checked against the official docs on 2026-10-11,
https://ai.google.dev/gemini-api/docs/video-understanding. Two paths:

* **inline** - base64 bytes in the request, for files under about 100 MB
* **Files API** - `client.files.upload(file=...)`, recommended above 100 MB or over
  10 minutes, and reusable across requests

Accepted types include video/mp4, video/mpeg, video/mov, video/avi, video/webm,
video/wmv and video/3gpp. All current Gemini models process video.

This is the real difference between the two providers, and the reason the abstraction
exists: Gemini samples the video server-side, so it sees frames Pupchat never chose.
Our evidence summary still describes the frames the KEYPOINT pass measured, which is a
different sampling again. Where the model's reading and the measurements disagree, that
mismatch is one candidate explanation, and the system prompt already tells the model to
surface disagreement rather than silently pick a side.
"""
from __future__ import annotations

import base64
import copy
from pathlib import Path

from ..config import system_prompt
from .base import MEDIA_VIDEO, Provider, ProviderResult, normalise_usage

INLINE_LIMIT_BYTES = 100 * 1024 * 1024
"""Above this the docs recommend the Files API."""

_MIME = {
    ".mp4": "video/mp4", ".mpeg": "video/mpeg", ".mpg": "video/mpg",
    ".mov": "video/mov", ".avi": "video/avi", ".flv": "video/x-flv",
    ".webm": "video/webm", ".wmv": "video/wmv", ".3gp": "video/3gpp",
}


def mime_for(path: Path) -> str:
    """Only the documented containers.

    Deliberately no `mimetypes.guess_type` fallback. It resolves .mkv to
    video/x-matroska, which starts with "video/" and looks fine - but Gemini's docs do
    not list it, so the request would be built, uploaded and then rejected server-side
    with an error that points at the API rather than at the file. Refusing here names
    the actual problem and costs nothing.
    """
    ext = path.suffix.lower()
    if ext in _MIME:
        return _MIME[ext]
    raise SystemExit(
        f"{path.name}: Gemini does not document a video MIME type for '{ext}'. "
        f"Supported: {', '.join(sorted(_MIME))}. Convert it first, e.g. with ffmpeg.")


class GeminiProvider(Provider):
    name = "gemini"
    media_kind = MEDIA_VIDEO

    def build_request(self, evidence_text, question, media, cfg,
                      history=None) -> dict:
        """`media` is the video path. Order matches OpenAI: evidence, media, question."""
        video = Path(media)
        size = video.stat().st_size
        mime = mime_for(video)

        if size <= INLINE_LIMIT_BYTES:
            part = {"inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(video.read_bytes()).decode("ascii")}}
        else:
            # Resolved to a real uploaded handle in send(); kept symbolic here so the
            # request can be built, inspected and tested without network access.
            part = {"file_upload": {"path": str(video), "mime_type": mime,
                                    "bytes": size}}

        contents: list[dict] = [{
            "role": "user",
            "parts": [
                {"text": "Automated measurements from this video:\n\n" + evidence_text},
                part,
                {"text": f"The owner asks: {question}"},
            ],
        }]

        if history:
            contents[0]["parts"] = contents[0]["parts"][:-1]
            for turn in history:
                contents.append({"role": "model" if turn["role"] == "assistant"
                                 else "user",
                                 "parts": [{"text": turn["text"]}]})
            contents.append({"role": "user", "parts": [{"text": question}]})

        return {
            "model": cfg.gemini.model,
            "contents": contents,
            "config": {
                "system_instruction": system_prompt(),
                "max_output_tokens": cfg.gemini.max_output_tokens,
            },
        }

    def send(self, payload, cfg, client=None) -> ProviderResult:
        if client is None:
            from google import genai
            client = genai.Client(api_key=cfg.gemini.key())

        payload = copy.deepcopy(payload)
        for content in payload["contents"]:
            for i, part in enumerate(content.get("parts", [])):
                if "file_upload" in part:
                    spec = part["file_upload"]
                    uploaded = client.files.upload(file=spec["path"])
                    content["parts"][i] = uploaded

        resp = client.models.generate_content(**payload)

        answer = getattr(resp, "text", None) or str(resp)
        um = getattr(resp, "usage_metadata", None)

        def g(*names):
            for n in names:
                v = getattr(um, n, None) if um is not None else None
                if v is None and isinstance(um, dict):
                    v = um.get(n)
                if v is not None:
                    return v
            return 0

        return ProviderResult(
            answer=answer,
            usage=normalise_usage(
                g("prompt_token_count", "input_tokens"),
                g("candidates_token_count", "output_tokens"),
                g("cached_content_token_count", "cached_tokens")),
            raw=resp)

    def redact(self, payload, keep_chars: int = 24) -> dict:
        p = copy.deepcopy(payload)
        for content in p.get("contents", []):
            for part in content.get("parts", []):
                if "inline_data" in part:
                    d = part["inline_data"].get("data", "")
                    part["inline_data"]["data"] = (
                        d[:keep_chars] + f"...[{len(d)} chars]")
        return p
