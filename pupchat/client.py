"""The provider-agnostic half: prepare media, call the chosen provider, account for it.

Nothing here knows how a provider formats a request. The only branch on provider is
`provider.media_kind`, because that is the one real difference between them:

    OpenAI  frames   no video input exists in the API, so Pupchat samples
    Gemini  video    the file goes up whole and Gemini samples it server-side

The system prompt and the evidence summary are identical either way. If they were not,
comparing the two providers would compare two prompts rather than two models.
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .config import PRICING, PROMPT_PATH, USAGE_LOG, Config, load_env
from .config import system_prompt as _system_prompt
from .pipeline import analyse, cache_dir, probe
from .providers import MEDIA_VIDEO, get_provider


def system_prompt(path: Path = PROMPT_PATH) -> str:
    return _system_prompt(path)


def sample_vision_frames(video: str | Path, cfg: Config,
                         progress=print) -> list[tuple[float, str]]:
    """(timestamp, base64 jpeg) for a frame-based provider.

    Deliberately much sparser than the keypoint pass. Conflating the two rates would
    either blow the token budget or starve the temporal features.
    """
    src_fps, _, duration = probe(video)
    rate = cfg.vision.fps_for(duration)
    stride = max(1, int(round(src_fps / rate)))

    cap = cv2.VideoCapture(str(video))
    out: list[tuple[float, str]] = []
    i = 0
    while len(out) < cfg.vision.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if w != cfg.vision.width:
                frame = cv2.resize(
                    frame, (cfg.vision.width, int(round(h * cfg.vision.width / w))),
                    interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.vision.jpeg_quality])
            if ok2:
                out.append((round(i / src_fps, 2),
                            base64.b64encode(buf.tobytes()).decode("ascii")))
        i += 1
    cap.release()
    progress(f"[vision] {len(out)} frames at {rate:g} fps "
             f"({duration:.1f}s clip)")
    return out


def prepare_media(video: Path, provider, cfg: Config, vhash: str, progress=print):
    """Whatever the provider takes. Frames are cached; a video path needs nothing."""
    if provider.media_kind == MEDIA_VIDEO:
        progress(f"[vision] sending {video.name} whole "
                 f"({video.stat().st_size / 1e6:.1f} MB) - Gemini samples it")
        return video
    frames_path = cache_dir(vhash) / "vision_frames.json"
    if frames_path.exists():
        frames = [(t, b) for t, b in json.loads(frames_path.read_text())]
        progress(f"[vision] reusing {len(frames)} cached frames")
        return frames
    frames = sample_vision_frames(video, cfg, progress)
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_text(json.dumps(frames), encoding="utf-8")
    return frames


def estimate_cost(model: str, usage: dict) -> float:
    """USD from the normalised usage dict. NaN for an unpriced model - zero would
    quietly understate a bill."""
    price = PRICING.get(model)
    if not price:
        return float("nan")
    inp = usage.get("input_tokens", 0) or 0
    cached = usage.get("cached_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    fresh = max(0, inp - cached)
    return (fresh * price["input"] + cached * price["cached"]
            + out * price["output"]) / 1_000_000.0


def log_usage(entry: dict, path: Path | None = None) -> None:
    path = Path(path) if path is not None else USAGE_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def ask_pupchat(video_path: str | Path, question: str, history: list | None = None,
                cfg: Config | None = None, *, progress=print,
                show_request: bool = False, client=None) -> dict:
    """video + question -> answer, through whichever provider cfg selects.

    `client` is injectable so tests mock the API without a key and without spending.
    """
    load_env()
    cfg = cfg or Config()
    video_path = Path(video_path)
    provider = get_provider(cfg.provider)
    settings = cfg.active

    result = analyse(video_path, cfg, progress=progress)
    evidence_text = result["evidence_text"]
    vhash = result["hash"]

    media = prepare_media(video_path, provider, cfg, vhash, progress)
    payload = provider.build_request(evidence_text, question, media, cfg, history)
    redacted = provider.redact(payload)

    if show_request:
        progress(f"\n--- request to {provider.name} "
                 f"(media shortened; no key in the payload) ---")
        progress(json.dumps(redacted, indent=2)[:4000])
        progress("--- end request ---\n")

    progress(f"[{provider.name}] {settings.model}")
    t0 = time.time()
    res = provider.send(payload, cfg, client=client)
    elapsed = time.time() - t0

    cost = estimate_cost(settings.model, res.usage)
    log_usage({
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider.name, "model": settings.model,
        "video": video_path.name, "hash": vhash,
        "question": question[:200],
        "media": (provider.media_kind if provider.media_kind == MEDIA_VIDEO
                  else f"{len(media)} frames"),
        "usage": res.usage, "cost_usd": round(cost, 6) if cost == cost else None,
        "seconds": round(elapsed, 2), "follow_up": bool(history),
    })

    return {
        "answer": res.answer, "evidence_text": evidence_text, "request": redacted,
        "usage": res.usage, "cost_usd": cost, "seconds": elapsed, "hash": vhash,
        "provider": provider.name, "model": settings.model,
        "history": provider.extend_history(history, question, res.answer),
    }


def build_payload_only(video_path: str | Path, question: str,
                       cfg: Config | None = None) -> dict:
    """The exact request, redacted, without sending it or needing a key.

    Used by `--dry-run`. Keeps request inspection honest: what this prints is what
    `ask_pupchat` would send, built by the same provider method, not a reconstruction.
    """
    load_env()
    cfg = cfg or Config()
    video_path = Path(video_path)
    provider = get_provider(cfg.provider)
    result = analyse(video_path, cfg, progress=lambda *a, **k: None)
    media = prepare_media(video_path, provider, cfg, result["hash"],
                          progress=lambda *a, **k: None)
    return provider.redact(
        provider.build_request(result["evidence_text"], question, media, cfg))
