"""HTTP API, so a front end built elsewhere can drive Pupchat.

Google Stitch exports a FRONT END only - HTML/CSS, Tailwind, Vue, Angular, Flutter or
SwiftUI. It generates no server, so a Stitch design has nothing to call until something
exposes the pipeline over HTTP. That is all this module is.

    .venv\\Scripts\\python.exe -m pupchat.api            # http://127.0.0.1:8000

CORS is wide open because a Stitch export opened from the filesystem or served on
another port is a different origin, and a prototype that silently fails CORS looks like
a broken backend. That is safe here only because this is a local, non-commercial
prototype and must not be deployed - see the licence note.

Endpoints are shaped around the two costs: measuring a video is minutes of CPU, asking
a question is seconds. So analysis happens once per file hash and every later call
refers to the hash rather than re-uploading.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile


def os_getenv(name: str) -> str:
    """Whether a key is present - never the key itself. The page needs to know it can
    ask a question; it must never learn the secret."""
    return os.getenv(name, "").strip()
from pathlib import Path

# Module scope, not inside build_app: `from __future__ import annotations` turns the
# endpoint annotations into strings, and FastAPI resolves them against module globals.
# Imported locally they are invisible there, and UploadFile fails to resolve at runtime.
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import LICENCE_NOTICE, Config, load_env
from .pipeline import analyse, cache_dir, probe, video_hash

WEBAPP = Path(__file__).resolve().parent / "webapp"

UPLOADS = Path(tempfile.gettempdir()) / "pupchat_uploads"


def _store_upload(name: str, data: bytes) -> Path:
    """Keep the bytes under a content hash, so the same clip uploaded twice is one
    file and hits the same analysis cache."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    suffix = Path(name).suffix or ".mp4"
    dest = UPLOADS / f"{digest}{suffix}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest


def json_safe(obj):
    """NaN and Inf -> None, recursively.

    Python's json.dumps writes bare NaN happily; Starlette's encoder refuses it, and
    so does every strict JSON parser a front end is likely to use. A missing
    measurement is genuinely null, so this is the correct shape as well as the
    serialisable one - and `null` is what a UI can test for, where NaN is not.
    """
    import math as _m
    if isinstance(obj, float):
        return None if (_m.isnan(obj) or _m.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def check_duration(path: Path, cfg: Config) -> float:
    """Enforce the clip-length window server-side. Returns the duration.

    The page checks this too, for instant feedback, but a browser check is a
    convenience and not a guarantee - anything can POST to this endpoint.
    """
    u = cfg.upload
    try:
        _, _, duration = probe(path)
    except SystemExit as e:
        raise ValueError(f"could not read that video: {e}")
    if duration <= 0:
        raise ValueError("could not read the length of that video")
    if duration < u.min_seconds - u.tolerance_s:
        raise ValueError(
            f"that clip is {duration:.1f}s. Pupchat needs {u.min_seconds:g}-"
            f"{u.max_seconds:g}s: shorter than {u.min_seconds:g}s has too few frames "
            f"for movement to be measurable.")
    if duration > u.max_seconds + u.tolerance_s:
        raise ValueError(
            f"that clip is {duration:.1f}s. Pupchat needs {u.min_seconds:g}-"
            f"{u.max_seconds:g}s: longer clips take too long to measure on CPU. "
            f"Trim it to the moment you want to ask about.")
    return duration


def _resolve(video: str | None, vhash: str | None) -> Path:
    """A request may name an uploaded path or a hash from an earlier analysis."""
    if video:
        p = Path(video)
        if p.exists():
            return p
    if vhash:
        for p in UPLOADS.glob(f"*"):
            if video_hash(p) == vhash:
                return p
    raise FileNotFoundError(
        "no video for this request: upload it to /api/analyse first, then send its "
        "hash")


def build_app():
    load_env()
    app = FastAPI(title="Pupchat API", version="0.1",
                  description=LICENCE_NOTICE)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/api/health")
    def health():
        cfg = Config()
        return {"ok": True, "provider": cfg.provider, "model": cfg.active.model,
                "licence": LICENCE_NOTICE,
                "upload": {"min_seconds": cfg.upload.min_seconds,
                           "max_seconds": cfg.upload.max_seconds,
                           "max_bytes": cfg.upload.max_bytes},
                "key_configured": bool(os_getenv(cfg.active.key_env))}

    @app.post("/api/analyse")
    async def api_analyse(file: UploadFile = File(...)):
        """Upload a clip, measure it, get the evidence. Minutes of CPU the first time,
        instant afterwards - the result is cached by the file's own hash."""
        cfg = Config()
        data = await file.read()
        if len(data) > cfg.upload.max_bytes:
            raise HTTPException(413, f"that file is "
                                     f"{len(data) / 1e6:.0f} MB; the limit is "
                                     f"{cfg.upload.max_bytes / 1e6:.0f} MB")
        path = _store_upload(file.filename or "clip.mp4", data)
        try:
            duration = check_duration(path, cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        res = analyse(path, cfg, progress=lambda *a, **k: None)
        return json_safe({
            "hash": res["hash"],
            "duration_s": round(duration, 2),
            "cached": res["cached"],
            "meta": res["meta"],
            "evidence_text": res["evidence_text"],
            "evidence": res["evidence"],
        })

    @app.post("/api/ask")
    async def api_ask(question: str = Form(...), hash: str = Form(None),
                      video: str = Form(None), history: str = Form(None),
                      profile: str = Form(None)):
        """Ask about an already-analysed clip. `history` is the JSON list returned by
        the previous call, so follow-ups cost no re-measurement."""
        from .client import ask_pupchat
        try:
            path = _resolve(video, hash)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        turns = json.loads(history) if history else None
        try:
            res = ask_pupchat(path, question, history=turns, cfg=Config(),
                              progress=lambda *a, **k: None,
                              profile=profile or None)
        except SystemExit as e:                      # missing key, bad provider
            raise HTTPException(status_code=503, detail=str(e))
        return json_safe({
            "answer": res["answer"], "history": res["history"],
            "provider": res["provider"], "model": res["model"],
            "usage": res["usage"], "cost_usd": res["cost_usd"],
            "seconds": res["seconds"], "truncated": res.get("truncated", False),
            "evidence_text": res["evidence_text"], "hash": res["hash"],
        })

    @app.get("/api/evidence/{vhash}")
    def api_evidence(vhash: str):
        d = cache_dir(vhash)
        if not (d / "evidence.json").exists():
            raise HTTPException(404, "not analysed yet")
        return json_safe({
            "hash": vhash,
            "evidence": json.loads((d / "evidence.json").read_text()),
            "evidence_text": (d / "evidence.txt").read_text(encoding="utf-8"),
            "clip": json.loads((d / "clip.json").read_text()),
        })

    @app.get("/api/track/{vhash}")
    def api_track(vhash: str, features: str = None):
        """Per-frame feature values with timestamps, for a UI that scrubs the video."""
        from .overlay import feature_track
        if not (cache_dir(vhash) / "frames.json").exists():
            raise HTTPException(404, "not analysed yet")
        names = [f.strip() for f in features.split(",")] if features else None
        return json_safe(feature_track(vhash, names))

    @app.get("/api/trace/{vhash}")
    def api_trace(vhash: str, frame: int = None):
        """Keypoint confidences joined to feature values: why a feature is missing."""
        from .trace import frame_trace, keypoint_coverage, load_keypoints
        d = cache_dir(vhash)
        if not (d / "keypoints.npz").exists():
            raise HTTPException(404, "not analysed yet")
        cfg = Config()
        kp = load_keypoints(d / "keypoints.npz")
        rows = json.loads((d / "frames.json").read_text())
        idx = len(rows) // 2 if frame is None else max(0, min(len(rows) - 1, frame))
        cov = keypoint_coverage(kp, cfg.keypoint.pcut)
        return json_safe({
            "hash": vhash, "pcut": cfg.keypoint.pcut,
            "coverage": [{"group": g, "name": n, "mean_conf": m, "above_cut": f}
                         for g, n, m, f in cov],
            "frame": frame_trace(kp, rows, idx, cfg.keypoint.pcut),
        })

    @app.get("/api/overlay/{vhash}")
    def api_overlay(vhash: str, force: bool = False):
        """The video with keypoints and the live feature panel drawn on it."""
        from .overlay import render_overlay
        if not (cache_dir(vhash) / "frames.json").exists():
            raise HTTPException(404, "not analysed yet")
        path = render_overlay(vhash, force=force, progress=lambda *a, **k: None)
        return FileResponse(path, media_type="video/mp4",
                            filename=f"pupchat_overlay_{vhash}.mp4")

    @app.post("/api/flag")
    async def api_flag(question: str = Form(...), answer: str = Form(...),
                       note: str = Form(""), hash: str = Form(None),
                       video: str = Form(None), evidence_text: str = Form("")):
        from .feedback import flag_answer
        try:
            path = _resolve(video, hash)
        except FileNotFoundError:
            path = Path(video or "unknown.mp4")
        cfg = Config()
        rec = flag_answer(path, question, answer, note,
                          evidence_text=evidence_text, provider=cfg.provider,
                          model=cfg.active.model, video_hash=hash or "")
        return JSONResponse(rec)

    @app.get("/api/flags")
    def api_flags():
        from .feedback import load_flags
        return {"flags": load_flags()}

    # The front end last, so /api/* always wins over a file of the same name.
    if WEBAPP.exists():
        app.mount("/", StaticFiles(directory=str(WEBAPP), html=True), name="webapp")

    return app


def main() -> None:
    import uvicorn
    print("=" * 74)
    print("Pupchat API   http://127.0.0.1:8000     docs: /docs")
    print(LICENCE_NOTICE)
    print("=" * 74)
    uvicorn.run(build_app(), host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
