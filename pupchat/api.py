"""HTTP API, so a front end built elsewhere can drive Pupchat.

Google Stitch exports a FRONT END only - HTML/CSS, Tailwind, Vue, Angular, Flutter or
SwiftUI. It generates no server, so a Stitch design has nothing to call until something
exposes the pipeline over HTTP. That is all this module is.

    .venv\\Scripts\\python.exe -m pupchat.api            # http://127.0.0.1:8000

CORS is wide open because a Stitch export opened from the filesystem or served on
another port is a different origin, and a prototype that silently fails CORS looks like
a broken backend. Open CORS is not an authorisation hole here - there is nothing to
authorise and no cookie to ride - but it does mean any page can drive this server, so
the spend caps in `guard` are what stand between a shared URL and a bill.

Endpoints are shaped around the two costs: measuring a video is minutes of CPU, asking
a question is seconds. So analysis happens once per file hash and every later call
refers to the hash rather than re-uploading.

Both costs are paid in a worker thread behind a one-at-a-time queue. `analyse()` is
~93 s of blocking CPU, and awaiting it directly on the event loop stalled every other
request - including the page itself - for the whole measurement.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import tempfile
import time


def os_getenv(name: str) -> str:
    """Whether a key is present - never the key itself. The page needs to know it can
    ask a question; it must never learn the secret."""
    return os.getenv(name, "").strip()
from pathlib import Path, PurePosixPath

# Module scope, not inside build_app: `from __future__ import annotations` turns the
# endpoint annotations into strings, and FastAPI resolves them against module globals.
# Imported locally they are invisible there, and UploadFile fails to resolve at runtime.
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import (ATTRIBUTION, LICENCE_NOTICE, OUT_DIR, Config, load_env,
                     share_mode)
from .guard import (AnalysisQueue, Budget, BudgetExceeded, QueueFull, client_ip,
                    is_local, read_capped, sweep)
from .jobs import JobStore, estimate, record_timing
from .library import PHOTO_DIR, library, profile_text
from .pipeline import analyse, cache_dir, probe, video_hash

WEBAPP = Path(__file__).resolve().parent / "webapp"

# Uploads live under outputs/, NOT in the system temp directory.
#
# Two reasons, both learned the hard way. tempfile.gettempdir() resolves differently
# depending on how the process was started, so a clip uploaded by one process was
# invisible to the server that had to serve it back - /api/video returned 404 for a file
# that was plainly on disk. And Windows cleans temp, which would silently break playback
# in chats the browser still lists.
UPLOADS = OUT_DIR / "uploads"

# Anything already in the old location is still found, so existing chats keep working.
LEGACY_UPLOADS = Path(tempfile.gettempdir()) / "pupchat_uploads"

OWNER_COOKIE = "pupchat_owner"
PHOTO_MAX_BYTES = 10 * 1024 * 1024


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
    """A request may name an uploaded path or a hash from an earlier analysis.

    Uploads are stored under their own hash, so the common case is a direct filename
    lookup rather than hashing every file in the directory - which was O(n) reads per
    request and grew with every clip ever uploaded.
    """
    if video:
        p = Path(video)
        if p.exists():
            return p
    if vhash:
        for folder in (UPLOADS, LEGACY_UPLOADS):
            if not folder.exists():
                continue
            hits = sorted(folder.glob(f"{vhash}.*"))
            if hits:
                return hits[0]
        for folder in (UPLOADS, LEGACY_UPLOADS):      # fallback: renamed by hand
            if not folder.exists():
                continue
            for p in folder.glob("*"):
                if p.is_file() and video_hash(p) == vhash:
                    return p
    raise FileNotFoundError(
        "no video for this request: upload it to /api/analyse first, then send its "
        "hash")


class SpaFiles(StaticFiles):
    """Static files, plus: a path the front end owns gets the page.

    The browser owns /chat/<id>, /dogs/<name> and /dogs/new. No such files exist, so a
    plain mount answers 404 - which means refreshing the page, or opening a link someone
    shared, lands on "Not Found" while the app is running perfectly well.

    It has to be a subclass rather than an extra route. `Mount("/")` matches every path
    and Starlette stops at the first match, so a route registered after the mount is
    unreachable, and one registered before it would shadow the real files.

    Two paths must NOT get the page, and both were found by their tests:

    * anything that looks like a file. Serving index.html for a mistyped
      /assets/foo.woff2 returns HTML with status 200, and the browser then reports a font
      parsing error instead of the 404 that names the actual problem.
    * anything under /api/. Only the API routes that EXIST are registered ahead of this
      mount; an unknown one falls through to here, and /api/typo has no dot in it, so
      without this guard a client expecting JSON gets a web page with status 200.
    """

    SPA_EXCLUDE = ("api",)

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            # StaticFiles hands this back in the HOST's separators, so on Windows
            # /api/nope arrives as "api\\nope" and a check against "api/" silently
            # never matches. Normalise before looking at it.
            parts = PurePosixPath(path.replace("\\", "/")).parts
            if (e.status_code != 404
                    or "." in (parts[-1] if parts else "")
                    or (parts and parts[0] in self.SPA_EXCLUDE)):
                raise
            return await super().get_response("index.html", scope)


def frames_for(path: Path, cfg: Config) -> int | None:
    """How many frames the keypoint pass will measure, known before it starts.

    Mirrors pipeline.sample_frames' striding, which is what makes the first estimate a
    calculation rather than a guess about clip length. Returns None if the video cannot
    be probed; the estimate then falls back to a default rather than failing the request.
    """
    try:
        src_fps, n_src, _ = probe(path)
    except Exception:                               # noqa: BLE001 - estimate only
        return None
    if src_fps <= 0 or n_src <= 0:
        return None
    stride = max(1, int(round(src_fps / max(cfg.keypoint.target_fps, 0.1))))
    return max(1, min(cfg.keypoint.max_frames, math.ceil(n_src / stride)))


def build_app():
    load_env()
    app = FastAPI(title="Pupchat API", version="0.1",
                  description=LICENCE_NOTICE)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def claim_owner_token(request: Request, call_next):
        """The QR link carries the owner token once; after that a cookie carries it.

        Redirecting to a clean URL is not cosmetic. A token left in the address bar ends
        up in browser history, in any screenshot of the phone, and in whatever link the
        owner pastes to someone next.

        `httponly` keeps the page's own JavaScript away from it, which costs nothing
        because the page never needs to read it - fetch sends it automatically. `secure`
        is deliberately NOT set: with it the cookie would never be stored on
        http://127.0.0.1:8000, which is the common case. Over the tunnel the hop to
        Cloudflare is TLS regardless.
        """
        token = request.query_params.get("k")
        if token and request.url.path == "/":
            resp = RedirectResponse("/", status_code=303)
            if secrets.compare_digest(token, library().owner_token()):
                resp.set_cookie(OWNER_COOKIE, token, max_age=30 * 86400,
                                httponly=True, samesite="lax")
            return resp
        return await call_next(request)

    cfg0 = Config()
    budget = Budget(OUT_DIR / "share_budget.json", cfg0.share)
    queue = AnalysisQueue(cfg0.share.max_waiting)
    app.state.budget, app.state.queue = budget, queue

    def spend(kind: str, request: Request) -> None:
        """Count a use against the daily caps, or refuse with a 429.

        Localhost never counts, so this is invisible when you run Pupchat yourself.
        """
        try:
            budget.spend(kind, client_ip(request))
        except BudgetExceeded as e:
            raise HTTPException(429, str(e),
                                headers={"Retry-After": str(e.retry_after_s)})

    async def queued(fn, *args, **kwargs):
        """Run blocking pipeline work in a thread, one caller at a time."""
        try:
            return await queue.run(fn, *args, **kwargs)
        except QueueFull as e:
            ahead = e.args[0] if e.args else queue.max_waiting
            raise HTTPException(
                503, f"{ahead} clips are already queued for measuring, and each "
                     f"takes about a minute and a half of CPU. Try again shortly.",
                headers={"Retry-After": "90"})

    if share_mode():
        gone = sweep(UPLOADS, OUT_DIR, cfg0.share.retention_hours)
        if gone:
            print(f"[share] removed {len(gone)} clip(s) past the "
                  f"{cfg0.share.retention_hours:g}h retention window")

    @app.get("/api/health")
    def health():
        cfg = Config()
        return {"ok": True, "provider": cfg.provider, "model": cfg.active.model,
                "licence": LICENCE_NOTICE,
                "attribution": ATTRIBUTION,
                "upload": {"min_seconds": cfg.upload.min_seconds,
                           "max_seconds": cfg.upload.max_seconds,
                           "max_bytes": cfg.upload.max_bytes},
                "share": {
                    "enabled": share_mode(),
                    "retention_hours": cfg.share.retention_hours,
                    "asks_per_day": cfg.share.asks_per_ip_per_day,
                    "videos_per_day": cfg.share.analyses_per_ip_per_day,
                    "queue_depth": queue.depth,
                },
                "key_configured": bool(os_getenv(cfg.active.key_env))}

    @app.post("/api/analyse")
    async def api_analyse(request: Request, file: UploadFile = File(...)):
        """Upload a clip, measure it, get the evidence. About 93 s of CPU the first
        time, instant afterwards - the result is cached by the file's own hash.

        The daily cap is charged only for work actually done. Re-sending a clip that
        is already measured is free, so a visitor who reloads the page does not spend
        their allowance on a cache hit.
        """
        cfg = Config()
        try:
            data = await read_capped(file, cfg.upload.max_bytes)
        except ValueError as e:
            raise HTTPException(413, str(e))
        path = _store_upload(file.filename or "clip.mp4", data)
        try:
            duration = check_duration(path, cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not (cache_dir(video_hash(path)) / "evidence.json").exists():
            spend("analyse", request)
        res = await queued(analyse, path, cfg, progress=lambda *a, **k: None)
        return json_safe({
            "hash": res["hash"],
            "duration_s": round(duration, 2),
            "cached": res["cached"],
            "meta": res["meta"],
            "evidence_text": res["evidence_text"],
            "evidence": res["evidence"],
        })

    @app.post("/api/ask")
    async def api_ask(request: Request, question: str = Form(...),
                      hash: str = Form(None), video: str = Form(None),
                      history: str = Form(None), profile: str = Form(None)):
        """Ask about an already-analysed clip. `history` is the JSON list returned by
        the previous call, so follow-ups cost no re-measurement.

        Queued like /api/analyse, because ask_pupchat() calls analyse() itself - on an
        unmeasured clip this endpoint pays the full CPU cost too.
        """
        from .client import ask_pupchat
        try:
            path = _resolve(video, hash)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        turns = json.loads(history) if history else None
        spend("ask", request)
        try:
            res = await queued(ask_pupchat, path, question, history=turns,
                               cfg=Config(), progress=lambda *a, **k: None,
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

    @app.get("/api/video/{vhash}")
    def api_video(vhash: str):
        """Serve an uploaded clip back, so the page can show what was sent.

        The browser never holds the bytes - a few seconds of H.264 is megabytes against
        a ~5 MB localStorage quota - so playback streams from the server's upload cache
        by the same hash the chat already stores. FileResponse handles Range requests,
        which is what lets a <video> element seek.
        """
        try:
            path = _resolve(None, vhash)
        except FileNotFoundError:
            raise HTTPException(404, "no uploaded video with that id")
        suffix = path.suffix.lower().lstrip(".") or "mp4"
        mime = {"mov": "video/quicktime", "webm": "video/webm",
                "avi": "video/x-msvideo"}.get(suffix, "video/mp4")
        return FileResponse(path, media_type=mime)

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

    # ------------------------------------------------------------------ the library
    lib = library()

    def is_owner(request: Request) -> bool:
        """Whoever is running this server, plus any device holding the owner token.

        The token exists because the owner's own phone arrives through the Cloudflare
        tunnel, so it is not local and cannot be recognised any other way.
        """
        if is_local(client_ip(request)):
            return True
        token = (request.cookies.get(OWNER_COOKIE)
                 or request.headers.get("x-pupchat-owner") or "")
        return bool(token) and secrets.compare_digest(token, lib.owner_token())

    def require_owner(request: Request) -> None:
        if not is_owner(request):
            raise HTTPException(
                403, "These dogs and chats belong to whoever is running this Pupchat. "
                     "You can still upload a clip and ask about it.")

    @app.get("/api/library")
    def api_library(request: Request):
        """Everything the sidebar needs, in one call.

        A visitor gets the same shape with nothing in it, so the page renders its empty
        states instead of breaking - and never learns this machine has dogs on it.
        """
        return json_safe(lib.snapshot(owner=is_owner(request)))

    @app.post("/api/dogs")
    async def api_dog_create(request: Request):
        require_owner(request)
        return lib.add_dog(await request.json())

    @app.put("/api/dogs/{dog_id}")
    async def api_dog_update(request: Request, dog_id: str):
        require_owner(request)
        dog = lib.update_dog(dog_id, await request.json())
        if dog is None:
            raise HTTPException(404, "no dog with that id")
        return dog

    @app.delete("/api/dogs/{dog_id}")
    def api_dog_delete(request: Request, dog_id: str):
        require_owner(request)
        detached = lib.delete_dog(dog_id)
        if detached < 0:
            raise HTTPException(404, "no dog with that id")
        return {"ok": True, "chats_detached": detached}

    @app.put("/api/dogs/{dog_id}/photo")
    async def api_dog_photo_set(request: Request, dog_id: str,
                                file: UploadFile = File(...)):
        require_owner(request)
        if lib.dog(dog_id) is None:
            raise HTTPException(404, "no dog with that id")
        try:
            data = await read_capped(file, PHOTO_MAX_BYTES)
        except ValueError as e:
            raise HTTPException(413, str(e))
        # Decode the header rather than trusting the extension. A .jpg that is really
        # HTML, served back from this origin, would be stored XSS on the owner's own
        # page.
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.verify()
            fmt = (img.format or "").lower()
        except Exception:
            raise HTTPException(400, "that file is not an image Pupchat can read. "
                                     "JPEG, PNG or WebP work.")
        if fmt not in ("jpeg", "png", "webp", "gif"):
            raise HTTPException(400, f"{fmt or 'that format'} is not supported. "
                                     f"Use JPEG, PNG or WebP.")
        suffix = {"jpeg": ".jpg"}.get(fmt, f".{fmt}")
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        for old in PHOTO_DIR.glob(f"{dog_id}.*"):
            old.unlink(missing_ok=True)
        (PHOTO_DIR / f"{dog_id}{suffix}").write_bytes(data)
        return lib.set_photo(dog_id, suffix)

    @app.get("/api/dogs/{dog_id}/photo")
    def api_dog_photo_get(request: Request, dog_id: str):
        require_owner(request)
        path = lib.photo_path(dog_id)
        if path is None:
            raise HTTPException(404, "that dog has no photo")
        return FileResponse(path)

    @app.post("/api/dogs/{dog_id}/baselines")
    async def api_baseline_add(request: Request, dog_id: str):
        require_owner(request)
        body = await request.json()
        if not body.get("hash"):
            raise HTTPException(400, "a baseline needs the hash of a measured clip")
        dog = lib.add_baseline(dog_id, {"hash": body["hash"],
                                        "name": body.get("name") or "clip",
                                        "duration": body.get("duration")})
        if dog is None:
            raise HTTPException(404, "no dog with that id")
        return dog

    @app.delete("/api/dogs/{dog_id}/baselines/{vhash}")
    def api_baseline_remove(request: Request, dog_id: str, vhash: str):
        require_owner(request)
        dog = lib.remove_baseline(dog_id, vhash)
        if dog is None:
            raise HTTPException(404, "no dog with that id")
        return dog

    @app.put("/api/active-dog")
    async def api_active_dog(request: Request):
        require_owner(request)
        lib.set_active_dog((await request.json()).get("dog_id"))
        return {"ok": True}

    @app.post("/api/chats")
    async def api_chat_create(request: Request):
        require_owner(request)
        body = await request.json()
        return lib.add_chat(body.get("dog_id"), body.get("title", ""))

    @app.get("/api/chats/{chat_id}")
    def api_chat_get(request: Request, chat_id: str):
        require_owner(request)
        chat = lib.chat(chat_id)
        if chat is None:
            raise HTTPException(404, "no chat with that id")
        return json_safe(chat)

    @app.patch("/api/chats/{chat_id}")
    async def api_chat_patch(request: Request, chat_id: str):
        require_owner(request)
        chat = lib.update_chat(chat_id, **(await request.json()))
        if chat is None:
            raise HTTPException(404, "no chat with that id")
        return json_safe(chat)

    @app.delete("/api/chats/{chat_id}")
    def api_chat_delete(request: Request, chat_id: str):
        require_owner(request)
        if not lib.delete_chat(chat_id):
            raise HTTPException(404, "no chat with that id")
        return {"ok": True}

    # ------------------------------------------------------------------ jobs
    jobs = JobStore()
    app.state.jobs = jobs

    def launch(job, fn):
        """Run `fn(job)` in the background and never let it vanish.

        Two traps, both silent. asyncio.create_task returns a Task the loop only holds
        weakly, so without keeping the reference it can be collected mid-flight; and an
        unhandled exception inside a detached task disappears into the loop's exception
        handler while the page keeps polling a job that will never finish.
        """
        async def runner():
            try:
                jobs.finish(job, result=await fn(job))
            except HTTPException as e:
                jobs.finish(job, error=str(e.detail))
            except Exception as e:                    # noqa: BLE001 - must not escape
                jobs.finish(job, error=f"{type(e).__name__}: {e}")

        job.task = asyncio.create_task(runner())
        return job.snapshot()

    @app.post("/api/jobs/analyse")
    async def api_job_analyse(request: Request, file: UploadFile = File(...)):
        """Measure a clip, reporting progress while it runs."""
        cfg = Config()
        try:
            data = await read_capped(file, cfg.upload.max_bytes)
        except ValueError as e:
            raise HTTPException(413, str(e))
        path = _store_upload(file.filename or "clip.mp4", data)
        try:
            duration = check_duration(path, cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))

        vhash = video_hash(path)
        cached = (cache_dir(vhash) / "evidence.json").exists()
        # The caps and the queue are checked HERE, not inside the task. A 429 that
        # arrives three polls later is indistinguishable from a slow answer, and the
        # Retry-After header is only useful on the request that was refused.
        if not cached:
            spend("analyse", request)
        job = jobs.create("analyse", eta_total_s=estimate(
            frames_for(path, cfg), cached=cached, queue_ahead=queue.depth, ask=False))

        async def work(job):
            res = await queued(analyse, path, cfg, progress=job.report)
            if not res.get("cached"):
                record_timing(time.time() - job.started, job.frames or 0)
            return json_safe({"hash": res["hash"], "duration_s": round(duration, 2),
                              "cached": res["cached"], "meta": res["meta"],
                              "evidence_text": res["evidence_text"],
                              "evidence": res["evidence"], "name": file.filename})

        return launch(job, work)

    @app.post("/api/jobs/ask")
    async def api_job_ask(request: Request, question: str = Form(...),
                          hash: str = Form(None), chat_id: str = Form(None),
                          video: str = Form(None)):
        """Ask about a clip. The server owns the transcript.

        It appends the question immediately and the answer when it arrives, so one
        writer keeps them in order and a crash mid-answer leaves a visible failed turn
        rather than a silent gap.
        """
        from .client import ask_pupchat
        try:
            path = _resolve(video, hash)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

        owner = is_owner(request)
        chat = lib.chat(chat_id) if (owner and chat_id) else None
        if chat_id and owner and chat is None:
            raise HTTPException(404, "no chat with that id")

        spend("ask", request)
        dog = lib.dog(chat.get("dog_id")) if chat else lib.dog(lib.active_dog_id())
        profile = profile_text(dog) if owner else ""
        history = (chat or {}).get("history") or None

        vhash = video_hash(path)
        cached = (cache_dir(vhash) / "evidence.json").exists()
        job = jobs.create("ask", eta_total_s=estimate(
            frames_for(path, Config()), cached=cached, queue_ahead=queue.depth))

        if chat:
            lib.append_message(chat["id"], {"role": "user", "text": question,
                                            "hash": vhash})
            lib.update_chat(chat["id"], pending_job_id=job.id, hash=vhash)

        async def work(job):
            try:
                res = await queued(ask_pupchat, path, question, history=history,
                                   cfg=Config(), progress=job.report, profile=profile)
            except SystemExit as e:                  # missing key, bad provider
                if chat:
                    lib.append_message(chat["id"], {"role": "error", "text": str(e)})
                    lib.update_chat(chat["id"], pending_job_id=None)
                raise HTTPException(503, str(e))
            out = json_safe({
                "answer": res["answer"], "history": res["history"],
                "provider": res["provider"], "model": res["model"],
                "usage": res["usage"], "cost_usd": res["cost_usd"],
                "seconds": res["seconds"], "truncated": res.get("truncated", False),
                "evidence_text": res["evidence_text"], "hash": res["hash"],
                "sources": res.get("sources") or [],
            })
            if chat:
                lib.append_message(chat["id"], {
                    "role": "assistant", "text": out["answer"], "model": out["model"],
                    "seconds": out["seconds"], "cost_usd": out["cost_usd"],
                    "sources": out["sources"]})
                lib.update_chat(chat["id"], history=out["history"],
                                pending_job_id=None)
            return out

        return launch(job, work)

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            # Evicted after an hour, or the server restarted. Saying so beats a bare
            # 404, because the page may be resuming a poll after a reload.
            raise HTTPException(404, "that job has finished and been cleared. "
                                     "Reload the page to see where it got to.")
        return json_safe(job.snapshot())

    # The front end last, so /api/* always wins over a file of the same name.
    if WEBAPP.exists():
        app.mount("/", SpaFiles(directory=str(WEBAPP), html=True), name="webapp")

    return app


PORT = 8000

BUSY_PORT_HELP = """
  Port {port} is already in use, so Pupchat cannot start.

  Almost always this is an earlier Pupchat window still running. That one
  answers the browser instead of this one, which means you would be using
  OLD code: any endpoint added since it started replies "Not Found", and
  the file it would have served is sitting on disk the whole time.

  Close the other Pupchat window, or stop it from PowerShell:

      Get-NetTCPConnection -LocalPort {port} -State Listen |
          ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}
"""


def port_is_taken(port: int = PORT, host: str = "127.0.0.1") -> bool:
    """Is something already listening here?

    Worth checking because uvicorn's failure mode is quiet and actively misleading.
    It logs "error while attempting to bind" at ERROR level, runs its own shutdown
    and exits 0 - so the banner still prints, the browser still loads the page, and
    every request is answered by the other process.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main() -> None:
    import uvicorn
    if port_is_taken():
        raise SystemExit(BUSY_PORT_HELP.format(port=PORT))
    print("=" * 74)
    print(f"Pupchat API   http://127.0.0.1:{PORT}     docs: /docs")
    print(LICENCE_NOTICE)
    print("=" * 74)
    uvicorn.run(build_app(), host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
