"""video -> keypoints -> features -> evidence, cached by file hash.

Feature maths is NOT reimplemented here. `run_video.clip_features` is imported and
called, so the numbers Pupchat shows and the numbers the research pipeline reports come
from one function. A second implementation that drifted would be invisible: both would
produce plausible rows.

The one thing this module does own is frame selection, and it carries the trap:
the video is strided down to roughly `target_fps`, and the STRIDED rate is what goes to
aggregate(). Handing it the source rate instead would scale every per-second derivative
and the oscillation rate by the stride factor, silently.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import _src
from .config import OUT_DIR, Config
from .evidence import build_evidence, evidence_to_text
from .reliability import load_reliability

CACHE_VERSION = 3
"""Bump when anything upstream of the cached rows changes meaning, so a stale cache is
replaced rather than silently reused."""


def video_hash(path: str | Path) -> str:
    """SHA-256 of the file contents, so the same clip under two names hits one cache."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def cache_dir(vhash: str) -> Path:
    return OUT_DIR / vhash


def probe(path: str | Path) -> tuple[float, int, float]:
    """(source fps, frame count, duration seconds) without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), n, (n / fps if fps else 0.0)


def sample_frames(path: str | Path, scratch: Path, *, target_fps: float,
                  max_frames: int, width: int) -> tuple[list[str], float]:
    """Decode every Nth frame to JPEG. Returns (paths, EFFECTIVE fps).

    The effective rate is `source_fps / stride`, and it is the only rate any downstream
    temporal calculation may use.
    """
    src_fps, _, _ = probe(path)
    stride = max(1, int(round(src_fps / max(target_fps, 0.1))))
    eff_fps = src_fps / stride

    scratch.mkdir(parents=True, exist_ok=True)
    for f in scratch.glob("*.jpg"):
        f.unlink()

    cap = cv2.VideoCapture(str(path))
    paths: list[str] = []
    i = 0
    while len(paths) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if width and w != width:
                frame = cv2.resize(frame, (width, int(round(h * width / w))),
                                   interpolation=cv2.INTER_AREA)
            p = scratch / f"k{len(paths):05d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            paths.append(str(p))
        i += 1
    cap.release()
    if not paths:
        raise SystemExit(f"no frames decoded from {path}")
    return paths, eff_fps


def analyse(video: str | Path, cfg: Config | None = None, *,
            force: bool = False, progress=print) -> dict:
    """Full measurement pass for one video. Cached by file hash."""
    cfg = cfg or Config()
    video = Path(video)
    vhash = video_hash(video)
    out = cache_dir(vhash)
    meta_path = out / "meta.json"

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_version") == CACHE_VERSION:
            progress(f"[cache] reusing measurements for {video.name} ({vhash})")
            return {
                "hash": vhash, "dir": out, "meta": meta,
                "frames": json.loads((out / "frames.json").read_text()),
                "clip": json.loads((out / "clip.json").read_text()),
                "evidence": json.loads((out / "evidence.json").read_text()),
                "evidence_text": (out / "evidence.txt").read_text(encoding="utf-8"),
                "cached": True,
            }

    V = _src.vision()
    src_fps, n_src, duration = probe(video)
    scratch = out / "frames"
    progress(f"[1/4] decoding {video.name} ({duration:.1f}s at {src_fps:.1f} fps)")
    paths, eff_fps = sample_frames(
        video, scratch, target_fps=cfg.keypoint.target_fps,
        max_frames=cfg.keypoint.max_frames, width=cfg.keypoint.width)
    progress(f"      {len(paths)} frames at {eff_fps:.1f} fps "
             f"(strided from {src_fps:.1f})")

    k = cfg.keypoint
    cascade_cfg = V["CascadeConfig"](
        facebox=V["FaceBoxConfig"](),
        post=V["PostConfig"](ear_correct=False, shape_refine=False,
                             min_confidence=k.pcut),
    )
    snapshot = str(cfg.snapshot)
    progress(f"[2/4] loading the cascade ({Path(snapshot).name})")
    cascade = V["Cascade"](V["config_beside"](snapshot), snapshot, cfg=cascade_cfg)

    t0 = time.time()
    progress("[3/4] body pose (this is the slow part on CPU)")
    boxes, poses = cascade.stage1(paths)

    results = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        fb = cascade.face_box_for(poses[i], boxes[i])
        face = cascade.stage2(img, fb) if fb is not None else None
        results.append(V["FrameResult"](boxes[i], poses[i], fb, face))
        if (i + 1) % 20 == 0:
            progress(f"      face {i + 1}/{len(paths)}")

    faces = [r.face for r in results]
    bodies = [r.body for r in results]
    if k.ransac:
        side = np.array([r.face_box.side if r.face_box is not None else np.nan
                         for r in results])
        win = V["window_frames"](k.ransac_ms, eff_fps, minimum=3)
        faces, _ = V["ransac_filter"](faces, window=win, thresh=k.ransac_thresh,
                                      degree=1, conf_min=k.pcut, scale=side,
                                      repair="model")

    progress("[4/4] features and evidence")
    frames, clip = V["clip_features"](results, bodies, faces,
                                      cascade.sa_bodyparts, eff_fps, k.pcut)

    rel = load_reliability()
    ev = build_evidence(frames, clip, rel, cfg.evidence)
    text = evidence_to_text(ev, cfg.evidence)

    box = cascade.report(results)
    meta = {
        "cache_version": CACHE_VERSION,
        "video": video.name,
        "hash": vhash,
        "source_fps": round(src_fps, 3),
        "source_frames": n_src,
        "duration_s": round(duration, 2),
        "keypoint_fps": round(eff_fps, 3),
        "keypoint_frames": len(paths),
        "snapshot": Path(snapshot).name,
        "feature_pcut": k.pcut,
        "seconds_elapsed": round(time.time() - t0, 1),
        "box_report": {k2: int(v) if isinstance(v, (int, np.integer)) else v
                       for k2, v in box.items()},
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
    (out / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (out / "evidence.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    (out / "evidence.txt").write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    progress(f"      wrote {out}")

    return {"hash": vhash, "dir": out, "meta": meta, "frames": frames, "clip": clip,
            "evidence": ev, "evidence_text": text, "cached": False}
