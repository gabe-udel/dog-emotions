"""Render the feature values onto the video, from the cache.

`src/run_video.py --feature-overlay` already draws this, but it re-runs the whole
cascade to do it. Pupchat has already paid that cost and stored the result, so this
re-uses the cached keypoints and feature rows and only redraws. On a 7.5 s clip that is
seconds instead of minutes.

The drawing itself is `draw.Renderer` and `draw.feature_panel`, imported rather than
reimplemented: an overlay that disagreed with the research pipeline's overlay would be
worse than no overlay, because both look authoritative.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import _src
from .pipeline import cache_dir
from .trace import load_keypoints

DEFAULT_PANEL = (
    "ear_angle_L_eyeframe", "ear_angle_R_eyeframe", "ear_height", "ear_closeness_iod",
    "tail_height", "tail_lateral", "head_tilt", "head_pitch",
    "mouth_open_jaw", "motion_energy",
)


def overlay_path(vhash: str) -> Path:
    return cache_dir(vhash) / "overlay.mp4"


def render_overlay(vhash: str, *, features: list[str] | None = None,
                   fps: float | None = None, force: bool = False,
                   progress=print) -> Path:
    """Draw keypoints and a live feature readout onto the cached frames."""
    out = overlay_path(vhash)
    if out.exists() and not force:
        progress(f"[overlay] reusing {out.name}")
        return out

    d = cache_dir(vhash)
    meta = json.loads((d / "meta.json").read_text())
    rows = json.loads((d / "frames.json").read_text())
    kp = load_keypoints(d / "keypoints.npz")
    paths = sorted((d / "frames").glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no cached frames for {vhash}; re-run the analysis")

    V = _src.vision()
    from draw import Renderer, banner, feature_panel        # noqa: E402
    from features import ANGULAR_FEATURES                   # noqa: E402

    names = [n for n in (features or DEFAULT_PANEL) if n in rows[0]]
    ranges: dict[str, tuple[float, float]] = {}
    for n in names:
        v = np.array([r.get(n, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            ranges[n] = (float(np.percentile(v, 2)), float(np.percentile(v, 98)))

    rend = Renderer(kp["sa_names"])
    first = cv2.imread(str(paths[0]))
    h, w = first.shape[:2]
    rate = float(fps or meta.get("keypoint_fps") or 12.5)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))

    n = min(len(paths), len(rows), len(kp["body"]))
    for i in range(n):
        img = cv2.imread(str(paths[i]))
        body = kp["body"][i]
        face = kp["face"][i]
        rend.draw(img, body=body if np.isfinite(body).any() else None,
                  face=face if np.isfinite(face).any() else None,
                  pcut=meta.get("feature_pcut", 0.35), lines=True)
        banner(img, "Pupchat measurements",
               sub=f"frame {i + 1}/{n}  |  {rate:.1f} fps  |  values are pre-smoothing")
        feature_panel(img, {k: rows[i].get(k) for k in names}, ranges,
                      angular=ANGULAR_FEATURES,
                      title=f"live features  ({i / rate:.2f}s)")
        vw.write(img)
        if (i + 1) % 40 == 0:
            progress(f"[overlay] {i + 1}/{n}")
    vw.release()
    progress(f"[overlay] wrote {out}")
    return out


def feature_track(vhash: str, features: list[str] | None = None) -> dict:
    """Per-frame values plus timestamps, for a UI that scrubs alongside the video."""
    d = cache_dir(vhash)
    meta = json.loads((d / "meta.json").read_text())
    rows = json.loads((d / "frames.json").read_text())
    fps = float(meta.get("keypoint_fps") or 12.5)
    names = [n for n in (features or list(rows[0])) if n in rows[0]]
    return {
        "hash": vhash,
        "fps": fps,
        "n_frames": len(rows),
        "features": names,
        "t": [round(i / fps, 3) for i in range(len(rows))],
        "values": {n: [None if not np.isfinite(r.get(n, np.nan)) else
                       round(float(r[n]), 5) for r in rows] for n in names},
    }
