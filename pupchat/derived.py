"""Two measurements built on top of the existing feature rows.

Both are NEW, and neither has been validated the way the 20 live features were. There
is no `error_sd` for either, because that requires computing the feature from a
prediction and from DogFLW ground truth and differencing them - which has not been done.
They are therefore reported with `source: "no reliability measurement yet"` and must not
be read as being on the same footing as, say, `ear_height` at 49.87 across-dog SNR.

**A naming decision worth challenging.** "Freezing" is a behavioural label with a
specific meaning in dog ethology - a fear or conflict response. What can be measured
here is narrower and duller: no movement that rises above the measurement error, for
some number of seconds. A dog standing calmly and a dog frozen in fear produce the same
reading. So the measurement is named `stillness` and reported as "no movement detectable
above measurement error", and whether that constitutes freezing is left to the model and
the owner, who can see the video. Calling it freezing in the evidence would be exactly
the kind of interpretation the evidence layer exists to keep out.
"""
from __future__ import annotations

import math

import numpy as np

from ._src import ANGULAR_FEATURES

NO_RELIABILITY = "no reliability measurement yet"


# ---------------------------------------------------------------------- stillness

def stillness_intervals(frames: list[dict], reliability: dict, fps: float, *,
                        min_seconds: float = 2.0,
                        features: list[str] | None = None) -> list[dict]:
    """Runs where EVERY checked feature stays inside its own noise floor.

    "Meaningful movement" is defined per feature by its measured error, not by one
    global threshold: 2 degrees of head tilt is noise, 2 degrees of ear angle is noise
    too but 0.02 face-widths of jaw opening is not. A feature with no measured error is
    skipped rather than assigned a guess.
    """
    if not frames:
        return []
    names = features if features is not None else list(frames[0])
    usable = []
    for n in names:
        rel = reliability.get(n)
        if rel is None or rel.scale is None:
            continue
        series = np.array([f.get(n, np.nan) for f in frames], dtype=float)
        if np.isfinite(series).mean() < 0.5:
            continue
        usable.append((n, series, rel.scale, n in ANGULAR_FEATURES))
    if not usable:
        return []

    n_frames = len(frames)
    min_len = max(1, int(round(min_seconds * fps)))

    def runs(flags: np.ndarray) -> list[tuple[int, int]]:
        out, i = [], 0
        while i < len(flags):
            if flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(flags) and not flags[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                out.append((i, j))
            i = j + 1
        return out

    per_feature_moving: dict[str, np.ndarray] = {}
    for name, series, scale, angular in usable:
        d = np.diff(series, prepend=series[0])
        if angular:
            d = (d + 180.0) % 360.0 - 180.0
        step = np.abs(d)
        per_feature_moving[name] = np.where(np.isfinite(step), step > scale, False)

    moving = np.zeros(n_frames, dtype=bool)
    for flags in per_feature_moving.values():
        moving |= flags

    def entry(a: int, b: int, scope: str, feature: str | None = None) -> dict:
        e = {
            "start_s": round(a / fps, 2),
            "end_s": round((b + 1) / fps, 2),
            "duration_s": round((b - a + 1) / fps, 2),
            "scope": scope,
            "basis": "per-feature measured noise floor",
            "source": NO_RELIABILITY,
        }
        if feature:
            e["feature"] = feature
        else:
            e["features_checked"] = [u[0] for u in usable]
        return e

    out = [entry(a, b, "all features") for a, b in runs(moving)]

    # Whole-dog stillness needs EVERY checked feature to stay inside its own noise
    # floor at once, so a single jittery feature vetoes it. Per-feature runs are
    # reported too, otherwise an empty list reads as "the dog never stopped moving"
    # when it may mean "one measurement never settled".
    for name, flags in per_feature_moving.items():
        out.extend(entry(a, b, "single feature", name) for a, b in runs(flags))

    out.sort(key=lambda e: (e["start_s"], e["scope"] != "all features"))
    return out


# -------------------------------------------------------------------- weight shift

FRONT_PAWS = ("front_left_paw", "front_right_paw")
BACK_PAWS = ("back_left_paw", "back_right_paw")


def weight_shift(body: np.ndarray, sa_names: list[str], pcut: float) -> np.ndarray:
    """Per-frame fore/aft position of the hips between the paw contacts.

    0 means the hips sit over the rear paws, 1 over the front paws, so a fall over the
    clip means weight moving backwards. Normalised by the front-to-rear paw span, so it
    does not depend on the dog's size or distance from the camera.

    NaN whenever a needed keypoint is below the confidence cut - which, per the request,
    is what makes it unavailable when the paws are not visible. It needs at least one
    front paw, one back paw, and both hips.
    """
    idx = {n: i for i, n in enumerate(sa_names)}
    need_hips = ("back_left_thai", "back_right_thai")
    out = np.full(len(body), np.nan)

    for t in range(len(body)):
        f = body[t]

        def pt(name):
            i = idx.get(name)
            if i is None:
                return None
            p = f[i]
            if not np.isfinite(p[:2]).all() or p[2] < pcut:
                return None
            return p[:2].astype(float)

        front = [p for p in (pt(n) for n in FRONT_PAWS) if p is not None]
        back = [p for p in (pt(n) for n in BACK_PAWS) if p is not None]
        hips = [p for p in (pt(n) for n in need_hips) if p is not None]
        if not front or not back or len(hips) < 2:
            continue

        fm = np.mean(front, axis=0)
        bm = np.mean(back, axis=0)
        hm = np.mean(hips, axis=0)
        axis = fm - bm
        span = float(np.linalg.norm(axis))
        if span < 1e-6:
            continue
        out[t] = float(np.dot(hm - bm, axis / span) / span)
    return out


def paw_visibility(body: np.ndarray, sa_names: list[str], pcut: float) -> dict:
    """Why weight shift was or was not available, in the terms the request asked for."""
    idx = {n: i for i, n in enumerate(sa_names)}
    out = {}
    for name in FRONT_PAWS + BACK_PAWS:
        i = idx.get(name)
        if i is None:
            out[name] = 0.0
            continue
        s = body[:, i, 2]
        s = s[np.isfinite(s)]
        out[name] = float((s >= pcut).mean()) if len(s) else 0.0
    out["any_front"] = max(out[n] for n in FRONT_PAWS)
    out["any_back"] = max(out[n] for n in BACK_PAWS)
    return out


def weight_shift_entry(body: np.ndarray, sa_names: list[str], pcut: float,
                       fps: float, min_coverage: float = 0.40) -> dict:
    """The evidence entry: a value when the paws allow it, a reason when they do not."""
    vis = paw_visibility(body, sa_names, pcut)
    series = weight_shift(body, sa_names, pcut)
    coverage = float(np.isfinite(series).mean())

    entry = {
        "feature": "weight_shift",
        "label": "hip position between the paws (0 = over rear paws, 1 = over front)",
        "coverage": round(coverage, 3),
        "paw_visibility": {k: round(v, 3) for k, v in vis.items()},
        "source": NO_RELIABILITY,
    }
    if vis["any_front"] < min_coverage or vis["any_back"] < min_coverage:
        entry["available"] = False
        entry["reason"] = (
            f"the paws were not visible enough to measure it "
            f"(front paws {vis['any_front'] * 100:.0f}% of frames, "
            f"rear paws {vis['any_back'] * 100:.0f}%)")
        return entry
    if coverage < min_coverage:
        entry["available"] = False
        entry["reason"] = (f"needed keypoints cleared the confidence cut together in "
                           f"only {coverage * 100:.0f}% of frames")
        return entry

    v = series[np.isfinite(series)]
    entry["available"] = True
    entry["level"] = round(float(np.median(v)), 4)
    entry["spread"] = round(float(np.median(np.abs(v - np.median(v)))), 4)
    entry["note"] = ("unvalidated: this feature has no measured error, so its spread "
                     "cannot be separated from tracking noise")
    return entry
