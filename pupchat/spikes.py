"""Frames where a feature jumps and snaps straight back.

A real movement goes somewhere and stays, at least for a moment. A tracking failure
jumps and returns, because the next frame re-detects the keypoint correctly. Separating
the two is the whole job here, and it is done by asking what the series looks like
BEFORE and AFTER the excursion:

    before median ~= after median, middle deviates   -> spike, flagged
    before median != after median                    -> a step, i.e. real movement

RANSAC upstream already rejects outliers in the FACE keypoint tracks. This is a
different and complementary check: it runs on the FEATURES, so it catches an excursion
produced by a body keypoint RANSAC never saw, or by a geometric blow-up where two
landmarks that define a ratio momentarily coincide.

Flagged frames are reported, never silently repaired. A spike is evidence that the
measurement is unreliable at that moment, and quietly smoothing it away would destroy
exactly the signal that says "do not trust this".
"""
from __future__ import annotations

import numpy as np

from ._src import ANGULAR_FEATURES


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def find_spikes(series: np.ndarray, *, scale: float, fps: float,
                window: int = 3, sigmas: float = 4.0,
                max_width: int = 3, angular: bool = False) -> list[dict]:
    """Indices where the series leaves and rejoins its local baseline.

    Args:
        scale: the feature's own measured error. The threshold is `sigmas` x this, so
            a noisy feature needs a bigger jump before anything is flagged.
        window: frames either side used to establish the local baseline.
        max_width: an excursion longer than this is a movement, not a spike.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2 * window + 3 or not np.isfinite(scale) or scale <= 0:
        return []

    thr = sigmas * scale
    out: list[dict] = []
    i = window
    while i < n - window:
        before = x[max(0, i - window):i]
        before = before[np.isfinite(before)]
        if len(before) == 0:
            i += 1
            continue
        base = float(np.median(before))

        dev = _wrap(np.array([x[i] - base]))[0] if angular else x[i] - base
        if not np.isfinite(dev) or abs(dev) <= thr:
            i += 1
            continue

        # How long does it stay out?
        j = i
        while j < n - window:
            d = (_wrap(np.array([x[j] - base]))[0] if angular else x[j] - base)
            if not np.isfinite(d) or abs(d) <= thr:
                break
            j += 1
        width = j - i
        if width > max_width:
            i = j + 1                     # a sustained move: real, not a spike
            continue

        after = x[j:j + window]
        after = after[np.isfinite(after)]
        if len(after) == 0:
            i = j + 1
            continue
        after_base = float(np.median(after))
        back = (_wrap(np.array([after_base - base]))[0] if angular
                else after_base - base)

        # Snapped back to where it started? Then the excursion was not a move.
        if abs(back) <= thr:
            peak = i + int(np.nanargmax(np.abs(
                (_wrap(x[i:j] - base) if angular else x[i:j] - base))))
            out.append({
                "frame": int(peak), "t": round(peak / fps, 2),
                "width_frames": int(width),
                "deviation": round(float(
                    (_wrap(np.array([x[peak] - base]))[0] if angular
                     else x[peak] - base)), 4),
                "threshold": round(float(thr), 4),
                "baseline": round(base, 4),
            })
        i = j + 1
    return out


def scan(frames: list[dict], reliability: dict, fps: float, *,
         sigmas: float = 4.0, window: int = 3, max_width: int = 3) -> dict:
    """Every feature's spikes, keyed by feature name. Features with no measured error
    are skipped: without a scale there is no honest threshold to apply."""
    if not frames:
        return {}
    out: dict[str, list[dict]] = {}
    for name in frames[0]:
        rel = reliability.get(name)
        scale = rel.scale if rel is not None else None
        if scale is None:
            continue
        series = np.array([f.get(name, np.nan) for f in frames], dtype=float)
        found = find_spikes(series, scale=scale, fps=fps, window=window,
                            sigmas=sigmas, max_width=max_width,
                            angular=name in ANGULAR_FEATURES)
        if found:
            out[name] = found
    return out


def summarise(spikes: dict) -> dict:
    """Counts and the worst offenders, for the evidence block."""
    frames = sorted({s["frame"] for v in spikes.values() for s in v})
    per_feature = {k: len(v) for k, v in sorted(spikes.items(),
                                                key=lambda kv: -len(kv[1]))}
    return {
        "n_flagged_frames": len(frames),
        "flagged_frames": frames,
        "per_feature": per_feature,
        "detail": spikes,
    }
