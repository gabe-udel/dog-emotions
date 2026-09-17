"""Turn a clip's per-frame features into one row per clip.

Three jobs, in this order, and the order is not arbitrary:

1. **Smooth**, with the window sized in milliseconds and converted using the clip's own
   fps. A fixed frame count means different amounts of smoothing on 25 and 60 fps
   footage.
2. **Differentiate** the smoothed series, per SECOND. Differentiating raw keypoints
   multiplies their error by the frame rate; differentiating per frame makes 25 and
   60 fps clips incomparable.
3. **Summarise** each series to a level, a spread, and - where the signal is an event
   rather than a pose - a percentile.

Angles are handled circularly throughout. A feature resting near +/-180 wraps, and a
plain mean or standard deviation reports a wild spread for a perfectly steady value:
measured on real video, head_tilt's apparent 145-degree spread was 1.23 degrees once
wrapping was handled.

Percentiles, never min/max: a minimum is a single frame, so it reports the clip's worst
keypoint error rather than the dog's deepest tuck.
"""
from __future__ import annotations

import math

import numpy as np

from features import ANGULAR_FEATURES

# Features whose signal is a brief event rather than a sustained pose, and the tail of
# the distribution that carries it.
EVENT_LOW = {"tail_tuck", "posture"}
EVENT_HIGH = {"mouth_open_jaw", "ear_closeness_iod"}

# Series whose rhythm matters: a wag and a pant are invisible in any single frame.
OSCILLATORY = {"tail_lateral": "wag", "mouth_open_jaw": "mouth"}

# Bookkeeping columns: reported as coverage, never summarised as if they were signals.
NOT_FEATURES = {"tail_conf", "head_tilt_used_ground", "face_px", "body_L", "n_paws"}


def load_noise_floors(error_json: str = "outputs/feature_error.json",
                      snr_json: str = "outputs/feature_snr.json") -> dict[str, float]:
    """Per-feature amplitude floors, taken from what each feature's error was measured
    to be - not guessed.

    Face features use `error_sd` from the ground-truth comparison, which is the real
    thing. Body features have no ground truth, so they fall back to the Monte-Carlo
    `noise_sd`, which overstates difference-features and is therefore a conservative
    floor: it will refuse a marginal rhythm rather than invent one.
    """
    import json
    from pathlib import Path

    floors: dict[str, float] = {}
    p = Path(snr_json)
    if p.exists():
        for k, v in json.loads(p.read_text()).get("features", {}).items():
            sd = v.get("noise_sd")
            if sd is not None and math.isfinite(sd):
                floors[k] = float(sd)
    p = Path(error_json)
    if p.exists():                       # ground truth wins where it exists
        for k, v in json.loads(p.read_text()).get("features", {}).items():
            sd = v.get("error_sd")
            if sd is not None and math.isfinite(sd):
                floors[k] = float(sd)
    return floors


def window_frames(ms: float, fps: float, minimum: int = 1) -> int:
    """Frames spanning `ms` at `fps`, forced odd so the window has a true centre."""
    if ms <= 0 or fps <= 0:
        return max(1, minimum)
    k = int(round(ms / 1000.0 * fps))
    if k % 2 == 0:
        k += 1
    return max(k, minimum, 1)


def circ_mean(deg: np.ndarray) -> float:
    r = np.exp(1j * np.radians(deg)).mean()
    return float(np.degrees(np.angle(r))) if abs(r) > 1e-12 else float("nan")


def circ_sd(deg: np.ndarray) -> float:
    R = float(np.abs(np.exp(1j * np.radians(deg)).mean()))
    if R <= 1e-12:
        return 180.0
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R)))))


def wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def smooth(x: np.ndarray, k: int, angular: bool = False) -> np.ndarray:
    """Centred rolling median, NaN-aware. Median, not mean - a mean is exactly the
    estimator a single bad keypoint drags around.

    Angular series need the circular mean instead, and not as a refinement. A linear
    median is wrong for wrapped data in a way that bites hardest at the edges of the
    series, where the window has EVEN length and `np.median` averages the two middle
    values: averaging +179 and -179.5 yields ~0. Two such frames in ninety pushed a
    steady head_tilt's circular spread from 0.8 to 17.3 degrees.

    The circular mean is a mean, so it is less robust than a median - acceptable here
    only because RANSAC rejection runs upstream of this.
    """
    if k <= 1:
        return x
    n, half = len(x), k // 2
    out = np.full(n, np.nan)
    for i in range(n):
        w = x[max(0, i - half):min(n, i + half + 1)]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = circ_mean(w) if angular else float(np.median(w))
    return out


def derivative(x: np.ndarray, fps: float, angular: bool) -> np.ndarray:
    """Central difference, per second. NaN wherever either neighbour is missing."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    a, b = x[2:], x[:-2]
    d = wrap(a - b) if angular else (a - b)
    out[1:-1] = d * fps / 2.0
    return out


def oscillation(x: np.ndarray, fps: float,
                noise_floor: float | None) -> tuple[float, float]:
    """(rate in Hz, amplitude). Rate is NaN unless the amplitude clears the floor.

    The gate FAILS CLOSED: `noise_floor=None` means no floor was supplied, and the rate
    is refused rather than reported. An earlier version defaulted the floor to 0.0,
    which disabled the gate silently - on a real 120-frame clip that produced a 9.97 Hz
    "wag" and a 13.10 Hz "mouth rhythm" (dog wags run 3-5 Hz). The mouth amplitude
    behind that number was 0.00203, an order of magnitude below that feature's own
    measured error of 0.0201. Counting the zero crossings of noise always yields a
    confident, meaningless frequency, so an absent floor must mean no answer.
    """
    v = x[np.isfinite(x)]
    if len(v) < max(5, int(fps // 2)):
        return float("nan"), float("nan")
    centred = v - np.median(v)
    amp = float(np.median(np.abs(centred)))
    if noise_floor is None or amp < noise_floor:
        return float("nan"), amp
    crossings = int((np.diff(np.sign(centred)) != 0).sum())
    return crossings / (2.0 * len(v) / fps), amp


def aggregate(rows: list[dict[str, float]], fps: float, smooth_ms: float = 100.0,
              osc_window_s: float = 1.0,
              noise_floor: dict[str, float] | None = None) -> dict[str, float]:
    """One clip's per-frame features -> one flat row."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    k_smooth = window_frames(smooth_ms, fps)
    floors = noise_floor or {}
    out: dict[str, float] = {"fps": fps, "n_frames": float(len(rows))}

    for key in keys:
        raw = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        valid = np.isfinite(raw)
        out[f"{key}__coverage"] = float(valid.mean())
        if key in NOT_FEATURES or valid.sum() < 5:
            continue

        angular = key in ANGULAR_FEATURES
        sm = smooth(raw, k_smooth, angular)
        v = sm[np.isfinite(sm)]
        if len(v) < 5:
            continue

        out[f"{key}__level"] = circ_mean(v) if angular else float(np.median(v))
        out[f"{key}__spread"] = circ_sd(v) if angular else float(
            np.median(np.abs(v - np.median(v))))
        if key in EVENT_LOW:
            out[f"{key}__p10"] = float(np.percentile(v, 10))
        if key in EVENT_HIGH:
            out[f"{key}__p90"] = float(np.percentile(v, 90))

        d = derivative(sm, fps, angular)
        dv = d[np.isfinite(d)]
        if len(dv) >= 5:
            out[f"{key}__rate"] = float(np.median(np.abs(dv)))

        if key in OSCILLATORY:
            tag = OSCILLATORY[key]
            # On the RAW series, not the smoothed one. Smoothing is a low-pass filter
            # and a wag or a pant is exactly the high-frequency content it removes.
            # Measured on synthetic sine series, amplitude lost to the 100 ms window:
            #
            #   3 Hz @30fps (k=3)   0%        <- the blind spot; testing only here
            #   5 Hz @60fps (k=7)  26.8%         would have shown no problem at all
            #   8 Hz @30fps (k=3)  45.3%,  rate collapses 7.8 -> 6.3 Hz
            #   8 Hz @60fps (k=7)  72.0%,  rate collapses 7.8 -> 4.3 Hz
            #
            # The damage grows with fps, because a fixed millisecond window is more
            # frames - so the higher the frame rate, the more of the band of interest
            # the window eats. Outlier rejection upstream is what protects this stage,
            # not smoothing.
            rate, amp = oscillation(raw, fps, floors.get(key))
            out[f"{tag}_rate_hz"] = rate
            out[f"{tag}_amp"] = amp

    return out


def motion_energy(bodies: np.ndarray, faces: np.ndarray, lengths: np.ndarray,
                  sides: np.ndarray, fps: float, pcut: float = 0.35) -> np.ndarray:
    """Per-frame motion energy, in scale-units per second.

    Body and face are kept apart rather than averaged into one number: their
    displacements live at different scales, and 38 face landmarks occupying a fraction
    of the frame would otherwise dominate a 39-keypoint body. Whichever is available is
    used, normalised by its own scale unit.
    """
    n = len(bodies)
    out = np.full(n, np.nan)
    for i in range(1, n):
        for arr, scale in ((bodies, lengths), (faces, sides)):
            a, b = arr[i], arr[i - 1]
            ok = (a[:, 2] >= pcut) & (b[:, 2] >= pcut)
            ok &= np.isfinite(a[:, :2]).all(axis=1) & np.isfinite(b[:, :2]).all(axis=1)
            s = scale[i]
            if ok.sum() < 4 or not (np.isfinite(s) and s > 0):
                continue
            d = a[ok, :2] - b[ok, :2]
            # Subtracting the mean displacement removes camera pan and the dog simply
            # travelling across frame, leaving motion of the body against itself.
            out[i] = float(np.mean(np.linalg.norm(d - d.mean(axis=0), axis=1))) / s * fps
            break
    return out
