"""One Euro filter over keypoint tracks, applied before features are computed.

Casiez, Roussel & Vogel (2012). A low-pass filter whose cutoff rises with the speed of
the signal: heavy smoothing when a keypoint is nearly still, almost none when it is
moving fast. That adaptivity is the whole reason to prefer it here over the rolling
median already in the pipeline - a fixed-width median removes jitter and lag-flattens a
fast movement equally, and on this project the fast movements ARE the signal. The
oscillation gate in aggregate.py exists precisely because smoothing had already been
measured to eat wag and pant frequencies.

    dx      = (x - x_prev) * rate
    dx_hat  = lowpass(dx, alpha(d_cutoff))
    cutoff  = min_cutoff + beta * |dx_hat|
    x_hat   = lowpass(x, alpha(cutoff))

Applied per keypoint, per coordinate, along time. Confidence scores are never touched:
a smoothed position is still only as trustworthy as the detection behind it, and
rewriting the score would hide that.

Gaps matter. A keypoint below the confidence cut has no position worth filtering, so
the filter holds its state across the gap rather than smoothing toward a garbage value
and dragging the track with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _alpha(cutoff: float, rate: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    te = 1.0 / max(rate, 1e-6)
    return 1.0 / (1.0 + tau / te)


@dataclass
class OneEuroConfig:
    min_cutoff: float = 1.0
    """Cutoff in Hz when the keypoint is still. Lower = smoother at rest, more lag."""

    beta: float = 0.7
    """How fast the cutoff opens with speed. Higher = less lag on fast motion, less
    smoothing. This is the knob that protects a wag from being flattened."""

    d_cutoff: float = 1.0
    """Cutoff for the speed estimate itself."""

    enabled: bool = True


class _Scalar:
    """One Euro over a single scalar stream, with explicit gap handling."""

    def __init__(self, cfg: OneEuroConfig, rate: float):
        self.cfg, self.rate = cfg, rate
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def __call__(self, x: float) -> float:
        if not np.isfinite(x):
            return x                       # a gap: emit it, keep state for the resume
        if self.x_prev is None:
            self.x_prev = float(x)
            return float(x)
        dx = (x - self.x_prev) * self.rate
        a_d = _alpha(self.cfg.d_cutoff, self.rate)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.cfg.min_cutoff + self.cfg.beta * abs(dx_hat)
        a = _alpha(cutoff, self.rate)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = float(x_hat), float(dx_hat)
        return float(x_hat)


def filter_series(x: np.ndarray, rate: float,
                  cfg: OneEuroConfig | None = None) -> np.ndarray:
    """One Euro along a 1-D series. NaNs pass through untouched."""
    cfg = cfg or OneEuroConfig()
    f = _Scalar(cfg, rate)
    return np.array([f(float(v)) for v in np.asarray(x, dtype=float)])


def filter_keypoints(track: np.ndarray, rate: float,
                     cfg: OneEuroConfig | None = None,
                     pcut: float | None = None) -> np.ndarray:
    """(N, K, 3) keypoint track -> smoothed copy. Column 2 (score) is left alone.

    Positions whose score is below `pcut` are treated as gaps: the filter holds state
    rather than chasing a detection the pipeline has already decided not to trust.
    """
    cfg = cfg or OneEuroConfig()
    out = np.array(track, dtype=float, copy=True)
    if not cfg.enabled or out.size == 0:
        return out
    n, k, _ = out.shape
    for j in range(k):
        for c in (0, 1):
            series = out[:, j, c].copy()
            if pcut is not None:
                series = np.where(out[:, j, 2] >= pcut, series, np.nan)
            smoothed = filter_series(series, rate, cfg)
            keep = np.isfinite(smoothed)
            out[keep, j, c] = smoothed[keep]
    return out


# ------------------------------------------------------------------ measurement

def jitter(track: np.ndarray, pcut: float = 0.35) -> float:
    """Median per-frame residual displacement, in pixels.

    The mean displacement of all keypoints in a frame is subtracted first, so camera
    pan and the dog simply travelling across frame do not count as jitter. What is left
    is the keypoints moving against each other, which on a still dog is measurement
    noise and nothing else. Same construction as aggregate.motion_energy, so the two
    numbers mean the same thing.
    """
    t = np.asarray(track, dtype=float)
    if t.shape[0] < 2:
        return float("nan")
    out = []
    for i in range(1, t.shape[0]):
        a, b = t[i], t[i - 1]
        ok = (a[:, 2] >= pcut) & (b[:, 2] >= pcut)
        ok &= np.isfinite(a[:, :2]).all(1) & np.isfinite(b[:, :2]).all(1)
        if ok.sum() < 4:
            continue
        d = a[ok, :2] - b[ok, :2]
        out.append(float(np.mean(np.linalg.norm(d - d.mean(axis=0), axis=1))))
    return float(np.median(out)) if out else float("nan")


def retention(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """Fraction of a signal's peak-to-peak range that survives filtering.

    The guard against the filter being tuned into a flattener. 1.0 means the excursion
    came through whole; 0.5 means half of it was smoothed away.
    """
    r = np.asarray(raw, float)
    s = np.asarray(smoothed, float)
    r, s = r[np.isfinite(r)], s[np.isfinite(s)]
    if len(r) < 2 or len(s) < 2:
        return float("nan")
    span = np.ptp(r)
    return float(np.ptp(s) / span) if span > 1e-12 else float("nan")
