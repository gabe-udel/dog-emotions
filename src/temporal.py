"""Temporal post-processing of per-frame keypoints: fps-scaled smoothing and RANSAC.

Two separate jobs that are easy to conflate:

* **Smoothing** removes small jitter. It assumes every sample is roughly right and
  averages them. A single badly wrong point still drags the result.
* **RANSAC** removes samples that are wrong, by asking whether a point is consistent
  with the short-term motion the *other* frames in its window agree on. It does not
  reduce jitter at all.

They compose in that order - reject, then smooth - because smoothing first spreads an
outlier's error across its neighbours and makes it harder to isolate.

Why the window is expressed in milliseconds
-------------------------------------------
A fixed frame count is not a fixed amount of smoothing. `k=3` spans 100 ms at 30 fps
and 50 ms at 60 fps, so the same flag means different things on different clips - and
the emotion dataset in `videos/` is deliberately mixed (24/25/30/60 fps). Sizing the
window in time and converting with the clip's own fps makes runs comparable.

Nothing here reads a file or touches global state.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

Seq = list[np.ndarray | None]


# --------------------------------------------------------------------------------------
# Window sizing
# --------------------------------------------------------------------------------------

def window_frames(ms: float, fps: float, minimum: int = 1) -> int:
    """Frames spanning `ms` milliseconds at `fps`, forced odd.

    Odd because every window here is centred on the frame being judged; an even window
    has no centre and would bias the result half a frame into the future.
    """
    if ms <= 0 or fps <= 0:
        return max(1, minimum)
    k = int(round(ms / 1000.0 * fps))
    if k % 2 == 0:
        k += 1
    return max(k, minimum, 1)


# --------------------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------------------

def rolling_median(seq: Seq, k: int) -> Seq:
    """Centred k-frame median over each channel. k <= 1 returns the input unchanged.

    Median rather than mean on purpose: the mean is the estimator RANSAC exists to
    protect you from, so using it here would reintroduce the problem one stage later.
    """
    if k <= 1:
        return seq
    out: Seq = []
    half = k // 2
    for i, cur in enumerate(seq):
        if cur is None:
            out.append(None)
            continue
        window = [s for s in seq[max(0, i - half):i + half + 1]
                  if s is not None and s.shape == cur.shape]
        out.append(np.median(np.stack(window), axis=0) if len(window) > 1 else cur)
    return out


# --------------------------------------------------------------------------------------
# RANSAC outlier rejection
# --------------------------------------------------------------------------------------

def _stack(seq: Seq) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(T, K, 3) array plus a (T, K) validity mask. Missing frames become all-invalid."""
    shape = next((s.shape for s in seq if s is not None), None)
    if shape is None:
        raise ValueError("sequence contains no predictions")
    t, (k, _) = len(seq), shape
    arr = np.full((t, k, 3), np.nan)
    for i, s in enumerate(seq):
        if s is not None and s.shape == shape:
            arr[i] = s
    valid = np.isfinite(arr[:, :, :2]).all(axis=2)
    return arr, valid, (t, k)


def ransac_filter(seq: Seq, *, window: int = 7, thresh: float = 0.05,
                  degree: int = 1, min_inliers: int | None = None,
                  conf_min: float = 0.0, scale: np.ndarray | None = None,
                  repair: str = "model") -> tuple[Seq, np.ndarray]:
    """Reject keypoints inconsistent with the short-term motion around them.

    Each landmark's trajectory is treated independently. For every frame, the window
    centred on it is fitted with a low-order polynomial in time by exhaustive-minimal-
    subset RANSAC: every subset of `degree + 1` frames defines one candidate model, the
    model with the most inliers wins, and the centre frame is an outlier if it does not
    sit in that consensus.

    The subsets are enumerated rather than sampled. For the window sizes that make sense
    here (5-9 frames, degree 1-2) that is 10-84 candidates - cheap, and it removes the
    seed from the result, so two runs of the same clip cannot disagree.

    Args:
        seq: per-frame (K, 3) arrays of x, y, score, or None where nothing was predicted.
        window: frames per fit. Use `window_frames` to size it from fps.
        thresh: inlier tolerance. Multiplied by `scale[i]` when `scale` is given, so it
            reads as a fraction of the face box; otherwise it is raw pixels. A fraction
            is what you want - a fixed pixel budget is a different amount of error on a
            near dog than on a far one, which is the confound the cascade removed from
            the accuracy numbers and should not reintroduce here.
        degree: 1 fits constant velocity, 2 constant acceleration. 1 is the safer
            default; 2 needs a larger window to stay determined and will happily bend
            through an outlier.
        min_inliers: consensus needed before the centre frame may be called an outlier.
            Defaults to `degree + 2`, i.e. one more than defines the model. Below this
            the window has nothing to say and the point is left alone.
        conf_min: points scoring under this do not vote and are never used to fit. They
            are still judged, so a low-confidence point that agrees with its neighbours
            survives.
        scale: optional (T,) per-frame size, normally the face box side.
        repair: 'model' replaces an outlier with the consensus model's prediction for
            that frame; 'drop' zeroes its score so downstream confidence cuts skip it.

    Returns:
        (cleaned sequence, (T, K) boolean outlier mask).
    """
    if repair not in ("model", "drop"):
        raise ValueError(f"repair must be 'model' or 'drop'; got {repair!r}")
    if window < degree + 2:
        raise ValueError(f"window {window} too small for degree {degree}")

    arr, valid, (t, k) = _stack(seq)
    votes = valid & (arr[:, :, 2] >= conf_min)
    need = degree + 2 if min_inliers is None else min_inliers
    half = window // 2
    subsets = list(combinations(range(window), degree + 1))

    tol = np.full(t, float(thresh))
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        tol = float(thresh) * np.where(np.isfinite(s) & (s > 0), s, np.nan)

    out = [None if s is None else s.copy() for s in seq]
    outlier = np.zeros((t, k), dtype=bool)

    for c in range(t):
        if not valid[c].any() or not np.isfinite(tol[c]):
            continue
        lo, hi = max(0, c - half), min(t, c + half + 1)
        idx = np.arange(lo, hi)
        ci = int(np.where(idx == c)[0][0])
        w = len(idx)
        if w < degree + 2:
            continue

        # Centre time on the frame being judged: t=0 there keeps the Vandermonde
        # well conditioned and makes the intercept the value we care about.
        tt = (idx - c).astype(float)
        pos = arr[idx][:, :, :2]                      # (w, k, 2)
        vote = votes[idx]                             # (w, k)
        vld = valid[idx]

        best_n = np.zeros(k, dtype=int)
        best_in = np.zeros((w, k), dtype=bool)
        best_rc = np.full(k, np.inf)                  # residual at the centre frame

        for sub in subsets:
            if max(sub) >= w:
                continue
            si = np.asarray(sub)
            usable = vote[si].all(axis=0)             # (k,)
            if not usable.any():
                continue
            a = np.vander(tt[si], degree + 1)
            if abs(np.linalg.det(a)) < 1e-12:
                continue
            coef = np.linalg.solve(a, pos[si].reshape(len(si), -1))
            pred = (np.vander(tt, degree + 1) @ coef).reshape(w, k, 2)
            resid = np.linalg.norm(pred - pos, axis=2)
            inl = (resid <= tol[c]) & vote & usable[None, :]
            n = inl.sum(axis=0)
            # Ties go to whichever model fits the frame under test best. Enumeration
            # order is otherwise arbitrary, and letting it decide meant a model that
            # merely TIED the honest one could evict a good point - which showed up as
            # whole clean trajectories being rejected at particular threshold values.
            # Only strictly better support may condemn the centre.
            rc = resid[ci]
            take = usable & ((n > best_n) | ((n == best_n) & (rc < best_rc)))
            best_n = np.where(take, n, best_n)
            best_rc = np.where(take, rc, best_rc)
            best_in = np.where(take[None, :], inl, best_in)

        judged = valid[c] & (best_n >= need) & ~best_in[ci]
        if not judged.any():
            continue
        outlier[c] |= judged

        if repair == "drop":
            out[c][judged, 2] = 0.0
            continue

        # Refit on the winning consensus so the replacement uses every inlier, not
        # just the minimal subset that happened to define the model.
        for j in np.flatnonzero(judged):
            m = best_in[:, j] & vld[:, j]
            if m.sum() < degree + 1:
                continue
            coef, *_ = np.linalg.lstsq(np.vander(tt[m], degree + 1), pos[m, j], rcond=None)
            out[c][j, :2] = np.vander(np.array([0.0]), degree + 1) @ coef

    return out, outlier


def summarise(outlier: np.ndarray, names: list[str] | None = None,
              top: int = 5) -> str:
    """One-line summary plus the landmarks rejected most often."""
    t, k = outlier.shape
    total = int(outlier.sum())
    if total == 0:
        return "RANSAC: no outliers"
    per_kp = outlier.sum(axis=0)
    order = np.argsort(per_kp)[::-1][:top]
    worst = ", ".join(
        f"{names[i] if names else i}:{int(per_kp[i])}" for i in order if per_kp[i]
    )
    return (f"RANSAC: {total} outliers, {total / (t * k) * 100:.1f}% of points "
            f"({int((outlier.any(axis=1)).sum())}/{t} frames touched) | worst {worst}")
