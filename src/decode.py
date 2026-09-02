"""Turn a heatmap into coordinates. Explicit, importable, tested - not a runtime patch.

The old pipeline monkeypatched DeepLabCut's `HeatmapPredictor.get_pose_prediction` at
import time, which meant decoding behaviour depended on whether some other module had
been imported first. It is a pure function of the heatmap and belongs here.

Why sub-pixel fitting is the default rather than an enhancement: these models are
trained with `generate_locref: false`, so DeepLabCut's decoder has no offset map to add
and every keypoint lands on a cell centre - 4 px in a 256 crop at stride 4. Two
landmarks peaking in the same cell then decode to byte-identical coordinates. On the
old architecture that collapsed 46 face channels onto 16 distinct positions. Argmax-only
decoding is a defect, not a baseline, so `subpixel=False` exists for tests and ablation
only.

Fitting a parabola through the peak and its two neighbours per axis and taking the
vertex is the standard HRNet / DARK-pose step.
"""
from __future__ import annotations

import numpy as np


def _as_bhwj(heatmaps: np.ndarray, n_keypoints: int | None) -> np.ndarray:
    """Normalise layout to (B, H, W, J).

    DeepLabCut emits (B, H, W, J); most PyTorch code emits (B, J, H, W). Guessing from
    shape alone is ambiguous for square maps with J == H, so `n_keypoints` disambiguates
    when supplied and a mismatch raises rather than silently transposing.
    """
    a = np.asarray(heatmaps)
    if a.ndim != 4:
        raise ValueError(f"heatmaps must be 4-D; got {a.shape}")
    if n_keypoints is None:
        return a
    if a.shape[3] == n_keypoints:
        return a
    if a.shape[1] == n_keypoints:
        return np.transpose(a, (0, 2, 3, 1))
    raise ValueError(
        f"heatmap shape {a.shape} has no axis of size n_keypoints={n_keypoints}"
    )


def decode_heatmaps(heatmaps: np.ndarray, stride: float,
                    subpixel: bool = True,
                    n_keypoints: int | None = None) -> np.ndarray:
    """Decode heatmaps to crop-pixel coordinates.

    Args:
        heatmaps: (B, H, W, J) or (B, J, H, W) activations.
        stride: crop pixels per heatmap cell, i.e. crop_size / heatmap_size. For a 256
            crop and a 64x64 map this is 4.0.
        subpixel: fit a parabola around each peak. See module docstring - off is a
            defect, kept only for ablation.
        n_keypoints: disambiguates channel-axis position when H == J.

    Returns:
        (B, J, 3) array of x, y, score in CROP coordinates. Use
        `CropTransform.to_image` to lift into image coordinates.
    """
    a = _as_bhwj(heatmaps, n_keypoints).astype(np.float64)
    b, h, w, j = a.shape
    flat = a.reshape(b, h * w, j)
    peak = flat.argmax(axis=1)                       # (B, J)
    score = np.take_along_axis(flat, peak[:, None, :], axis=1)[:, 0, :]
    y = (peak // w).astype(np.float64)
    x = (peak % w).astype(np.float64)

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    if subpixel:
        dx, dy = _parabolic_offsets(a, x.astype(int), y.astype(int))

    # Cell index -> crop pixel, taking the cell's centre. Matches the convention
    # DeepLabCut's own decoder uses, so a model trained under it is read back correctly.
    out = np.empty((b, j, 3), dtype=np.float64)
    out[:, :, 0] = (x + dx) * stride + 0.5 * stride
    out[:, :, 1] = (y + dy) * stride + 0.5 * stride
    out[:, :, 2] = score
    return out


def _parabolic_offsets(a: np.ndarray, x: np.ndarray,
                       y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex of the parabola through each peak and its axial neighbours.

    Offsets are clamped to +/- half a cell: a vertex further out than that means the
    quadratic fit is not describing this peak, and trusting it would move the keypoint
    into a neighbouring cell that has its own, higher, sample.
    """
    b, h, w, j = a.shape
    bi = np.arange(b)[:, None]
    ji = np.arange(j)[None, :]

    def at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return a[bi, np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1), ji]

    centre = at(y, x)

    def offset(hp: np.ndarray, hm: np.ndarray) -> np.ndarray:
        den = 2.0 * centre - hp - hm
        # A flat or concave neighbourhood has no meaningful vertex; fall back to 0.
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return np.clip(0.5 * (hp - hm) / den, -0.5, 0.5)

    dx = offset(at(y, x + 1), at(y, x - 1))
    dy = offset(at(y + 1, x), at(y - 1, x))
    # A peak on the border is missing a neighbour on one side; keep the cell centre.
    edge = (x <= 0) | (x >= w - 1) | (y <= 0) | (y >= h - 1)
    return np.where(edge, 0.0, dx), np.where(edge, 0.0, dy)


def distinct_positions(coords: np.ndarray, decimals: int = 3) -> int:
    """How many of the decoded points are at distinct positions.

    The diagnostic that exposed argmax quantisation in the first place: with argmax-only
    decoding this collapses far below the keypoint count, because co-located peaks
    decode identically. Kept as a regression check rather than a metric.
    """
    c = np.asarray(coords)
    pts = c.reshape(-1, c.shape[-1])[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    return len(np.unique(np.round(pts, decimals), axis=0))
