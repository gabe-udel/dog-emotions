"""Derive the landmarks a heatmap head cannot see, from the ones it can.

QUARANTINED. This was fitted against the unified 76-channel model on whole-dog crops,
where head_top_left/right scored NME 0.13 at PCK@5% 0.6% - chance - because a heatmap
head is a *local* detector and the skull between the ears is featureless fur. On that
architecture the shape model reached 0.074, a 43% improvement with no retraining.

It may now be obsolete. Part of why those two points failed was that they sat near the
edge of a crop in which the face occupied ~36 of 64 heatmap cells; on a face-filling
crop the network has ~64 cells and may simply find them. So this defaults to OFF
(`PostConfig.shape_refine`) and must be re-measured on the new model before it is
switched on:

    python src/postfit.py --fit-shape --snapshot <face checkpoint>
    python src/evaluate_face.py --split val --snapshot <...> --shape-refine

The same trick does NOT work on ears (0.100 -> 0.180 when applied to all 16). An ear can
be perked or flopped independently of the rest of the face, so geometry cannot predict
it and the CNN's local evidence is genuinely better. Only landmarks that are globally
determined *and* locally featureless belong in TARGET - on the old architecture that was
exactly two of the 46.

Indexing note: the face model emits the 46 DogFLW landmarks directly, so channel index
== DogFLW index. The old `dogflw_to_model_idx` indirection into a 76-channel head is
gone along with the model that needed it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from keypoint_scheme import DOGFLW_NAMES, REGION_OF

MODEL_PATH = Path("data/shape_model.npz")

# Landmarks the shape model predicts. Deliberately small: a candidate must be shown to
# beat the CNN on the VALIDATION split before it is added.
TARGET = [14, 15]                       # head_top_right, head_top_left
# Landmarks it predicts *from*: eye / nose / mouth / muzzle.
SOURCE = [d for d in range(46) if REGION_OF[d] not in ("ear", "head")]


def similarity_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid to origin, RMS radius to 1. Returns (normalised, centre, scale)."""
    c = points.mean(-2, keepdims=True)
    s = np.sqrt(np.mean(np.sum((points - c) ** 2, -1), -1))[..., None, None]
    return (points - c) / np.clip(s, 1e-6, None), c, s


def fit(truth: np.ndarray, out_path: Path = MODEL_PATH, alpha: float = 1e-3) -> dict:
    """Fit the shape model on ground-truth landmarks from the TRAIN split only.

    Args:
        truth: (N, 46, 2) ground-truth landmarks. Must come from the train split -
            fitting on val or test would make every downstream number meaningless.
    """
    from sklearn.linear_model import Ridge

    truth = np.asarray(truth, dtype=float)
    good = np.isfinite(truth).all(axis=(1, 2))
    L = truth[good]
    if len(L) < 100:
        raise ValueError(f"only {len(L)} usable faces; refusing to fit a shape model")

    n, c, s = similarity_frame(L[:, SOURCE, :])
    y = (L[:, TARGET, :] - c) / s
    m = Ridge(alpha=alpha).fit(n.reshape(len(L), -1), y.reshape(len(L), -1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, coef=m.coef_, intercept=m.intercept_,
             source=np.array(SOURCE), target=np.array(TARGET), n_train=len(L))
    print(f"[shape] fitted on {len(L)} training faces -> {out_path}")
    print(f"[shape]   predicts {[DOGFLW_NAMES[t] for t in TARGET]} "
          f"from {len(SOURCE)} landmarks")
    return {"coef": m.coef_, "intercept": m.intercept_, "n_train": len(L)}


class Refiner:
    """Applies the fitted shape model to a (46, 3) prediction array, in place."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        z = np.load(path)
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.source = [int(v) for v in z["source"]]
        self.target = [int(v) for v in z["target"]]
        self.names = [DOGFLW_NAMES[d] for d in self.target]

    def apply(self, kpts: np.ndarray, pcut: float = 0.1, min_src: int = 20) -> int:
        """Overwrite the target channels. Returns how many were replaced.

        A derived point's confidence is the median confidence of the sources it came
        from - exactly as trustworthy as its inputs, which keeps the display cutoff
        meaningful instead of stamping a fabricated 1.0 on it.
        """
        src = kpts[self.source]
        ok = np.isfinite(src[:, :2]).all(1) & (src[:, 2] >= pcut)
        if ok.sum() < min_src:
            return 0
        p = src[:, :2].copy()
        # A source that failed the gate is replaced by the mean of those that passed, so
        # one stray point cannot drag the similarity frame off.
        p[~ok] = p[ok].mean(0)
        n, c, s = similarity_frame(p[None])
        pred = (n.reshape(1, -1) @ self.coef.T + self.intercept)
        kpts[self.target, :2] = pred.reshape(len(self.target), 2) * s[0] + c[0]
        kpts[self.target, 2] = np.median(src[ok, 2])
        return len(self.target)
