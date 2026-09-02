"""Correct the systematic ear-landmark bias, without retraining the CNN.

QUARANTINED, but this is the one most likely to survive. DogFLW mixes erect-eared and
floppy-eared breeds and one output channel has to serve both, so the model averages the
two geometries and its ear error comes out *systematic and opposite by type*. On the
unified model the bias on `ear_*_tip` was 0.16 for erect ears against 0.78 for floppy,
while on `ear_*_outer_base` it was 0.99 erect against 0.26 floppy - which is why the two
cancel to a 1.11x ratio in aggregate and look like noise. A systematic error can be
subtracted; that took ear NME 0.0882 -> 0.0631.

Why it probably still applies: ear-type multimodality is a property of the label
distribution, not of the crop. A face-filling crop gives the network more pixels but the
same ambiguity - one channel, two geometries. Measured on the old architecture, ear error
moved 4% across a 3x crop-scale range and 20% by ear type.

Still defaults to OFF (`PostConfig.ear_correct`), because the *magnitude* of the bias is
a property of the model that was fitted, and this one has not been fitted yet:

    python src/postfit.py --fit-ear --snapshot <face checkpoint>
    python src/evaluate_face.py --split val --snapshot <...> --ear-correct

Two pieces, both fitted on the TRAIN split only: a logistic classifier that recovers ear
type from the model's OWN predicted ear landmarks, so nothing external is needed at
inference; and a per-type mean residual in a similarity frame defined by the reliable
landmarks, so the correction is invariant to head scale and position.

Indexing note: the face model emits the 46 DogFLW landmarks directly, so channel index
== DogFLW index, and the old 76-channel `dogflw_to_model_idx` indirection is gone.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from keypoint_scheme import REGION_OF

MODEL_PATH = Path("data/ear_bias.pkl")
EAR = [d for d in range(46) if REGION_OF[d] == "ear"]
REL = [d for d in range(46) if REGION_OF[d] not in ("ear", "head")]
TYPES = ["pointy", "half_floppy", "floppy"]
MIN_PER_TYPE = 10


def frames(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity frame from the reliable landmarks: (centre, RMS radius)."""
    c = pred[:, REL, :].mean(1, keepdims=True)
    s = np.sqrt(np.mean(np.sum((pred[:, REL, :] - c) ** 2, 2), 1))[:, None, None]
    return c, np.clip(s, 1e-6, None)


def fit(pred: np.ndarray, truth: np.ndarray, ear_types: np.ndarray,
        out_path: Path = MODEL_PATH) -> dict:
    """Fit classifier + per-type bias from TRAIN-split predictions.

    Args:
        pred: (N, 46, 2) model predictions in image coordinates.
        truth: (N, 46, 2) ground-truth landmarks.
        ear_types: (N,) strings from data/ear_types.csv.
    """
    from sklearn.linear_model import LogisticRegression

    pred = np.asarray(pred, dtype=float)[:, :, :2]
    truth = np.asarray(truth, dtype=float)[:, :, :2]
    ear_types = np.asarray(ear_types)
    c, s = frames(pred)
    res = (pred[:, EAR, :] - truth[:, EAR, :]) / s
    feat = ((pred - c) / s)[:, EAR, :].reshape(len(pred), -1)

    counts = {t: int((ear_types == t).sum()) for t in TYPES}
    clf = LogisticRegression(max_iter=3000).fit(feat, ear_types)
    bias = {t: res[ear_types == t].mean(0) for t in TYPES
            if counts[t] >= MIN_PER_TYPE}
    if not bias:
        raise ValueError(f"no ear type has >= {MIN_PER_TYPE} examples: {counts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"clf": clf, "bias": bias, "ear": EAR, "rel": REL,
                     "counts": counts, "n_train": len(pred)}, fh)
    print(f"[ear] fitted on {len(pred)} faces {counts} -> {out_path}")
    print(f"[ear]   biases for {sorted(bias)}")
    return {"counts": counts, "types": sorted(bias)}


class EarCorrector:
    """Applies the fitted correction to a (46, 3) prediction array, in place."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        self.clf, self.bias = m["clf"], m["bias"]
        self.ear = [int(v) for v in m["ear"]]
        self.rel = [int(v) for v in m["rel"]]

    def apply(self, kpts: np.ndarray, pcut: float = 0.1) -> str | None:
        """Returns the predicted ear type, or None if the frame could not be used."""
        rel = kpts[self.rel, :2]
        ear = kpts[self.ear, :2]
        if not (np.isfinite(rel).all() and np.isfinite(ear).all()):
            return None
        c = rel.mean(0)
        s = float(np.sqrt(np.mean(np.sum((rel - c) ** 2, 1))))
        if s < 1e-6:
            return None
        feat = ((ear - c) / s).reshape(1, -1)
        t = str(self.clf.predict(feat)[0])
        if t not in self.bias:
            return None
        kpts[self.ear, :2] = ear - self.bias[t] * s
        return t
