"""Derive the landmarks a heatmap head cannot see, from the ones it can.

A heatmap head is a *local* detector: each output channel responds to image evidence
near its own point.  That works for eyes, nostrils and lip corners, and fails for
landmarks that sit on featureless fur.  Measured on the DogFLW test split, the two
head-top landmarks - plain skull between the ears - come out at NME 0.13 with PCK@5%
of 0.6%, which is chance.

But "no local texture" is not the same as "undetermined".  The top of the head is
implied by the rest of the head, so a shape model fitted on the training split predicts
those two points at **NME 0.074 from the CNN's own predictions** of the 30 reliable
landmarks - a 43% improvement, no retraining.

The same trick does NOT work on ears (0.100 -> 0.180 when applied to all 16).  An ear
can be perked or flopped independently of the rest of the face, so geometry cannot
predict it and the CNN's local evidence is genuinely better.  Only landmarks that are
globally determined *and* locally featureless belong in TARGET.

Fit:    .venv\\Scripts\\python.exe src\\shape_refine.py --fit
Apply:  run_video.py --refine
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES, REGION_OF

MODEL_PATH = Path("data/shape_model.npz")

# Landmarks the shape model predicts.  Kept deliberately small: every candidate must be
# shown to beat the CNN on the test split before it is added here.
TARGET = [14, 15]                       # head_top_right, head_top_left
# Landmarks it predicts *from*: the ones measured reliable (eye/nose/mouth/muzzle).
SOURCE = [d for d in range(46) if REGION_OF[d] not in ("ear", "head")]


def _similarity(P):
    """Centroid to origin, RMS radius to 1.  Returns normalised pts, centre, scale."""
    c = P.mean(-2, keepdims=True)
    s = np.sqrt(np.mean(np.sum((P - c) ** 2, -1), -1))[..., None, None]
    return (P - c) / np.clip(s, 1e-6, None), c, s


def fit(out_path: Path = MODEL_PATH, alpha: float = 1e-3) -> dict:
    """Fit on the DogFLW *train* split only, so the test split stays honest."""
    from sklearn.linear_model import Ridge

    recs = [r for r in json.load(open("data/dogflw/annotations.json"))
            if r["split"] == "train"]
    L = []
    for r in recs:
        lm = np.array(r["landmarks"], float)
        if np.isfinite(lm).all():
            L.append(lm)
    L = np.array(L)
    N, c, s = _similarity(L[:, SOURCE, :])
    Y = (L[:, TARGET, :] - c) / s
    m = Ridge(alpha=alpha).fit(N.reshape(len(L), -1), Y.reshape(len(L), -1))
    np.savez(out_path, coef=m.coef_, intercept=m.intercept_,
             source=np.array(SOURCE), target=np.array(TARGET), n_train=len(L))
    print(f"fitted on {len(L)} training faces -> {out_path}")
    print(f"  predicts {[DOGFLW_NAMES[t] for t in TARGET]}")
    print(f"  from {len(SOURCE)} reliable landmarks")
    return {"coef": m.coef_, "intercept": m.intercept_}


class Refiner:
    """Applies the fitted shape model to a (K,3) prediction array, in place."""

    def __init__(self, path: Path = MODEL_PATH, keypoint_map="data/keypoint_map.json"):
        z = np.load(path)
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.source, self.target = list(z["source"]), list(z["target"])
        km = json.load(open(keypoint_map))
        d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
        # channels in the model's own 76-output indexing
        self.src_ch = [d2m[d] for d in self.source]
        self.tgt_ch = [d2m[d] for d in self.target]
        self.names = [DOGFLW_NAMES[d] for d in self.target]

    def apply(self, kpts, pcut=0.1, min_src=20):
        """kpts: (K,3) x,y,score.  Overwrites the target channels.  Returns n replaced.

        Confidence for a derived point is the median confidence of the sources it was
        derived from - it is exactly as trustworthy as its inputs, and that keeps the
        display cutoff meaningful rather than stamping a fake 1.0 on it.
        """
        src = kpts[self.src_ch]
        ok = np.isfinite(src[:, :2]).all(1) & (src[:, 2] >= pcut)
        if ok.sum() < min_src:
            return 0
        P = src[:, :2].copy()
        # a source that failed the gate is replaced by the mean of those that passed,
        # so the similarity frame is not dragged off by a stray point
        P[~ok] = P[ok].mean(0)
        N, c, s = _similarity(P[None])
        pred = (N.reshape(1, -1) @ self.coef.T + self.intercept).reshape(len(self.tgt_ch), 2)
        kpts[self.tgt_ch, :2] = pred * s[0] + c[0]
        kpts[self.tgt_ch, 2] = np.median(src[ok, 2])
        return len(self.tgt_ch)


def evaluate(path: Path = MODEL_PATH):
    """Score the fitted model on the held-out test split, against the CNN it replaces."""
    from run_video import build_pose_runner

    km = json.load(open("data/keypoint_map.json"))
    d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
    recs = [r for r in json.load(open("data/dogflw/annotations.json"))
            if r["split"] == "test"]
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    order = {i: k for k, i in enumerate(z["ids"])}

    items, L, diag = [], [], []
    for r in recs:
        lm = np.array(r["landmarks"], float)
        k = order[r["id"]]
        if not np.isfinite(lm).all() or z["srcs"][k] <= 0:
            continue
        x1, y1, x2, y2 = z["boxes"][k]
        items.append((f"data/dogflw/{r['file']}",
                      {"bboxes": np.array([[x1, y1, x2 - x1, y2 - y1]])}))
        L.append(lm)
        bb = r["bbox_xyxy"]
        diag.append(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
    L, diag = np.array(L), np.array(diag)
    print(f"evaluating on {len(items)} held-out test images")

    runner = build_pose_runner("model_weights/superanimal_quadruped_dogface_sigma8.yaml",
                               "model_weights/superanimal_quadruped_dogface_sigma8.pt",
                               len(km["bodyparts"]))
    out, B = [], 32
    for i in range(0, len(items), B):
        out += runner.inference(items[i:i + B])
    P = np.stack([np.asarray(r["bodyparts"])[0] for r in out])

    ref = Refiner(path)
    before = P.copy()
    n = 0
    for i in range(len(P)):
        n += ref.apply(P[i]) > 0
    print(f"refined {n}/{len(P)} images\n")
    print(f"{'landmark':22s} {'CNN':>8s} {'refined':>9s} {'change':>9s}")
    for j, d in enumerate(ref.target):
        ch = d2m[d]
        e0 = np.mean(np.linalg.norm(before[:, ch, :2] - L[:, d], axis=1) / diag)
        e1 = np.mean(np.linalg.norm(P[:, ch, :2] - L[:, d], axis=1) / diag)
        print(f"{DOGFLW_NAMES[d]:22s} {e0:8.4f} {e1:9.4f} {(e1-e0)/e0:+8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="fit on the train split")
    ap.add_argument("--eval", action="store_true", help="score it on the test split")
    ap.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()
    if a.fit:
        fit(Path(a.out))
    if a.eval:
        evaluate(Path(a.out))
    if not (a.fit or a.eval):
        ap.error("nothing to do: pass --fit and/or --eval")
