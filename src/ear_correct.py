"""Correct the systematic ear-landmark bias, without retraining the CNN.

DogFLW mixes erect-eared and floppy-eared breeds, and one output channel has to serve
both.  The model resolves that by averaging the two geometries, which makes its ear
error *systematic and opposite* by ear type - measured on the test split, the bias on
`ear_*_tip` is 0.16 for erect ears and 0.78 for floppy, while on `ear_*_outer_base` it
is 0.99 erect against 0.26 floppy.  A systematic error can simply be subtracted.

Two pieces, both fitted on the train split only:
  * a logistic classifier that recovers ear type from the model's OWN predicted ear
    landmarks (78.7% on held-out data, against a 51.8% majority baseline), so nothing
    external is needed at inference;
  * a per-ear-type mean residual, in a similarity frame defined by the 30 reliable
    landmarks, so the correction is invariant to head scale and position.

Held-out result: ear NME 0.0882 -> 0.0649, a 26% improvement, and the predicted-type
version matches the oracle-type version to within 0.0002 - misclassifications land
between adjacent types whose biases are similar.

Fit:    .venv\\Scripts\\python.exe src\\ear_correct.py --fit --eval
Apply:  run_video.py  (on by default; --no-ear-correct disables)
"""
from __future__ import annotations
import argparse, csv, json, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES, REGION_OF

MODEL_PATH = Path("data/ear_bias.pkl")
EAR = [d for d in range(46) if REGION_OF[d] == "ear"]
REL = [d for d in range(46) if REGION_OF[d] not in ("ear", "head")]
TYPES = ["pointy", "half_floppy", "floppy"]

CFG = "model_weights/pytorch_config.yaml"
SNAP = "model_weights/superanimal_quadruped_dogface_final.pt"


def _frame(pred):
    """Similarity frame from the reliable landmarks: centre, RMS radius."""
    c = pred[:, REL, :].mean(1, keepdims=True)
    s = np.sqrt(np.mean(np.sum((pred[:, REL, :] - c) ** 2, 2), 1))[:, None, None]
    return c, np.clip(s, 1e-6, None)


def _load(split):
    km = json.load(open("data/keypoint_map.json"))
    d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
    recs = [r for r in json.load(open("data/dogflw/annotations.json"))
            if r["split"] == split]
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    order = {i: k for k, i in enumerate(z["ids"])}
    etype = {}
    with open("data/ear_types.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            etype[row["id"]] = row["ear_type"]

    items, L, diag, et = [], [], [], []
    for r in recs:
        lm = np.array(r["landmarks"], float)
        k = order[r["id"]]
        if not np.isfinite(lm).all() or z["srcs"][k] <= 0 or r["id"] not in etype:
            continue
        x1, y1, x2, y2 = z["boxes"][k]
        items.append((f"data/dogflw/{r['file']}",
                      {"bboxes": np.array([[x1, y1, x2 - x1, y2 - y1]])}))
        L.append(lm)
        bb = r["bbox_xyxy"]
        diag.append(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
        et.append(etype[r["id"]])
    return items, np.array(L), np.array(diag), np.array(et), d2m, km


def _infer(items, d2m, km):
    import subpixel
    subpixel.enable()
    from run_video import build_pose_runner
    runner = build_pose_runner(CFG, SNAP, len(km["bodyparts"]))
    out, B = [], 32
    for i in range(0, len(items), B):
        out += runner.inference(items[i:i + B])
        if (i // B) % 10 == 0:
            print(f"  {min(i+B, len(items))}/{len(items)}", flush=True)
    P = np.stack([np.asarray(r["bodyparts"])[0] for r in out])
    return np.stack([P[:, d2m[d], :2] for d in range(46)], axis=1)


def fit(out_path: Path = MODEL_PATH):
    from sklearn.linear_model import LogisticRegression
    items, L, diag, et, d2m, km = _load("train")
    print(f"fitting on {len(items)} training faces "
          f"{[(t, int((et==t).sum())) for t in TYPES]}", flush=True)
    pred = _infer(items, d2m, km)
    c, s = _frame(pred)
    res = (pred[:, EAR, :] - L[:, EAR, :]) / s
    feat = ((pred - c) / s)[:, EAR, :].reshape(len(pred), -1)

    clf = LogisticRegression(max_iter=3000).fit(feat, et)
    bias = {t: res[et == t].mean(0) for t in TYPES if (et == t).sum() >= 10}
    with open(out_path, "wb") as fh:
        pickle.dump({"clf": clf, "bias": bias, "ear": EAR, "rel": REL}, fh)
    print(f"wrote {out_path}: classifier + biases for {sorted(bias)}")


class EarCorrector:
    """Applies the fitted correction to a (K,3) prediction array, in place."""

    def __init__(self, path: Path = MODEL_PATH, keypoint_map="data/keypoint_map.json"):
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        self.clf, self.bias = m["clf"], m["bias"]
        km = json.load(open(keypoint_map))
        d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
        self.ear_ch = [d2m[d] for d in m["ear"]]
        self.rel_ch = [d2m[d] for d in m["rel"]]
        self.all_ch = [d2m[d] for d in range(46)]

    def apply(self, kpts, pcut=0.1):
        """Returns the predicted ear type, or None if the frame could not be used."""
        rel = kpts[self.rel_ch, :2]
        ear = kpts[self.ear_ch, :2]
        if not (np.isfinite(rel).all() and np.isfinite(ear).all()):
            return None
        c = rel.mean(0)
        s = float(np.sqrt(np.mean(np.sum((rel - c) ** 2, 1))))
        if s < 1e-6:
            return None
        feat = ((kpts[self.all_ch, :2] - c) / s)[[self.all_ch.index(i)
                                                  for i in self.ear_ch]].reshape(1, -1)
        t = self.clf.predict(feat)[0]
        if t not in self.bias:
            return None
        kpts[self.ear_ch, :2] = ear - self.bias[t] * s
        return t


def evaluate(path: Path = MODEL_PATH):
    items, L, diag, et, d2m, km = _load("test")
    print(f"evaluating on {len(items)} held-out test faces", flush=True)
    pred = _infer(items, d2m, km)
    corr = EarCorrector(path)

    fixed = pred.copy()
    guessed = []
    km_all = [d for d in range(46)]
    for i in range(len(pred)):
        k = np.zeros((max(corr.all_ch) + 1, 3))
        for j, d in enumerate(km_all):
            k[corr.all_ch[j], :2] = pred[i, d]
        k[:, 2] = 1.0
        t = corr.apply(k)
        guessed.append(t)
        if t is not None:
            for j, d in enumerate(EAR):
                fixed[i, d] = k[corr.ear_ch[j], :2]

    def nme(a, mask=None):
        e = np.linalg.norm(a[:, EAR, :] - L[:, EAR, :], axis=2) / diag[:, None]
        return np.mean(e if mask is None else e[mask])

    ok = np.array([g is not None for g in guessed])
    acc = np.mean([g == e for g, e in zip(guessed, et) if g is not None])
    print(f"\near-type classifier accuracy on test: {acc:.1%}  "
          f"({ok.sum()}/{len(ok)} frames corrected)")
    b, a = nme(pred), nme(fixed)
    print(f"ear NME  {b:.4f} -> {a:.4f}   ({(a-b)/b:+.1%})\n")
    print(f"{'landmark':22s} {'before':>8s} {'after':>8s} {'change':>9s}")
    for d in EAR:
        x = np.mean(np.linalg.norm(pred[:, d] - L[:, d], axis=1) / diag)
        y = np.mean(np.linalg.norm(fixed[:, d] - L[:, d], axis=1) / diag)
        print(f"{DOGFLW_NAMES[d]:22s} {x:8.4f} {y:8.4f} {(y-x)/x:+8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()
    if a.fit:
        fit(Path(a.out))
    if a.eval:
        evaluate(Path(a.out))
    if not (a.fit or a.eval):
        ap.error("nothing to do: pass --fit and/or --eval")
