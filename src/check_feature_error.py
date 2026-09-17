"""Experiment D: each face feature's TRUE error, measured against ground truth.

Experiment C estimated feature noise by Monte Carlo - perturbing each landmark
independently by its own sigma and watching the feature move. That overstates the noise
of every feature built as a difference between two nearby landmarks, which is most of
them. Real pose error is dominated by a COMMON-MODE component: when the face box lands
slightly off, every landmark shifts together, and a shift shared by both ends of a
difference cancels out. Independent injection destroys that cancellation, so C reported
eyelid opening and the lip measures as noise-dominated on an assumption rather than a
measurement.

No model is needed. DogFLW ships ground-truth landmarks for the validation split, so a
feature can be computed twice - once from the prediction, once from the truth - and the
difference IS the error, with every correlation already in it.

Body features have no ground truth on this project and are not covered here; they keep
C's estimate, which remains an upper bound on their noise.

    python src/check_feature_error.py --snapshot face_project/face38/snapshot-best-008.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import splits as splits_mod
import superanimal as sa
from cascade import Cascade, config_beside
from check_feature_snr import ANGULAR, NOT_FEATURES
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from features import frame_features
from keypoint_scheme import ACTIVE, ACTIVE_NAMES


def wrap(d: float) -> float:
    """Fold an angular difference into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--snr-json", default="outputs/feature_snr.json",
                    help="supplies the observed within-clip spread from video")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pcut", type=float, default=0.35)
    ap.add_argument("--out", default="outputs/feature_error.json")
    a = ap.parse_args()

    cfg = CascadeConfig(facebox=FaceBoxConfig(),
                        post=PostConfig(ear_correct=False, shape_refine=False,
                                        min_confidence=a.pcut))
    records = json.loads(Path(a.annotations).read_text())
    sp = splits_mod.load(a.splits)
    ids = set(sp.val if a.split == "val" else sp.test)
    cascade = Cascade(a.config or config_beside(a.snapshot), a.snapshot, cfg=cfg)
    SA = sa.sa_bodyparts()

    todo = [r for r in records if r["id"] in ids]
    if a.limit:
        todo = todo[:a.limit]

    diffs: dict[str, list[float]] = {}
    truth_vals: dict[str, list[float]] = {}
    n_used = 0

    for i, rec in enumerate(todo):
        path = f"data/dogflw/{rec['file']}"
        img = cv2.imread(path)
        if img is None:
            continue
        res = cascade.run_image(path, image_bgr=img)
        lm = np.asarray(rec["landmarks"], dtype=float)
        if res.face is None or res.face_box is None or not np.isfinite(lm).all():
            continue

        # Ground truth in the same channel order the model predicts, confidence 1 so
        # nothing is gated out of the reference.
        gt = np.concatenate([lm[ACTIVE], np.ones((len(ACTIVE), 1))], axis=1)
        S = res.face_box.side

        # body=None: body features come back NaN and are skipped. Only the face is
        # under test here.
        pred_f = frame_features(None, res.face, S, SA, ACTIVE_NAMES, a.pcut)
        true_f = frame_features(None, gt, S, SA, ACTIVE_NAMES, a.pcut)

        for k in pred_f:
            p, t = pred_f[k], true_f[k]
            if not (math.isfinite(p) and math.isfinite(t)):
                continue
            diffs.setdefault(k, []).append(wrap(p - t) if k in ANGULAR else p - t)
            truth_vals.setdefault(k, []).append(t)
        n_used += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)

    observed = {}
    if Path(a.snr_json).exists():
        observed = {k: v.get("observed_sd")
                    for k, v in json.loads(Path(a.snr_json).read_text())["features"].items()}

    rows = {}
    for k, vals in diffs.items():
        if k in NOT_FEATURES or len(vals) < 30:
            continue
        d = np.asarray(vals, dtype=float)
        err_sd = float(d.std())
        bias = float(d.mean())
        # Spread of the TRUE feature across the 574 dogs: how much the quantity really
        # varies between animals and poses, independent of the video clips.
        t = np.asarray(truth_vals[k], dtype=float)
        true_sd = float(t.std())
        obs = observed.get(k)
        snr_video = (max(0.0, obs ** 2 - err_sd ** 2) / err_sd ** 2
                     if obs and err_sd > 1e-12 and math.isfinite(obs) else float("nan"))
        snr_pop = (max(0.0, true_sd ** 2 - err_sd ** 2) / err_sd ** 2
                   if err_sd > 1e-12 else float("nan"))
        rows[k] = {"n": len(d), "error_sd": err_sd, "bias": bias,
                   "true_sd": true_sd, "observed_sd_video": obs,
                   "snr_within_clip": snr_video, "snr_across_dogs": snr_pop}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"split": a.split, "n_images": n_used,
                                       "features": rows}, indent=2))

    print(f"\n{'='*94}\nFACE FEATURE ERROR vs GROUND TRUTH - {a.split}, {n_used} images")
    print(f"{'='*94}")
    print("snr_across_dogs : does the feature separate DIFFERENT DOGS/POSES "
          "(true spread vs its own error)")
    print("snr_within_clip : does it resolve motion inside one clip "
          "(video spread vs its own error)\n")
    print(f"{'feature':26s} {'n':>5s} {'err sd':>9s} {'bias':>9s} {'true sd':>9s} "
          f"{'SNR dogs':>9s} {'SNR clip':>9s}")
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["snr_across_dogs"]):
        print(f"  {k:24s} {v['n']:5d} {v['error_sd']:9.4f} {v['bias']:+9.4f} "
              f"{v['true_sd']:9.4f} {v['snr_across_dogs']:9.2f} "
              f"{v['snr_within_clip']:9.2f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
