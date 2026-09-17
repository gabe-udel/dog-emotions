"""Experiment A: does the model's confidence predict its error?

Everything that gates a feature on confidence - `pcut`, the validity flags, any
confidence-weighted fit - assumes a tall heatmap peak means a correct answer. That has
never been checked on this architecture, and on the PREVIOUS one it was measured running
the wrong way: confidence rose as accuracy fell across crop scales. So it is an open
question, not a safe assumption.

The correlation must be computed WITHIN each landmark, never pooled across all 38.
Pooled, ear tips (low confidence, high error) and eyelids (high confidence, low error)
produce a strong negative correlation on their own, which only restates "ears are worse
than eyes" - already known, and useless for deciding whether confidence carries
information. The question that matters is narrower: for the ear tip specifically, are the
images where it reported low confidence the images where it was actually wrong?

    python src/check_confidence.py --snapshot face_project/face38/snapshot-best-008.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import splits as splits_mod
from cascade import Cascade, config_beside
from evaluate_face import face_diagonal
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from keypoint_scheme import ACTIVE, DOGFLW_NAMES, REGION_OF


def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. Avoids a scipy dependency for twenty lines of work."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied groups
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def collect(cascade: Cascade, records: list[dict], ids: set[str],
            limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(N, K) error and (N, K) confidence, aligned, over the split."""
    todo = [r for r in records if r["id"] in ids]
    if limit:
        todo = todo[:limit]

    errs, confs = [], []
    for i, rec in enumerate(todo):
        path = f"data/dogflw/{rec['file']}"
        img = cv2.imread(path)
        if img is None:
            continue
        res = cascade.run_image(path, image_bgr=img)
        lm = np.asarray(rec["landmarks"], dtype=float)
        if res.face is None or not np.isfinite(lm).all():
            continue
        truth = lm[ACTIVE]
        d = face_diagonal(rec)
        errs.append(np.linalg.norm(res.face[:, :2] - truth, axis=1) / d)
        confs.append(res.face[:, 2].copy())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)

    if not errs:
        raise RuntimeError("no usable predictions")
    return np.stack(errs), np.stack(confs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/confidence_check.json")
    a = ap.parse_args()

    # Corrections OFF, and not as a workaround for the stale fit on disk: ear_correct
    # and shape_refine move points AFTER decoding, so the confidence a landmark reports
    # would no longer describe the position being scored. Measuring the raw decoder is
    # the only way the question means anything.
    cfg = CascadeConfig(facebox=FaceBoxConfig(),
                        post=PostConfig(ear_correct=False, shape_refine=False))
    records = json.loads(Path(a.annotations).read_text())
    sp = splits_mod.load(a.splits)
    ids = set(sp.val if a.split == "val" else sp.test)

    cascade = Cascade(a.config or config_beside(a.snapshot), a.snapshot, cfg=cfg)
    err, conf = collect(cascade, records, ids, a.limit)
    n, k = err.shape
    names = [DOGFLW_NAMES[d] for d in ACTIVE]

    per_lm = {}
    for j in range(k):
        per_lm[names[j]] = {
            "spearman": spearman(conf[:, j], err[:, j]),
            "mean_conf": float(np.nanmean(conf[:, j])),
            "mean_err": float(np.nanmean(err[:, j])),
        }

    within = np.array([v["spearman"] for v in per_lm.values()], dtype=float)
    pooled = spearman(conf.ravel(), err.ravel())

    # How much accuracy is actually bought by the confidence cut: error of the points
    # it keeps vs the points it drops, per landmark, pooled over images.
    gates = {}
    for cut in (0.2, 0.35, 0.5, 0.7):
        keep = conf >= cut
        gates[str(cut)] = {
            "kept_frac": float(keep.mean()),
            "err_kept": float(np.nanmean(err[keep])) if keep.any() else float("nan"),
            "err_dropped": float(np.nanmean(err[~keep])) if (~keep).any() else float("nan"),
        }

    out = {
        "split": a.split, "n_images": int(n), "n_landmarks": int(k),
        "pooled_spearman": pooled,
        "within_landmark_spearman": {
            "mean": float(np.nanmean(within)),
            "median": float(np.nanmedian(within)),
            "min": float(np.nanmin(within)),
            "max": float(np.nanmax(within)),
            "n_negative": int((within < 0).sum()),
            "n_below_-0.2": int((within < -0.2).sum()),
        },
        "per_landmark": per_lm,
        "confidence_cut": gates,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"\n{'='*66}\nCONFIDENCE vs ERROR - {a.split}, {n} images, {k} landmarks")
    print(f"{'='*66}")
    print(f"pooled Spearman (CONFOUNDED, for contrast only): {pooled:+.3f}")
    w = out["within_landmark_spearman"]
    print(f"within-landmark Spearman: mean {w['mean']:+.3f}  median {w['median']:+.3f}"
          f"  range [{w['min']:+.3f}, {w['max']:+.3f}]")
    print(f"  landmarks with negative correlation: {w['n_negative']}/{k}"
          f"   stronger than -0.2: {w['n_below_-0.2']}/{k}")
    print("\n  (negative = higher confidence goes with LOWER error = confidence works)")

    print(f"\n{'landmark':26s} {'region':8s} {'rho':>7s} {'conf':>6s} {'err':>7s}")
    for nm, v in sorted(per_lm.items(), key=lambda kv: kv[1]["spearman"]):
        d = ACTIVE[names.index(nm)]
        print(f"  {nm:24s} {REGION_OF[d]:8s} {v['spearman']:+7.3f} "
              f"{v['mean_conf']:6.3f} {v['mean_err']:7.4f}")

    print(f"\n{'cut':>6s} {'kept':>8s} {'err(kept)':>11s} {'err(dropped)':>13s}")
    for cut, g in gates.items():
        print(f"  {cut:>4s} {g['kept_frac']*100:7.1f}% {g['err_kept']:11.4f} "
              f"{g['err_dropped']:13.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
