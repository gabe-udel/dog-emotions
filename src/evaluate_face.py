"""Evaluate the face cascade against the 76-channel baseline.

Replaces evaluate.py. The forgetting / body-drift check is gone: stage 1 is the released
SuperAnimal checkpoint run unmodified, so there is nothing to drift from.

Boxes are derived by the cascade, never taken from DogFLW's ground-truth face boxes, so
the number includes first-stage error the way deployment does. An oracle-box mode exists
for diagnosis only and is labelled as such in the output - it is the ceiling, not the
result.

    python src/evaluate_face.py --split val      # while tuning
    python src/evaluate_face.py --split test     # once, at the end
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import splits as splits_mod
from cascade import Cascade
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from keypoint_scheme import DOGFLW_NAMES, REGION_OF

# The unified 76-channel model on the same 479 held-out images, shipping config.
# CLAUDE.md section 12. Every new number is reported against this row.
BASELINE = {
    "all_mean": 0.0438, "all_median": 0.0319, "pck5": 0.710, "pck10": 0.925,
    "eye": 0.0261, "nose": 0.0269, "mouth": 0.0325, "muzzle": 0.0505,
    "ear": 0.0631, "head": 0.0880,
}


def face_diagonal(rec: dict) -> float:
    """NME normaliser: the diagonal of DogFLW's annotated face box, as in the baseline."""
    b = rec["bbox_xyxy"]
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def evaluate(cascade: Cascade, records: list[dict], ids: set[str],
             limit: int | None = None) -> dict:
    preds, truths, diags, box_px, results = [], [], [], [], []
    n_missing = 0
    todo = [r for r in records if r["id"] in ids]
    if limit:
        todo = todo[:limit]

    for i, rec in enumerate(todo):
        path = f"data/dogflw/{rec['file']}"
        img = cv2.imread(path)
        if img is None:
            n_missing += 1
            continue
        res = cascade.run_image(path, image_bgr=img)
        results.append(res)
        lm = np.asarray(rec["landmarks"], dtype=float)
        if res.face is None or not np.isfinite(lm).all():
            continue
        preds.append(res.face)
        truths.append(lm)
        diags.append(face_diagonal(rec))
        box_px.append(res.face_box.side if res.face_box else np.nan)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)

    if not preds:
        raise RuntimeError("no usable predictions; check the snapshot and the split")

    P = np.stack(preds)
    L = np.stack(truths)
    d = np.asarray(diags)
    err = np.linalg.norm(P[:, :, :2] - L, axis=2) / d[:, None]      # (N, 46)

    out: dict = {
        "n_images": len(todo), "n_scored": len(preds), "n_unreadable": n_missing,
        "box": cascade.report(results),
        "all": _stats(err.ravel()),
        "regions": {}, "per_landmark": {},
        "by_box_size": _by_box_size(err, np.asarray(box_px)),
    }
    groups = defaultdict(list)
    for k in range(len(DOGFLW_NAMES)):
        groups[REGION_OF[k]].append(k)
    for region, idx in groups.items():
        out["regions"][region] = _stats(err[:, idx].ravel()) | {"n": len(idx)}
    for k in range(len(DOGFLW_NAMES)):
        out["per_landmark"][DOGFLW_NAMES[k]] = float(err[:, k].mean())
    return out


def _stats(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    return {"nme": float(e.mean()), "median": float(np.median(e)),
            "pck5": float((e < 0.05).mean()), "pck10": float((e < 0.10).mean())}


def _by_box_size(err: np.ndarray, box_px: np.ndarray) -> list[dict]:
    """NME against face-box size in input pixels - the quantity the cascade normalises.

    The unified model's error rose with face size between images (r = +0.31), because
    face size is confounded with real camera resolution. If the cascade is doing its
    job, this table should be much flatter than that: every face now arrives at the
    model at the same scale regardless of how big it was in the frame.
    """
    per_image = err.mean(axis=1)
    ok = np.isfinite(box_px) & np.isfinite(per_image)
    if ok.sum() < 10:
        return []
    e, b = per_image[ok], box_px[ok]
    edges = np.percentile(b, [0, 20, 40, 60, 80, 100])
    edges[-1] += 1e-6
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (b >= lo) & (b < hi)
        if m.sum():
            rows.append({"box_px_lo": float(lo), "box_px_hi": float(hi),
                         "n": int(m.sum()), "nme": float(e[m].mean())})
    rows.append({"correlation_nme_vs_box_px": float(np.corrcoef(b, e)[0, 1])})
    return rows


def report(res: dict, split: str) -> None:
    a = res["all"]
    print(f"\n{'=' * 74}\nFACE CASCADE - {split} split, {res['n_scored']} images scored")
    print("=" * 74)
    print(f"{'metric':16s} {'cascade':>10s} {'baseline':>10s} {'change':>10s}")
    print("-" * 74)
    for label, key, base in (("NME mean", "nme", BASELINE["all_mean"]),
                             ("NME median", "median", BASELINE["all_median"]),
                             ("PCK@5%", "pck5", BASELINE["pck5"]),
                             ("PCK@10%", "pck10", BASELINE["pck10"])):
        v = a[key]
        better = (v - base) / base
        arrow = f"{better:+.1%}"
        print(f"{label:16s} {v:10.4f} {base:10.4f} {arrow:>10s}")
    print("-" * 74)
    print(f"{'region':16s} {'n':>4s} {'cascade':>10s} {'baseline':>10s} {'change':>10s}"
          f" {'PCK@5%':>8s}")
    for region in ("eye", "nose", "mouth", "muzzle", "ear", "head"):
        r = res["regions"].get(region)
        if not r:
            continue
        base = BASELINE[region]
        print(f"{region:16s} {r['n']:4d} {r['nme']:10.4f} {base:10.4f} "
              f"{(r['nme'] - base) / base:+9.1%} {r['pck5']:8.1%}")

    b = res["box"]
    print("-" * 74)
    print(f"face box: derived {b['derived']}  fallback {b['fallback']} "
          f"({b['fallback_rate']:.1%})  failed {b['failed']} ({b['failure_rate']:.1%})"
          f"  no dog found {b.get('no_dog', 0)}")

    if res["by_box_size"]:
        print("\nNME by face-box size in input pixels:")
        for row in res["by_box_size"]:
            if "correlation_nme_vs_box_px" in row:
                print(f"  correlation NME vs box px: "
                      f"{row['correlation_nme_vs_box_px']:+.3f}  "
                      f"(unified model was +0.310)")
            else:
                print(f"  {row['box_px_lo']:7.0f} - {row['box_px_hi']:7.0f} px  "
                      f"n={row['n']:4d}  NME {row['nme']:.4f}")

    worst = sorted(res["per_landmark"].items(), key=lambda kv: -kv[1])[:6]
    print("\nworst landmarks: " + ", ".join(f"{k} {v:.4f}" for k, v in worst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--config", default="face_project/face1/pytorch_config.yaml")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pad", type=float, default=FaceBoxConfig.pad)
    ap.add_argument("--ear-correct", action="store_true",
                    help="quarantined; off unless measured to help on THIS model")
    ap.add_argument("--shape-refine", action="store_true",
                    help="quarantined; off unless measured to help on THIS model")
    ap.add_argument("--no-subpixel", action="store_true")
    a = ap.parse_args()

    cfg = CascadeConfig(
        facebox=FaceBoxConfig(pad=a.pad),
        post=PostConfig(ear_correct=a.ear_correct, shape_refine=a.shape_refine,
                        subpixel=not a.no_subpixel),
    )
    records = json.loads(Path(a.annotations).read_text())
    sp = splits_mod.load(a.splits)
    ids = set(sp.val if a.split == "val" else sp.test)
    if a.split == "test":
        print("NOTE: scoring the TEST split. Tune on val; touch this once.\n")

    cascade = Cascade(a.config, a.snapshot, cfg=cfg)
    res = evaluate(cascade, records, ids, limit=a.limit)
    res["config"] = cfg.to_dict()
    res["split"] = a.split
    report(res, a.split)

    out = a.out or f"outputs/face_eval_{a.split}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
