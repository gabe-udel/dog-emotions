"""Fit the quarantined post-hoc corrections against the NEW face model.

Neither `ear_correct` nor `shape_refine` carries over automatically: both learn a
correction to a specific model's residuals, and the cascade's residuals are not the
unified model's. This regenerates them from the train split and nothing else.

    python src/postfit.py --snapshot face_project/face1/snapshot-004.pt --fit-ear --fit-shape

Then measure each on val before enabling it:

    python src/evaluate_face.py --split val --snapshot ... --ear-correct
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import ear_correct
import shape_refine
import splits as splits_mod
from cascade import Cascade
from faceconfig import CascadeConfig, PostConfig


def predict_split(cascade: Cascade, records: list[dict], ids: set[str],
                  limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the cascade over a split. Returns (pred (N,46,3), truth (N,46,2), ids)."""
    pred, truth, kept = [], [], []
    todo = [r for r in records if r["id"] in ids]
    if limit:
        todo = todo[:limit]
    for i, rec in enumerate(todo):
        lm = np.asarray(rec["landmarks"], dtype=float)
        if not np.isfinite(lm).all():
            continue
        path = f"data/dogflw/{rec['file']}"
        img = cv2.imread(path)
        if img is None:
            continue
        res = cascade.run_image(path, image_bgr=img)
        if res.face is None:
            continue
        pred.append(res.face)
        truth.append(lm)
        kept.append(rec["id"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    if not pred:
        raise RuntimeError("cascade produced no usable predictions on this split")
    return np.stack(pred), np.stack(truth), kept


def load_ear_types(path: str = "data/ear_types.csv") -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["id"]] = row["ear_type"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="face_project/face1/pytorch_config.yaml")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--fit-ear", action="store_true")
    ap.add_argument("--fit-shape", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not (a.fit_ear or a.fit_shape):
        ap.error("nothing to do: pass --fit-ear and/or --fit-shape")

    records = json.loads(Path(a.annotations).read_text())
    sp = splits_mod.load(a.splits)
    train_ids = set(sp.train)

    if a.fit_shape:
        # The shape model needs ground truth only - no inference required, so it does
        # not depend on which checkpoint exists yet.
        truth = np.stack([np.asarray(r["landmarks"], float) for r in records
                          if r["id"] in train_ids
                          and np.isfinite(np.asarray(r["landmarks"], float)).all()])
        shape_refine.fit(truth)

    if a.fit_ear:
        # The ear bias is a property of a specific model's residuals, so this one does
        # need the checkpoint.
        cfg = CascadeConfig(post=PostConfig(ear_correct=False, shape_refine=False))
        cascade = Cascade(a.config, a.snapshot, cfg=cfg)
        print(f"running the cascade over {len(train_ids)} train images", flush=True)
        pred, truth, kept = predict_split(cascade, records, train_ids, a.limit)
        etypes = load_ear_types()
        have = np.array([k in etypes for k in kept])
        if have.sum() < 100:
            raise RuntimeError(f"only {have.sum()} train images have an ear-type label")
        ear_correct.fit(pred[have], truth[have],
                        np.array([etypes[k] for k in kept if k in etypes]))


if __name__ == "__main__":
    main()
