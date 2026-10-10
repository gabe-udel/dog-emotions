"""Extract one feature row per labelled image, for the Bremhorst/LRc pilot.

The pilot question is whether keypoint-derived geometry carries affective signal at all.
This script does only the first half - turning images into feature rows. Clustering and
classification come after, and deliberately so: the clustering stage must never see the
labels while it fits, and keeping the two steps in separate files makes that hard to get
wrong by accident.

LRc is cropped face IMAGES, not video, so:

* No temporal features. Wag rate, mouth rhythm, motion energy and every rate-of-change
  column need consecutive frames and are simply absent here.
* No usable body features. There are no paws for a ground line and no spine for `L`.
  They are still computed and written, because their COVERAGE column is the evidence for
  that claim rather than an assumption - expect near-zero and check.

What survives is the face subset, which is the half measured against ground truth.

Labels come from the immediate parent directory by default (`positive/`, `negative/`),
or from an explicit manifest. Nothing is inferred from filenames unless you ask for it
with --dog-id-regex, because a wrong guess there silently destroys the by-dog split that
makes the eventual score meaningful.

    python src/pilot_extract.py --images data/lrc --out outputs/pilot_features.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

import superanimal as sa
from cascade import Cascade, config_beside
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from features import feature_names, frame_features
from keypoint_scheme import ACTIVE_NAMES

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover(root: Path) -> list[tuple[Path, str]]:
    """(path, label) for every image, label taken from its immediate parent folder."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXT and p.parent != root:
            out.append((p, p.parent.name))
    if not out:                      # flat directory: no labels available
        out = [(p, "") for p in sorted(root.iterdir())
               if p.suffix.lower() in IMAGE_EXT]
    return out


def from_manifest(path: Path) -> list[tuple[Path, str]]:
    raw = json.loads(path.read_text()) if path.suffix == ".json" else None
    if raw is None:
        with path.open() as fh:
            raw = list(csv.DictReader(fh))
    return [(Path(r["path"]), str(r.get("label", ""))) for r in raw]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, help="root directory; label = parent folder")
    ap.add_argument("--manifest", type=Path, help="csv/json with path,label[,dog]")
    ap.add_argument("--snapshot", default="face_project/face38/snapshot-best-008.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dog-id-regex", default=None,
                    help=r"capture group 1 of this pattern, matched against the file "
                         r"stem, becomes the dog id - e.g. '^(dog\d+)_'. Without it the "
                         r"dog column is empty and a by-dog split is IMPOSSIBLE.")
    ap.add_argument("--pcut", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/pilot_features.csv"))
    a = ap.parse_args()

    if not a.images and not a.manifest:
        raise SystemExit("give --images or --manifest")
    items = from_manifest(a.manifest) if a.manifest else discover(a.images)
    if a.limit:
        items = items[:a.limit]
    if not items:
        raise SystemExit("no images found")

    rx = re.compile(a.dog_id_regex) if a.dog_id_regex else None
    SA = sa.sa_bodyparts()
    cols = feature_names(SA, ACTIVE_NAMES)
    cfg = CascadeConfig(facebox=FaceBoxConfig(),
                        post=PostConfig(ear_correct=False, shape_refine=False,
                                        min_confidence=a.pcut))
    cascade = Cascade(a.config or config_beside(a.snapshot), a.snapshot, cfg=cfg)

    labels: dict[str, int] = {}
    rows, no_face = [], 0
    for i, (path, label) in enumerate(items):
        img = cv2.imread(str(path))
        if img is None:
            continue
        res = cascade.run_image(str(path), image_bgr=img)
        if res.face is None:
            no_face += 1
            continue
        side = res.face_box.side if res.face_box else None
        feats = frame_features(res.body, res.face, side, SA, ACTIVE_NAMES, a.pcut)
        m = rx.search(path.stem) if rx else None
        rows.append({"id": path.stem, "label": label,
                     "dog": (m.group(1) if m else ""), **feats})
        labels[label] = labels.get(label, 0) + 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    if not rows:
        raise SystemExit("no usable images - every one failed face-box derivation")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "dog"] + cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*70}\nPILOT FEATURE EXTRACTION\n{'='*70}")
    print(f"images given {len(items)}   rows written {len(rows)}   "
          f"no face box {no_face}")
    print(f"labels: {labels}")
    # Built before the f-string: a replacement field cannot span lines with implicit
    # concatenation until Python 3.12, and this project is pinned to 3.11.
    dogs = {r["dog"] for r in rows if r["dog"]}
    dog_note = str(len(dogs)) if dogs else (
        "NONE - a by-dog split is impossible, and a frame-level split will inflate "
        "any score, because frames from one video are near-duplicates")
    print(f"distinct dog ids: {dog_note}")
    print(f"\n{'feature':26s} {'coverage':>9s}")
    for c in cols:
        vals = np.array([r[c] for r in rows], dtype=float)
        cov = float(np.isfinite(vals).mean())
        flag = "" if cov > 0.5 else ("   <- unusable here" if cov < 0.05 else "   <- sparse")
        print(f"  {c:24s} {cov*100:8.1f}%{flag}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
