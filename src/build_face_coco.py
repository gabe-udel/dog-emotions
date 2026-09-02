"""Build the COCO dataset the DogFLW face model trains on.

Replaces make_coco.py. Two things are structurally different from the old pipeline:

* **Boxes come from `derive_face_box`, not from DogFLW's shipped face bounding boxes.**
  This is the invariant the whole cascade rests on. DogFLW ships a ground-truth face box
  and using it here would be easier and would score better in isolation - and would also
  guarantee that the face model never sees, during training, the box distribution it
  receives in production. Boxes are derived from SuperAnimal's *predictions*, so first-
  stage error is part of the training distribution rather than a deployment surprise.

* **No memory replay.** The face model predicts 46 DogFLW landmarks and nothing else, so
  every channel has ground truth. The old unified model supervised only 60.5% of its
  channels from ground truth, with 22.9% pseudo-labels and 16.6% masked; here it is
  100% ground truth, minus the few landmarks a crop does not cover.

Landmarks are written in IMAGE coordinates. DeepLabCut's dataloader applies the crop
transform itself, so writing crop coordinates here would double-transform them.

    python src/build_face_coco.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import splits as splits_mod
from crop import to_dlc_bbox, transform_for
from facebox import anchor_indices, derive_face_box, failure_rate
from faceconfig import CropConfig, FaceBoxConfig
from keypoint_scheme import DOGFLW_NAMES

# COCO visibility. -1 is DeepLabCut's "mask this channel entirely": its
# HeatmapGaussianGenerator zeroes the loss weights, so a landmark outside the crop
# neither invents a target nor pushes the channel toward background.
VIS_LABELLED = 2
VIS_MASKED = -1


def build(records: list[dict], poses: np.ndarray, dog_boxes: np.ndarray,
          id_order: dict[str, int], sa_bodyparts: list[str], split_ids: set[str],
          box_cfg: FaceBoxConfig, crop_cfg: CropConfig) -> tuple[dict, dict]:
    """One COCO split. Returns (coco_dict, stats)."""
    anchor_idx = anchor_indices(sa_bodyparts)
    images, annotations = [], []
    boxes_seen: list = []
    stats = Counter()

    for rec in records:
        if rec["id"] not in split_ids:
            continue
        k = id_order[rec["id"]]
        lm = np.asarray(rec["landmarks"], dtype=float)

        box = derive_face_box(poses[k], anchor_idx, box_cfg, dog_boxes[k])
        boxes_seen.append(box)
        if box is None:
            stats["no_box"] += 1
            continue

        t = transform_for(box, (rec["height"], rec["width"]), crop_cfg)
        finite = np.isfinite(lm).all(axis=1)
        inside = t.inside(lm) & finite

        kp = np.full((len(DOGFLW_NAMES), 3), 0.0)
        kp[:, :2] = np.where(inside[:, None], lm, 0.0)
        kp[:, 2] = np.where(inside, VIS_LABELLED, VIS_MASKED)
        stats["labelled"] += int(inside.sum())
        stats["masked"] += int((~inside).sum())
        stats["clipped_images"] += int((~inside).any() and finite.all())

        if not inside.any():
            stats["no_landmarks"] += 1
            continue

        x, y, w, h = to_dlc_bbox(box)
        images.append({"id": k, "file_name": rec["file"].split("/", 1)[1],
                       "width": rec["width"], "height": rec["height"]})
        annotations.append({
            "id": k, "image_id": k, "category_id": 1, "iscrowd": 0, "individual_id": 0,
            "area": float(w * h), "bbox": [float(x), float(y), float(w), float(h)],
            "num_keypoints": int(inside.sum()),
            "keypoints": [float(v) for v in kp.reshape(-1)],
            "face_box_source": box.source, "face_box_anchors": box.n_anchors,
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "dog_face", "supercategory": "animal",
                        "keypoints": [DOGFLW_NAMES[i] for i in range(len(DOGFLW_NAMES))],
                        "skeleton": []}],
    }
    stats.update(failure_rate(boxes_seen))
    return coco, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--poses", default="data/sa_dogboxes.npz")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", type=Path, default=Path("face_project"))
    ap.add_argument("--pad", type=float, default=FaceBoxConfig.pad)
    ap.add_argument("--anchor-conf", type=float, default=FaceBoxConfig.anchor_conf)
    ap.add_argument("--crop", type=int, default=CropConfig.size)
    a = ap.parse_args()

    box_cfg = FaceBoxConfig(pad=a.pad, anchor_conf=a.anchor_conf)
    crop_cfg = CropConfig(size=a.crop)

    records = json.loads(Path(a.annotations).read_text())
    z = np.load(a.poses, allow_pickle=True)
    # Materialise once: NpzFile decompresses on every __getitem__, which turns a loop
    # over 4,335 records into 4,335 full reads of the array.
    poses, ids, xyxy = z["poses"], z["ids"], z["boxes"]
    dog_boxes = np.stack([xyxy[:, 0], xyxy[:, 1],
                          xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
    id_order = {i: k for k, i in enumerate(ids)}

    sa_bodyparts = _superanimal_bodyparts()
    sp = splits_mod.load(a.splits)

    (a.out / "annotations").mkdir(parents=True, exist_ok=True)
    _link_images(a.out / "images", Path("data/dogflw/images").resolve())

    summary = {"face_box": box_cfg.__dict__, "crop": crop_cfg.__dict__, "splits": {}}
    for name, ids_list in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        coco, stats = build(records, poses, dog_boxes, id_order, sa_bodyparts,
                            set(ids_list), box_cfg, crop_cfg)
        (a.out / "annotations" / f"{name}.json").write_text(json.dumps(coco))
        summary["splits"][name] = stats
        lab, msk = stats.get("labelled", 0), stats.get("masked", 0)
        pct = 100.0 * lab / max(1, lab + msk)
        print(f"{name:5s} {len(coco['images']):5d} images  "
              f"landmarks {lab:,} labelled / {msk:,} masked ({pct:.1f}% supervised)  "
              f"fallback {stats.get('fallback', 0)}  no-box {stats.get('failed', 0)}")

    Path("data/face_coco_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}/annotations/{{train,val,test}}.json")
    print("wrote data/face_coco_summary.json")


def _superanimal_bodyparts() -> list[str]:
    import superanimal as sa
    return sa.sa_bodyparts()


def _link_images(link: Path, target: Path) -> None:
    """Point `link` at the image directory without copying 349 MB of JPEGs."""
    if link.exists():
        return
    import subprocess
    import sys
    try:
        link.symlink_to(target)
        return
    except OSError:
        pass
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    import shutil
    shutil.copytree(target, link)


if __name__ == "__main__":
    main()
