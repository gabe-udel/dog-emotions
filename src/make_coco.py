"""Build the COCO dataset the fine-tune trains on: 39 SuperAnimal keypoints + the added
DogFLW face keypoints, on whole-dog crops.

Supervision per keypoint (COCO visibility flag):
  2  -> supervised.  DogFLW ground truth for face keypoints; SuperAnimal pseudo-labels
        (memory replay) for body keypoints the pretrained model is confident about.
 -1  -> loss fully masked for that keypoint (DeepLabCut's HeatmapGaussianGenerator sets
        its weights to 0), used where the pretrained model is not confident, so training
        neither invents a label nor pushes the channel to background.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")

PSEUDO_THRESH = 0.4          # min SuperAnimal score for a pseudo-label to be used
OUT = Path("dlc_project")


def link_dir(link: Path, target: Path) -> str:
    """Point `link` at `target` without copying 349 MB of JPEGs.

    Plain symlinks need Developer Mode or elevation on Windows (WinError 1314), so fall
    back to a directory junction, which any user may create. Copying is the last resort.
    """
    try:
        link.symlink_to(target)
        return "symlink"
    except OSError:
        pass
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return "junction"
    import shutil
    shutil.copytree(target, link)
    return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("data/keypoint_map.json"),
                    help="keypoint map from analyze_correspondence.py")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="project dir to write annotations/ into")
    args = ap.parse_args()
    out_dir = args.out

    recs = json.load(open("data/dogflw/annotations.json"))
    km = json.load(open(args.map))
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    boxes, srcs, poses = z["boxes"], z["srcs"], z["poses"]

    sa_bps = km["superanimal_bodyparts"]
    bodyparts = km["bodyparts"]
    new_idx = km["new_dogflw_indices"]
    merge = {sa_bps.index(k): v for k, v in km["merge_sa_to_dogflw"].items()}
    n_kpt = len(bodyparts)
    print(f"{n_kpt} keypoints = 39 SuperAnimal + {len(new_idx)} added DogFLW")

    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    img_link = out_dir / "images"
    if not img_link.exists():
        how = link_dir(img_link, Path("data/dogflw/images").resolve())
        print(f"dlc_project/images -> data/dogflw/images ({how})")

    stats = {"gt": 0, "pseudo": 0, "masked": 0}
    out = {"train": {"images": [], "annotations": []}, "test": {"images": [], "annotations": []}}
    skipped = 0
    box_src = {0: 0, 1: 0, 2: 0}

    for i, r in enumerate(recs):
        # Images where the detector found nothing keep the grown face box as their crop -
        # the face landmarks are still valid supervision there, only the body pseudo-labels
        # are unreliable, and those are gated by PSEUDO_THRESH anyway.
        box_src[int(srcs[i])] += 1
        split = r["split"]
        lm = np.array(r["landmarks"], float)
        p = poses[i]
        kp = np.full((n_kpt, 3), -1.0)

        for j in range(39):
            if j in merge:                                   # DogFLW GT supervises it
                xy = lm[merge[j]]
                if np.all(np.isfinite(xy)):
                    kp[j] = [xy[0], xy[1], 2]; stats["gt"] += 1
                else:
                    stats["masked"] += 1
            elif p[j, 2] >= PSEUDO_THRESH:                    # memory-replay pseudo-label
                kp[j] = [p[j, 0], p[j, 1], 2]; stats["pseudo"] += 1
            else:
                stats["masked"] += 1

        for k, di in enumerate(new_idx):                      # added face keypoints
            xy = lm[di]
            if np.all(np.isfinite(xy)):
                kp[39 + k] = [xy[0], xy[1], 2]; stats["gt"] += 1
            else:
                stats["masked"] += 1

        if not (kp[:, 2] > 0).any():
            skipped += 1
            continue

        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(r["width"]), float(x2)), min(float(r["height"]), float(y2))
        w, h = x2 - x1, y2 - y1
        out[split]["images"].append(
            {"id": i, "file_name": r["file"].split("/", 1)[1], "width": r["width"], "height": r["height"]})
        out[split]["annotations"].append(
            {"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "individual_id": 0,
             "area": float(w * h), "bbox": [x1, y1, w, h],
             "num_keypoints": int((kp[:, 2] > 0).sum()),
             "keypoints": [float(v) for v in kp.reshape(-1)]})

    cat = [{"id": 1, "name": "dog", "supercategory": "animal",
            "keypoints": bodyparts, "skeleton": []}]
    for split in ("train", "test"):
        out[split]["categories"] = cat
        (out_dir / "annotations" / f"{split}.json").write_text(json.dumps(out[split]))
        print(f"  {split}: {len(out[split]['images'])} images")

    tot = sum(stats.values())
    print(f"skipped {skipped} images (no supervised keypoint)")
    print(f"crop box: detector+contains-face {box_src[2]}, detector-only {box_src[1]}, "
          f"grown face box {box_src[0]}")
    print(f"keypoint supervision: ground-truth {stats['gt']:,} ({100*stats['gt']/tot:.1f}%)  "
          f"pseudo-label {stats['pseudo']:,} ({100*stats['pseudo']/tot:.1f}%)  "
          f"masked {stats['masked']:,} ({100*stats['masked']/tot:.1f}%)")
    json.dump({"n_keypoints": n_kpt, "bodyparts": bodyparts, "supervision": stats,
               "box_source": {"detector_contains_face": box_src[2], "detector_only": box_src[1],
                              "grown_face_box": box_src[0]},
               "pseudo_thresh": PSEUDO_THRESH},
              open("data/coco_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
