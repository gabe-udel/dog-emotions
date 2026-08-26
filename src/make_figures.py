"""Figures documenting the keypoint scheme, the training supervision and the results."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES, REGION_OF

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
RC = {"ear": "#ffab3c", "head": "#ffd93c", "eye": "#dc5aff", "nose": "#78f078",
      "muzzle": "#e6e63c", "mouth": "#ff6e5a"}


def load(rec):
    im = cv2.imread(f"data/dogflw/{rec['file']}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def fig_scheme(recs):
    km = json.load(open("data/keypoint_map.json"))
    merged = set(km["merge_sa_to_dogflw"].values())
    L = np.array([r["landmarks"] for r in recs]); B = np.array([r["bbox_xyxy"] for r in recs])
    nx = (L[:, :, 0] - B[:, 0:1]) / (B[:, 2] - B[:, 0])[:, None]
    ny = (L[:, :, 1] - B[:, 1:2]) / (B[:, 3] - B[:, 1])[:, None]
    m = np.stack([np.nanmean(nx, 0), np.nanmean(ny, 0)], 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="white")
    ax = axes[0]
    ax.set_title("DogFLW mean face shape (n=4,335)\ncolour = region, ring = already in SuperAnimal",
                 fontsize=11)
    for i, (x, y) in enumerate(m):
        ax.scatter(x, y, s=170, c=RC[REGION_OF[i]], zorder=3,
                   edgecolors="k" if i in merged else "none", linewidths=2.0)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7.5, zorder=4, weight="bold")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(1.02, -.02); ax.set_aspect("equal"); ax.axis("off")

    r = recs[3]
    ax = axes[1]; ax.imshow(load(r)); ax.axis("off")
    ax.set_title("46 annotated landmarks on a DogFLW image", fontsize=11)
    lm = np.array(r["landmarks"])
    for i, (x, y) in enumerate(lm):
        ax.scatter(x, y, s=42, c=RC[REGION_OF[i]], edgecolors="k", linewidths=.5, zorder=3)
    x1, y1, x2, y2 = r["bbox_xyxy"]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="w", lw=1.5, ls="--"))
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=9, label=k) for k, c in RC.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=.85)
    fig.tight_layout(); fig.savefig(OUT / "01_keypoint_scheme.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "01_keypoint_scheme.png")


def fig_supervision(recs):
    """One training sample: which of the 39+N keypoints are GT, pseudo-labelled or masked."""
    km = json.load(open("data/keypoint_map.json"))
    coco = json.load(open("dlc_project/annotations/train.json"))
    bodyparts = km["bodyparts"]; n_sa = 39
    by_id = {r["id"]: r for r in recs}
    ann = next(a for a in coco["annotations"] if a["num_keypoints"] > n_sa)
    img = next(i for i in coco["images"] if i["id"] == ann["image_id"])
    stem = Path(img["file_name"]).stem
    rec = by_id[stem]
    kp = np.array(ann["keypoints"]).reshape(-1, 3)

    fig, ax = plt.subplots(figsize=(8.6, 8.6), facecolor="white")
    ax.imshow(load(rec)); ax.axis("off")
    x, y, w, h = ann["bbox"]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#4fc3f7", lw=2))
    gt = (kp[:, 2] > 0) & (np.arange(len(kp)) >= n_sa)
    ps = (kp[:, 2] > 0) & (np.arange(len(kp)) < n_sa)
    ax.scatter(kp[ps, 0], kp[ps, 1], s=70, c="#4fc3f7", edgecolors="k", lw=.6,
               label=f"SuperAnimal pseudo-label ({ps.sum()})", zorder=3)
    ax.scatter(kp[gt, 0], kp[gt, 1], s=52, c="#7CFC7C", edgecolors="k", lw=.6, marker="D",
               label=f"DogFLW ground truth ({gt.sum()})", zorder=4)
    n_mask = int((kp[:, 2] <= 0).sum())
    ax.scatter([], [], s=70, c="#888", label=f"masked, no loss ({n_mask})")
    ax.legend(loc="lower right", fontsize=10, framealpha=.9)
    ax.set_title(f"One training sample: whole-dog crop, {len(bodyparts)} keypoint channels\n"
                 "generalised memory replay = GT for the face, pretrained predictions for the body",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "02_training_supervision.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "02_training_supervision.png")


def fig_results(config, snapshot, n=6):
    from evaluate import pose_runner_from
    km = json.load(open("data/keypoint_map.json"))
    d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
    recs = [r for r in json.load(open("data/dogflw/annotations.json")) if r["split"] == "test"]
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    order = {i: k for k, i in enumerate(z["ids"])}
    sel = [r for r in recs if z["srcs"][order[r["id"]]] > 0][:n]
    items = []
    for r in sel:
        x1, y1, x2, y2 = z["boxes"][order[r["id"]]]
        items.append((f"data/dogflw/{r['file']}", {"bboxes": np.array([[x1, y1, x2 - x1, y2 - y1]])}))
    runner = pose_runner_from(config, snapshot, len(km["bodyparts"]))
    preds = [np.asarray(o["bodyparts"])[0] for o in runner.inference(items)]

    cols = 3; rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), facecolor="white")
    for ax, r, p in zip(np.array(axes).ravel(), sel, preds):
        ax.imshow(load(r)); ax.axis("off")
        gt = np.array(r["landmarks"])
        ax.scatter(gt[:, 0], gt[:, 1], s=30, facecolors="none", edgecolors="w", lw=1.1, zorder=3)
        for di in range(46):
            q = p[d2m[di]]
            ax.scatter(q[0], q[1], s=22, c=RC[REGION_OF[di]], zorder=4)
    for ax in np.array(axes).ravel()[len(sel):]:
        ax.axis("off")
    fig.suptitle("DogFLW test set - white rings: ground truth, filled: fine-tuned model", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "03_test_predictions.png", dpi=125); plt.close(fig)
    print("wrote", OUT / "03_test_predictions.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--snapshot")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    recs = json.load(open("data/dogflw/annotations.json"))
    if a.only in ("", "scheme"):
        fig_scheme(recs)
    if a.only in ("", "supervision"):
        fig_supervision(recs)
    if a.snapshot and a.only in ("", "results"):
        fig_results(a.config, a.snapshot)
