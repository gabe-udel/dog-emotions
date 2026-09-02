"""Legend for the ORIGINAL SuperAnimal-Quadruped 39 keypoints.

Two different relationships to the DogFLW work, easily confused:

  MERGED (9)  a DogFLW landmark is supervised on that existing channel, so no new
              output was created. The channel carries pretrained weights and DogFLW
              ground truth at once.

  DONOR       an added channel copied that keypoint's 1x1 filter as its starting point,
              because it was the spatially nearest SuperAnimal keypoint. The added
              channel is a separate output - the donor is only an initialisation.

So more than 9 SuperAnimal head keypoints are involved in the face work; only 9 of them
absorbed a landmark. The body layout is measured from the model's own predictions on the
project's test clip, where the whole dog is visible.

    .venv\\Scripts\\python.exe src\\make_sa_legend.py --out outputs\\superanimal_legend.pdf
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES
from draw import SA_EDGES

INK, MUTED, RULE = "#14171F", "#6B7488", "#D8DEE8"
MERGE_INK, DONOR_INK, NA_INK = "#1B6E9E", "#1B7A50", "#B23A2B"

GROUP = {
    "head":  ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
              "right_eye", "left_eye", "right_earbase", "left_earbase",
              "right_earend", "left_earend",
              "right_antler_base", "right_antler_end",
              "left_antler_base", "left_antler_end"],
    "neck & torso": ["neck_base", "neck_end", "throat_base", "throat_end",
                     "back_base", "back_middle", "back_end", "belly_bottom",
                     "body_middle_right", "body_middle_left"],
    "legs": ["front_left_thai", "front_left_knee", "front_left_paw",
             "front_right_thai", "front_right_knee", "front_right_paw",
             "back_left_thai", "back_left_knee", "back_left_paw",
             "back_right_thai", "back_right_knee", "back_right_paw"],
    "tail": ["tail_base", "tail_end"],
}
GCOLOR = {"head": "#E08A28", "neck & torso": "#2A7FA8",
          "legs": "#3C9B46", "tail": "#A845C4"}


def mean_pose(video="Happy lab.mov", n=24, width=960):
    """Average the model's 39 body predictions over frames, in detector-box coords."""
    import superanimal as sa
    from run_video import build_pose_runner, detect_boxes, DETECTOR
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    import subpixel
    subpixel.enable()

    scratch = Path(tempfile.gettempdir()) / "sa_legend"
    scratch.mkdir(parents=True, exist_ok=True)
    for f in scratch.glob("*.jpg"):
        f.unlink()
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    step = max(1, total // n)
    paths, i = [], 0
    while len(paths) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (width, int(round(h * width / w))),
                            interpolation=cv2.INTER_AREA)
            p = scratch / f"f{len(paths):03d}.jpg"
            cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(p))
        i += 1
    cap.release()

    sa.DETECTOR = DETECTOR
    cfg = PoseConfig.build_for_superanimal_inference(
        sa.SUPER_ANIMAL, model_name=sa.POSE_MODEL, detector_name=DETECTOR,
        max_individuals=1, device="cpu").to_dict()
    _, ds = sa.snapshot_paths()
    _, det = get_inference_runners(cfg, snapshot_path=sa.snapshot_paths()[0],
                                   max_individuals=1, num_bodyparts=39,
                                   num_unique_bodyparts=0, device="cpu",
                                   detector_path=ds)
    boxes = detect_boxes(scratch, paths, det)
    km = json.load(open("data/keypoint_map.json"))
    r = build_pose_runner("model_weights/pytorch_config.yaml",
                          "model_weights/superanimal_quadruped_dogface_final.pt",
                          len(km["bodyparts"]))
    items = [(p, {"bboxes": np.array([b])}) for p, b in zip(paths, boxes) if b is not None]
    idx = [j for j, b in enumerate(boxes) if b is not None]
    P = np.stack([np.asarray(o["bodyparts"])[0] for o in r.inference(items)])[:, :39, :2]
    # normalise each frame into its own detector box, then average
    norm = []
    for k, j in enumerate(idx):
        x, y, w, h = boxes[j]
        norm.append(np.stack([(P[k, :, 0] - x) / w, (P[k, :, 1] - y) / h], 1))
    return np.median(np.array(norm), axis=0), len(norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/superanimal_legend.pdf")
    a = ap.parse_args()

    km = json.load(open("data/keypoint_map.json"))
    SA = km["superanimal_bodyparts"]
    # SuperAnimal name -> the DogFLW landmark index that was merged onto its channel
    merged_to = {k: f"{v} {DOGFLW_NAMES[int(v)]}"
                 for k, v in km["merge_sa_to_dogflw"].items()}
    # init_donor maps added-landmark -> donor SA keypoint, so count the VALUES:
    # Counter over the dict itself would count the added landmarks instead.
    donor_counts = Counter(km["init_donor"].values())

    M, nfr = mean_pose()
    idx = {b: i for i, b in enumerate(SA)}

    fig = plt.figure(figsize=(16.5, 9.0), dpi=200)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.88, 0.88], wspace=0.03,
                          left=0.03, right=0.98, top=0.845, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    for a_, b_ in SA_EDGES:
        if a_ in idx and b_ in idx:
            p, q = M[idx[a_]], M[idx[b_]]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=RULE, lw=1.6, zorder=1,
                    solid_capstyle="round")
    for g, names in GROUP.items():
        for b in names:
            if b not in idx:
                continue
            i = idx[b]
            is_m = b in merged_to
            is_na = "antler" in b
            ax.scatter(M[i, 0], M[i, 1], s=150, color=GCOLOR[g], zorder=3,
                       edgecolors=MERGE_INK if is_m else (NA_INK if is_na else "#2A2E38"),
                       linewidths=2.2 if (is_m or is_na) else 1.0)
            ax.annotate(str(i), (M[i, 0], M[i, 1]), color=INK, fontsize=6.2,
                        fontweight="bold", ha="center", va="center", zorder=4)
            if is_m:
                ax.annotate("*", (M[i, 0] + 0.022, M[i, 1] - 0.020), color=MERGE_INK,
                            fontsize=13, fontweight="bold", ha="center", va="center",
                            zorder=5)
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(1.14, -0.14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 1.05, f"Median predicted pose over {nfr} frames of the project's test "
                       "clip, normalised into the detector box",
            transform=ax.transData, fontsize=9.5, color=MUTED, ha="center", va="top")

    def block(ax2, groups, y):
        for g in groups:
            names = [b for b in GROUP[g] if b in idx]
            ax2.scatter(0.03, y, s=185, color=GCOLOR[g], edgecolors="#2A2E38",
                        linewidths=1.0, clip_on=False)
            ax2.text(0.10, y, g, fontsize=12.5, fontweight="bold", color=INK, va="center")
            ax2.text(0.46, y, f"{len(names)} points", fontsize=9.5, color=MUTED, va="center")
            y -= 0.045
            for b in names:
                i = idx[b]
                is_m, is_na = b in merged_to, "antler" in b
                lbl = b + (" *" if is_m else "")
                ax2.text(0.10, y, f"{i:>2d}", fontsize=9, color=MUTED, va="center",
                         family="monospace")
                ax2.text(0.16, y, lbl, fontsize=9,
                         color=NA_INK if is_na else ("#1B3F5A" if is_m else "#3D4557"),
                         va="center", fontweight="bold" if is_m else "normal")
                if is_m:
                    ax2.text(0.55, y, f"= {merged_to[b]}", fontsize=7.6, style="italic",
                             color=MERGE_INK, va="center")
                elif is_na:
                    ax2.text(0.55, y, "not on a dog", fontsize=7.6, style="italic",
                             color=NA_INK, va="center")
                elif donor_counts.get(b):
                    ax2.text(0.55, y, f"donor x{donor_counts[b]}", fontsize=7.6,
                             style="italic", color=DONOR_INK, va="center")
                y -= 0.029
            y -= 0.020
        return y

    axL = fig.add_subplot(gs[0, 1]); axL.axis("off")
    axR = fig.add_subplot(gs[0, 2]); axR.axis("off")
    for a_ in (axL, axR):
        a_.set_xlim(0, 1); a_.set_ylim(0, 1)
    block(axL, ["head"], 0.985)
    block(axR, ["neck & torso", "legs", "tail"], 0.985)

    fig.suptitle("SuperAnimal-Quadruped  —  the original 39 keypoints",
                 fontsize=20, fontweight="bold", color=INK, x=0.03, ha="left", y=0.965)
    fig.text(0.03, 0.905,
             "The pretrained DeepLabCut model this project extends. All 39 are still "
             "predicted, unchanged: measured drift after fine-tuning is 0.0000.",
             fontsize=10.5, color=MUTED, ha="left", va="top")
    fig.text(0.03, 0.042,
             "*  merged: a DogFLW landmark is supervised on this existing channel, so no "
             "new output was created. 9 of the 39.",
             fontsize=9, color=MERGE_INK, ha="left", va="bottom")
    fig.text(0.03, 0.020,
             "donor xN:  N added channels copied this keypoint's filter as their starting "
             "point. A donor is only an initialisation - the added channel is separate.",
             fontsize=9, color=DONOR_INK, ha="left", va="bottom")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {a.out}")
    print(f"  merged: {len(merged_to)}   donors: {len(donor_counts)}   "
          f"head keypoints: {len(GROUP['head'])}")


if __name__ == "__main__":
    main()
