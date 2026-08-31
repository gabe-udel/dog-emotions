"""Standalone legend image: what every keypoint colour and index means.

The face diagram is not schematic - it is the dataset's own mean shape, each of the
4,335 DogFLW faces normalised into its annotation box and averaged, which is also how
the landmark names in keypoint_scheme.py were derived.

    .venv\\Scripts\\python.exe src\\make_legend.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES, REGION_OF, UNRELIABLE, SHAPE_DERIVED
from draw import FACE_CONTOURS, REGION_COLOR, SA_COLOR


def bgr(t):
    return (t[2] / 255, t[1] / 255, t[0] / 255)


ORDER = ["ear", "head", "eye", "nose", "muzzle", "mouth"]
INK, MUTED, RULE = "#14171F", "#6B7488", "#D8DEE8"


def mean_shape():
    recs = json.load(open("data/dogflw/annotations.json"))
    acc = []
    for r in recs:
        lm = np.array(r["landmarks"], float)
        bb = r["bbox_xyxy"]
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if np.isfinite(lm).all() and w > 0 and h > 0:
            acc.append(np.stack([(lm[:, 0] - bb[0]) / w, (lm[:, 1] - bb[1]) / h], 1))
    return np.mean(acc, 0), len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/keypoint_legend.png")
    a = ap.parse_args()

    M, n = mean_shape()
    fig = plt.figure(figsize=(16.5, 8.6), dpi=200)
    fig.patch.set_facecolor("white")
    # Three columns: the diagram, then the key split in two so the page stays wide
    # rather than becoming a single 55-line column.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.92, 0.92], wspace=0.03,
                          left=0.03, right=0.98, top=0.845, bottom=0.045)

    # ---------------- face diagram ----------------
    ax = fig.add_subplot(gs[0, 0])
    for c in FACE_CONTOURS:
        pts = M[[i for i in c]]
        ax.plot(pts[:, 0], pts[:, 1], color=RULE, lw=1.6, zorder=1,
                solid_capstyle="round")
    for i in range(46):
        col = bgr(REGION_COLOR[REGION_OF[i]])
        dead = i in UNRELIABLE
        ax.scatter(M[i, 0], M[i, 1], s=150, color=col, zorder=3,
                   edgecolors="#2A2E38" if not dead else "#B23A2B",
                   linewidths=1.0 if not dead else 2.0)
        ax.annotate(str(i), (M[i, 0], M[i, 1]), color="#14171F", fontsize=6.4,
                    fontweight="bold", ha="center", va="center", zorder=4)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(1.16, -0.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 1.13, f"Mean face shape over all {n:,} DogFLW images  ·  "
                       "the animal's right is on the image left",
            transform=ax.transData, fontsize=9.5, color=MUTED, ha="center", va="top")

    # ---------------- key, split across two columns ----------------
    def block(ax2, regions, y):
        for reg in regions:
            idx = [i for i in range(46) if REGION_OF[i] == reg]
            ax2.scatter(0.03, y, s=185, color=bgr(REGION_COLOR[reg]),
                        edgecolors="#2A2E38", linewidths=1.0, clip_on=False)
            ax2.text(0.10, y, reg, fontsize=12.5, fontweight="bold", color=INK,
                     va="center")
            ax2.text(0.34, y, f"{len(idx)} points", fontsize=9.5, color=MUTED,
                     va="center")
            y -= 0.045
            for i in idx:
                dead = i in UNRELIABLE
                ax2.text(0.10, y, f"{i:>2d}", fontsize=9, color=MUTED, va="center",
                         family="monospace")
                ax2.text(0.16, y, DOGFLW_NAMES[i], fontsize=9,
                         color="#B23A2B" if dead else "#3D4557", va="center")
                if dead:
                    ax2.text(0.63, y, "not drawn", fontsize=8, style="italic",
                             color="#B23A2B", va="center")
                elif i in SHAPE_DERIVED:
                    ax2.text(0.63, y, "derived", fontsize=8, style="italic",
                             color="#1B7A50", va="center")
                y -= 0.029
            y -= 0.020
        return y

    axL = fig.add_subplot(gs[0, 1]); axL.axis("off")
    axR = fig.add_subplot(gs[0, 2]); axR.axis("off")
    for a_ in (axL, axR):
        a_.set_xlim(0, 1); a_.set_ylim(0, 1)
    block(axL, ["ear", "head", "eye"], 0.985)
    y = block(axR, ["nose", "muzzle", "mouth"], 0.985)

    axR.scatter(0.03, y, s=185, color=bgr(SA_COLOR), edgecolors="#2A2E38",
                linewidths=1.0, clip_on=False)
    axR.text(0.10, y, "body / skeleton", fontsize=12.5, fontweight="bold", color=INK,
             va="center")
    axR.text(0.52, y, "30 points", fontsize=9.5, color=MUTED, va="center")
    y -= 0.045
    for line in ("SuperAnimal-Quadruped body keypoints:",
                 "spine, legs, paws, tail. Not part of the 46",
                 "facial landmarks; unchanged by fine-tuning."):
        axR.text(0.10, y, line, fontsize=8.8, color=MUTED, va="center")
        y -= 0.029

    fig.suptitle("Dog face keypoints  —  76 outputs = 39 body + 37 added facial",
                 fontsize=20, fontweight="bold", color=INK, x=0.03, ha="left", y=0.965)
    fig.text(0.03, 0.905,
             "46 DogFLW facial landmarks. 9 share a channel with an existing "
             "SuperAnimal keypoint, so only 37 channels were added.",
             fontsize=10.5, color=MUTED, ha="left", va="top")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
