"""Measure SuperAnimal's body keypoints against StanfordExtra ground truth.

The first accuracy numbers for stage 1 on this project. Every body figure quoted until
now - in the feature spec, in the SNR experiment, in CLAUDE.md - has been an estimate;
the face model had per-landmark NME on 574 val images and the body model had nothing.

Both datasets are built on Stanford Dogs, so DogFLW ids join StanfordExtra image names
directly: 2,545 of 4,335 images overlap.

Two properties were verified from the data before writing this, not assumed:

**Laterality is DIRECT.** StanfordExtra's "left" is SuperAnimal's "left" (the animal's
own left). Median normalised error 0.0165 direct against 0.1667 with the sides swapped -
a 10x gap, so there is no ambiguity.

**The mapping below is empirical.** For each StanfordExtra joint the nearest SuperAnimal
keypoint was computed over the overlap rather than taken from the published keypoint
ordering, and all twenty agree with the documented names except where argmax collapse
makes the answer arbitrary (see COLLAPSED).

**Only 20 of the 24 joints carry annotations.** Indices 20-23 - both eyes, withers and
throat - are flag 0 in all 2,545 overlapping images: slots in the array that v12 never
filled. Withers is the expensive absence, because it is `back_base`, which together with
`back_end` defines the body length every body feature divides by.

    python src/check_body_accuracy.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import superanimal as sa

# StanfordExtra joint index -> SuperAnimal bodypart. Verified empirically.
MAPPING: dict[int, str] = {
    0: "front_left_paw",   1: "front_left_knee",   2: "front_left_thai",
    3: "back_left_paw",    4: "back_left_knee",    5: "back_left_thai",
    6: "front_right_paw",  7: "front_right_knee",  8: "front_right_thai",
    9: "back_right_paw",  10: "back_right_knee",  11: "back_right_thai",
    12: "tail_base",      13: "tail_end",
    14: "left_earbase",   15: "right_earbase",
    16: "nose",           17: "lower_jaw",
    18: "left_earend",    19: "right_earend",
}

# Where argmax collapse makes one annotation the ground truth for two channels. The
# nearest-neighbour test picks between them arbitrarily, so both are scored against the
# same joint rather than letting a coin toss decide which one gets measured.
COLLAPSED: dict[int, list[str]] = {
    12: ["back_end"],            # tail_base == back_end in 95.6% of images
    13: ["right_antler_end", "left_antler_end"],
    15: ["right_antler_base"],
    14: ["left_antler_base"],
}

# Never annotated in v12 - recorded so the gap is visible in the output rather than
# looking like a keypoint that merely scored badly.
UNANNOTATED = {20: "L eye", 21: "R eye", 22: "Withers (back_base)", 23: "Throat"}

JOINT_NAMES = [
    "LF paw", "LF knee", "LF top", "LR paw", "LR knee", "LR top",
    "RF paw", "RF knee", "RF top", "RR paw", "RR knee", "RR top",
    "Tail start", "Tail end", "L ear base", "R ear base",
    "Nose", "Chin", "L ear tip", "R ear tip", "L eye", "R eye", "Withers", "Throat",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stanfordextra",
                    default="data/stanfordextra_v12/StanfordExtra_V12/StanfordExtra_v12.json")
    ap.add_argument("--poses", default="data/sa_dogboxes.npz")
    ap.add_argument("--pcut", type=float, default=0.35)
    ap.add_argument("--allow-multiple-dogs", action="store_true",
                    help="by default images with several dogs are skipped: the detector "
                         "picks one dog and the annotation may describe another")
    ap.add_argument("--out", default="outputs/body_accuracy.json")
    a = ap.parse_args()

    SA = sa.sa_bodyparts()
    se = json.loads(Path(a.stanfordextra).read_text())
    z = np.load(a.poses, allow_pickle=True)
    poses = {i: p for i, p in zip(z["ids"].tolist(), z["poses"])}

    targets: dict[int, list[str]] = {
        k: [v] + COLLAPSED.get(k, []) for k, v in MAPPING.items()
    }
    err: dict[str, list[float]] = {}
    seen = skipped_multi = 0

    for r in se:
        sid = Path(r["img_path"]).stem
        p = poses.get(sid)
        if p is None:
            continue
        if r.get("is_multiple_dogs") and not a.allow_multiple_dogs:
            skipped_multi += 1
            continue
        b = r["img_bbox"]                       # x, y, w, h
        diag = float(np.hypot(b[2], b[3]))      # the analogue of the face-box diagonal
        if diag < 1:
            continue
        seen += 1
        J = np.asarray(r["joints"], dtype=float)
        for k, names in targets.items():
            if J[k, 2] <= 0:
                continue
            for name in names:
                q = p[SA.index(name)]
                if q[2] < a.pcut:
                    continue
                err.setdefault(name, []).append(
                    float(np.linalg.norm(q[:2] - J[k, :2])) / diag)

    rows = {}
    for name, vals in err.items():
        e = np.asarray(vals)
        rows[name] = {
            "n": int(len(e)), "nme": float(e.mean()), "median": float(np.median(e)),
            "pck5": float((e < 0.05).mean()), "pck10": float((e < 0.10).mean()),
        }
    pooled = np.concatenate([np.asarray(v) for v in err.values()]) if err else np.array([])

    out = {
        "images_scored": seen, "skipped_multiple_dogs": skipped_multi,
        "pcut": a.pcut,
        "normaliser": "StanfordExtra dog-box diagonal",
        "unannotated_in_v12": list(UNANNOTATED.values()),
        "overall": {
            "n": int(pooled.size), "nme": float(pooled.mean()),
            "median": float(np.median(pooled)),
            "pck5": float((pooled < 0.05).mean()),
            "pck10": float((pooled < 0.10).mean()),
        } if pooled.size else {},
        "per_keypoint": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"\n{'='*74}\nSUPERANIMAL BODY ACCURACY vs StanfordExtra ground truth")
    print(f"{'='*74}")
    print(f"images scored {seen}   skipped (multiple dogs) {skipped_multi}   "
          f"pcut {a.pcut}")
    print("normalised by the dog-box diagonal, the analogue of the face-box diagonal "
          "used for face NME\n")
    print(f"{'SuperAnimal keypoint':22s} {'n':>5s} {'NME':>8s} {'median':>8s} "
          f"{'PCK@5':>7s} {'PCK@10':>7s}")
    for name, v in sorted(rows.items(), key=lambda kv: kv[1]["nme"]):
        print(f"  {name:20s} {v['n']:5d} {v['nme']:8.4f} {v['median']:8.4f} "
              f"{v['pck5']*100:6.1f}% {v['pck10']*100:6.1f}%")
    if pooled.size:
        o = out["overall"]
        print(f"  {'ALL':20s} {o['n']:5d} {o['nme']:8.4f} {o['median']:8.4f} "
              f"{o['pck5']*100:6.1f}% {o['pck10']*100:6.1f}%")
    print("\nNO GROUND TRUTH (flag 0 in every overlapping image):")
    for i, n in UNANNOTATED.items():
        print(f"  joint {i:2d}  {n}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
