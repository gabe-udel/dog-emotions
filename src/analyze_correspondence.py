"""Decide which DogFLW landmarks are *new* keypoints and which already exist in SuperAnimal.

Uses the SuperAnimal predictions computed on whole-dog crops (build_dogboxes.py).  A
DogFLW landmark is treated as the same point as a SuperAnimal keypoint only when the two
are mutual nearest neighbours AND their median separation is under MERGE_THRESH of the
face-bbox diagonal.  Everything else becomes an added keypoint.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import superanimal as sa
from keypoint_scheme import DOGFLW_NAMES, REGION_OF, TRIM_DROP, TRIM_KEEP

MERGE_THRESH = 0.05     # fraction of the face-bbox diagonal
MIN_SA_CONF = 0.3       # only trust SuperAnimal keypoints it is reasonably sure about

# SuperAnimal-Quadruped carries four antler keypoints for deer.  On a dog they are
# degenerate - the model puts antler_base on top of earbase - so they win the
# mutual-nearest-neighbour test against the ear landmarks purely by accident.  Attaching
# real dog annotations to an "antler" channel would be nonsense, so they are excluded from
# merging (they stay in the model, supervised by pseudo-labels like the other body parts).
NO_MERGE = {"right_antler_base", "right_antler_end", "left_antler_base", "left_antler_end"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", action="store_true",
                    help="drop dense contour points (keypoint_scheme.TRIM_DROP): 76 -> 63 outputs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output map (default: data/keypoint_map.json, "
                         "or data/keypoint_map_trim.json with --trim)")
    args = ap.parse_args()
    out_path = args.out or Path("data/keypoint_map_trim.json" if args.trim
                                else "data/keypoint_map.json")

    recs = json.load(open("data/dogflw/annotations.json"))
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    P, srcs = z["poses"], z["srcs"]
    bps = sa.sa_bodyparts()

    L = np.array([r["landmarks"] for r in recs])
    bb = np.array([r["bbox_xyxy"] for r in recs])
    diag = np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])

    ok = srcs > 0                                    # only images with a real detection
    P, L, diag = P[ok], L[ok], diag[ok]
    print(f"using {ok.sum()}/{len(ok)} images with a detector box\n")

    conf = np.nanmedian(P[:, :, 2], axis=0)
    d = np.linalg.norm(P[:, :, None, :2] - L[:, None, :, :], axis=-1) / diag[:, None, None]
    med = np.nanmedian(np.where(np.isfinite(d), d, np.nan), axis=0)     # (39, 46)

    sa_best = med.argmin(1)          # for each SA keypoint, nearest DogFLW landmark
    eligible = np.array([bp not in NO_MERGE for bp in bps])
    med_elig = np.where(eligible[:, None], med, np.inf)
    fl_best = med_elig.argmin(0)     # for each DogFLW landmark, nearest *mergeable* SA keypoint

    print("SuperAnimal keypoints whose nearest DogFLW landmark is within "
          f"{MERGE_THRESH} of the face diagonal:\n")
    print(f"{'SuperAnimal kpt':20s} {'conf':>5s} {'DogFLW':>3s} {'name':22s} {'dist':>6s} {'mutual':>6s} {'merge':>5s}")
    merge = {}      # sa_index -> dogflw_index
    for j, bp in enumerate(bps):
        i = int(sa_best[j])
        mutual = int(fl_best[i]) == j
        do = bool(mutual and med[j, i] < MERGE_THRESH and conf[j] >= MIN_SA_CONF
                  and bp not in NO_MERGE)
        if med[j, i] < 0.12 or do:
            print(f"{bp:20s} {conf[j]:5.2f} {i:3d} {DOGFLW_NAMES[i]:22s} {med[j,i]:6.3f} "
                  f"{str(mutual):>6s} {'YES' if do else '-':>5s}")
        if do:
            merge[j] = i

    new_idx = [i for i in range(46) if i not in set(merge.values())]
    if args.trim:
        dropped = [i for i in new_idx if i in TRIM_DROP]
        new_idx = [i for i in new_idx if i in TRIM_KEEP]
        print(f"\n--trim: dropping {len(dropped)} added contour channels: "
              f"{', '.join(f'{i} {DOGFLW_NAMES[i]}' for i in dropped)}")
        # A merged landmark is DogFLW ground truth sitting on a *SuperAnimal* channel, so
        # trimming one would not remove an output - only downgrade it to a pseudo-label.
        still = sorted(set(merge.values()) & TRIM_DROP)
        if still:
            print(f"  (kept as merged, costs no channel: "
                  f"{', '.join(f'{i} {DOGFLW_NAMES[i]}' for i in still)})")

    # Warm start for the added output channels: the heatmap head is a 1x1 conv over the
    # 32-channel HRNet feature map, so copying the filter of the *spatially nearest*
    # SuperAnimal keypoint makes a new channel start out predicting a nearby point
    # instead of noise - training then only has to shift it.
    donors = {int(i): int(np.nanargmin(med[:, i])) for i in new_idx}
    print(f"\nmerged {len(merge)} DogFLW landmarks onto existing SuperAnimal keypoints")
    print(f"added  {len(new_idx)} new keypoints -> model will have {39 + len(new_idx)} outputs")

    bodyparts = list(bps) + [DOGFLW_NAMES[i] for i in new_idx]
    # where each of the 46 DogFLW landmarks ends up in the final model output
    inv_merge = {int(i): j for j, i in merge.items()}
    # With --trim some landmarks map to no output channel at all, so they are simply
    # absent here; consumers must treat this as a lookup that can miss, not a dense list.
    supervised = set(inv_merge) | set(new_idx)
    dogflw_to_model = {i: (inv_merge[i] if i in inv_merge else 39 + new_idx.index(i))
                       for i in range(46) if i in supervised}
    out = {
        "superanimal_bodyparts": list(bps),
        "merge_sa_to_dogflw": {bps[j]: int(i) for j, i in merge.items()},
        "new_dogflw_indices": new_idx,
        "new_bodyparts": [DOGFLW_NAMES[i] for i in new_idx],
        "bodyparts": bodyparts,
        "regions": {DOGFLW_NAMES[i]: REGION_OF[i] for i in range(46)},
        "sa_median_conf": {bps[j]: float(conf[j]) for j in range(39)},
        "merge_threshold": MERGE_THRESH,
        "dogflw_to_model_idx": {str(i): int(j) for i, j in dogflw_to_model.items()},
        "init_donor": {DOGFLW_NAMES[i]: bps[j] for i, j in donors.items()},
        "init_donor_idx": {str(i): j for i, j in donors.items()},
        "new_kpt_donor_dist": {DOGFLW_NAMES[i]: float(med[j, i]) for i, j in donors.items()},
    }
    out["trimmed"] = bool(args.trim)
    out["dropped_dogflw_indices"] = sorted(TRIM_DROP) if args.trim else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
