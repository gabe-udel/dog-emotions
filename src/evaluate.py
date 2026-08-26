"""Evaluate the fine-tuned model on the DogFLW test split, and check for forgetting.

Reports:
  * NME (normalised mean error) of every facial landmark, as a fraction of the face
    bounding-box diagonal - the standard facial-landmark metric.
  * PCK at 5% / 10% of the face diagonal.
  * For the keypoints that already existed in SuperAnimal, the same numbers for the
    original model, so the fine-tune can be checked for regression.
  * Mean displacement of the 39 SuperAnimal body keypoints between the original and the
    fine-tuned model, i.e. how much memory replay preserved.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import superanimal as sa


def pose_runner_from(config, snapshot, n_kpt):
    from deeplabcut.core.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    cfg = read_config_as_dict(config); cfg["device"] = "cpu"
    r, _ = get_inference_runners(cfg, snapshot_path=snapshot, max_individuals=1,
                                 num_bodyparts=n_kpt, num_unique_bodyparts=0,
                                 device="cpu", detector_path=None)
    return r


def infer(runner, items, nk, label):
    res, t, B = [], time.time(), 32
    for i in range(0, len(items), B):
        res += runner.inference(items[i:i + B])
        if (i // B) % 4 == 0:
            print(f"  {label} {min(i+B,len(items))}/{len(items)} ({time.time()-t:.0f}s)", flush=True)
    return np.stack([np.asarray(r["bodyparts"])[0] for r in res])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default="outputs/evaluation.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    km = json.load(open("data/keypoint_map.json"))
    bodyparts, sa_bps = km["bodyparts"], km["superanimal_bodyparts"]
    d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
    from keypoint_scheme import DOGFLW_NAMES, REGION_OF

    recs = [r for r in json.load(open("data/dogflw/annotations.json")) if r["split"] == "test"]
    ids = [r["id"] for r in recs]
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    order = {i: k for k, i in enumerate(z["ids"])}
    rows = [order[i] for i in ids]
    boxes, srcs = z["boxes"][rows], z["srcs"][rows]
    keep = [k for k in range(len(recs)) if srcs[k] > 0]
    if args.limit:
        keep = keep[:args.limit]
    print(f"evaluating on {len(keep)} of {len(recs)} DogFLW test images")

    items, L, diag = [], [], []
    for k in keep:
        r = recs[k]; x1, y1, x2, y2 = boxes[k]
        items.append((f"data/dogflw/{r['file']}", {"bboxes": np.array([[x1, y1, x2 - x1, y2 - y1]])}))
        L.append(r["landmarks"])
        bb = r["bbox_xyxy"]
        diag.append(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
    L = np.array(L); diag = np.array(diag)

    new = infer(pose_runner_from(args.config, args.snapshot, len(bodyparts)), items,
                len(bodyparts), "fine-tuned")
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    # detector_name must stay set so the config remains top-down (see superanimal.py)
    old_cfg = PoseConfig.build_for_superanimal_inference(
        sa.SUPER_ANIMAL, model_name=sa.POSE_MODEL,
        detector_name="fasterrcnn_mobilenet_v3_large_fpn",
        max_individuals=1, device="cpu").to_dict()
    old_runner, _ = get_inference_runners(old_cfg, snapshot_path=sa.snapshot_paths()[0],
                                          max_individuals=1, num_bodyparts=39,
                                          num_unique_bodyparts=0, device="cpu", detector_path=None)
    old = infer(old_runner, items, 39, "original")

    # ---- facial landmark accuracy ----
    per_lm, table = {}, []
    for di in range(46):
        mi = d2m[di]
        gt = L[:, di, :]
        ok = np.isfinite(gt).all(1)
        e = np.linalg.norm(new[:, mi, :2] - gt, axis=1)[ok] / diag[ok]
        row = {"dogflw_idx": di, "name": DOGFLW_NAMES[di], "region": REGION_OF[di],
               "model_idx": int(mi), "added": bool(mi >= 39),
               "nme": float(np.mean(e)), "median": float(np.median(e)),
               "pck05": float(np.mean(e < .05)), "pck10": float(np.mean(e < .10))}
        if mi < 39:      # existed in SuperAnimal -> can compare with the original model
            eo = np.linalg.norm(old[:, mi, :2] - gt, axis=1)[ok] / diag[ok]
            row["nme_original"] = float(np.mean(eo))
            row["pck10_original"] = float(np.mean(eo < .10))
        per_lm[DOGFLW_NAMES[di]] = row
        table.append(row)

    added = [r for r in table if r["added"]]
    kept = [r for r in table if not r["added"]]
    print(f"\n{'landmark':24s} {'region':7s} {'NME':>7s} {'PCK@5%':>7s} {'PCK@10%':>8s}   original NME")
    for r in sorted(table, key=lambda r: (r["region"], r["name"])):
        extra = f"   {r['nme_original']:.4f}" if "nme_original" in r else ""
        print(f"{r['name']:24s} {r['region']:7s} {r['nme']:7.4f} {r['pck05']:7.1%} {r['pck10']:8.1%}{extra}")

    print(f"\nadded facial keypoints ({len(added)}):  NME {np.mean([r['nme'] for r in added]):.4f}"
          f"  PCK@5% {np.mean([r['pck05'] for r in added]):.1%}"
          f"  PCK@10% {np.mean([r['pck10'] for r in added]):.1%}")
    if kept:
        print(f"kept SuperAnimal face keypoints ({len(kept)}):  NME {np.mean([r['nme'] for r in kept]):.4f}"
              f"  (original model {np.mean([r['nme_original'] for r in kept]):.4f})")

    # ---- forgetting check on the 39 SuperAnimal keypoints ----
    conf = old[:, :, 2] > 0.5
    disp = np.linalg.norm(new[:, :39, :2] - old[:, :, :2], axis=2) / diag[:, None]
    body = {}
    for j, bp in enumerate(sa_bps):
        m = conf[:, j]
        if m.sum() >= 10:
            body[bp] = {"median_shift": float(np.median(disp[m, j])),
                        "conf_original": float(np.median(old[m, j, 2])),
                        "conf_finetuned": float(np.median(new[m, j, 2]))}
    shifts = np.array([v["median_shift"] for v in body.values()])
    print(f"\nSuperAnimal keypoint drift after fine-tuning (median over confident detections):")
    print(f"  median across the 39 keypoints: {np.median(shifts):.4f} of the face diagonal")
    worst = sorted(body.items(), key=lambda kv: -kv[1]["median_shift"])[:6]
    for bp, v in worst:
        print(f"    {bp:22s} shift {v['median_shift']:.4f}   conf {v['conf_original']:.2f} -> {v['conf_finetuned']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_images": len(keep), "landmarks": table, "superanimal_drift": body,
               "summary": {
                   "added_nme": float(np.mean([r["nme"] for r in added])),
                   "added_pck05": float(np.mean([r["pck05"] for r in added])),
                   "added_pck10": float(np.mean([r["pck10"] for r in added])),
                   "kept_nme": float(np.mean([r["nme"] for r in kept])) if kept else None,
                   "kept_nme_original": float(np.mean([r["nme_original"] for r in kept])) if kept else None,
                   "median_sa_drift": float(np.median(shifts))}},
              open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
