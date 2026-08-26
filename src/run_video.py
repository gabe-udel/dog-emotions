"""Run the fine-tuned model on a video and write a labelled MP4.

Also supports rendering the original SuperAnimal-Quadruped model side by side, which is
what actually demonstrates the added keypoints: same frames, 39 keypoints on the left,
39 + the new DogFLW face keypoints on the right.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, "src")
import superanimal as sa
from draw import Renderer, banner, legend, REGION_COLOR, SA_COLOR

DETECTOR = "fasterrcnn_mobilenet_v3_large_fpn"


def build_pose_runner(model_cfg_path, snapshot, n_kpt, device="cpu"):
    from deeplabcut.core.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    cfg = read_config_as_dict(model_cfg_path)
    cfg["device"] = device
    pose_runner, _ = get_inference_runners(
        cfg, snapshot_path=snapshot, max_individuals=1,
        num_bodyparts=n_kpt, num_unique_bodyparts=0, device=device, detector_path=None)
    return pose_runner


def face_inset(clean, kpts, rend, pcut, size=300):
    """Zoomed crop around the predicted face, so the dense landmarks are legible."""
    pts = kpts[rend.face_indices]
    vis = pts[:, 2] >= pcut
    if vis.sum() < 10:
        return None
    x1, y1 = pts[vis, :2].min(0)
    x2, y2 = pts[vis, :2].max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.75 + 6
    H, W = clean.shape[:2]
    half = min(half, cx, cy, W - cx, H - cy)
    if half < 20:
        return None
    a, b = int(cx - half), int(cy - half)
    crop = clean[b:b + int(2 * half), a:a + int(2 * half)]
    if crop.size == 0:
        return None
    sc = size / crop.shape[0]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    k = kpts.copy()
    k[:, 0] = (k[:, 0] - a) * sc
    k[:, 1] = (k[:, 1] - b) * sc
    rend.draw(crop, k, pcut=pcut, r=3, thick=2, face_only=True)
    cv2.rectangle(crop, (0, 0), (size - 1, size - 1), (255, 255, 255), 2)
    cv2.putText(crop, "face landmarks (zoom)", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def smooth_series(k, win):
    """Temporal median over `win` frames, applied only where the score passes."""
    if win <= 1:
        return k
    out = k.copy()
    h = win // 2
    for i in range(len(k)):
        lo, hi = max(0, i - h), min(len(k), i + h + 1)
        w = k[lo:hi]
        ok = np.isfinite(w).all(2)
        for j in range(k.shape[1]):
            m = ok[:, j]
            if m.sum() >= 2:
                out[i, j, :2] = np.median(w[m, j, :2], axis=0)
    return out


def detect_boxes(frames_dir, paths, det_runner, min_score=0.5):
    out = []
    for d in det_runner.inference(images=paths):
        bb = np.asarray(d["bboxes"], float).reshape(-1, 4)
        sc = np.asarray(d.get("bbox_scores", []), float).reshape(-1)
        keep = [(s, b) for b, s in zip(bb, sc) if s >= min_score]
        out.append(max(keep, key=lambda t: t[0])[1] if keep else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--pcutoff", type=float, default=0.25)
    ap.add_argument("--compare", action="store_true", help="side-by-side vs original SuperAnimal")
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--smooth", type=int, default=1, help="temporal median window (1 = off)")
    ap.add_argument("--no-inset", action="store_true")
    ap.add_argument("--also-solo", default="", help="additionally write the fine-tuned panel alone")
    args = ap.parse_args()

    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

    rend = Renderer()
    n_kpt = len(rend.bodyparts)

    # ---- decode frames to a scratch dir (the DLC runners take file paths) ----
    scratch = Path("/tmp/dlc_frames"); scratch.mkdir(exist_ok=True)
    for f in scratch.glob("*.jpg"):
        f.unlink()
    cap = cv2.VideoCapture(args.video)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 24.0
    paths, frames = [], []
    k, seen = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok or (args.max_frames and k >= args.max_frames):
            break
        seen += 1
        if seen <= args.start_frame:
            continue
        h, w = fr.shape[:2]
        if w != args.width:
            fr = cv2.resize(fr, (args.width, int(round(h * args.width / w))), interpolation=cv2.INTER_AREA)
        p = scratch / f"f{k:05d}.jpg"
        cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(str(p)); frames.append(fr); k += 1
    cap.release()
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames @ {W}x{H}, {fps:.2f} fps", flush=True)

    # ---- detector ----
    sa.DETECTOR = DETECTOR
    det_cfg = PoseConfig.build_for_superanimal_inference(
        sa.SUPER_ANIMAL, model_name=sa.POSE_MODEL, detector_name=DETECTOR,
        max_individuals=1, device="cpu").to_dict()
    _, det_snap = sa.snapshot_paths()
    _, det_runner = get_inference_runners(
        det_cfg, snapshot_path=sa.snapshot_paths()[0], max_individuals=1,
        num_bodyparts=39, num_unique_bodyparts=0, device="cpu", detector_path=det_snap)
    t = time.time()
    boxes = detect_boxes(scratch, paths, det_runner)
    n_det = sum(b is not None for b in boxes)
    print(f"detector: {n_det}/{len(boxes)} frames with a dog  ({time.time()-t:.0f}s)", flush=True)

    # fill gaps with the nearest available box so the overlay does not flicker
    last = None
    for i, b in enumerate(boxes):
        if b is None:
            boxes[i] = last
        else:
            last = b
    for i in range(len(boxes) - 1, -1, -1):
        if boxes[i] is None:
            boxes[i] = next((b for b in boxes[i:] if b is not None), None)

    def run_pose(runner, nk):
        items = [(p, {"bboxes": np.array([b])}) for p, b in zip(paths, boxes) if b is not None]
        idxs = [i for i, b in enumerate(boxes) if b is not None]
        res = np.full((len(paths), nk, 3), np.nan)
        t0 = time.time()
        out, B = [], 24
        for i in range(0, len(items), B):
            out += runner.inference(items[i:i + B])
            print(f"  pose {min(i+B,len(items))}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        for i, r in zip(idxs, out):
            res[i] = np.asarray(r["bodyparts"])[0]
        return res

    pose_new = build_pose_runner(args.config, args.snapshot, n_kpt)
    kp_new = run_pose(pose_new, n_kpt)

    kp_old = None
    if args.compare:
        # detector_name must stay set so the config remains top-down (see superanimal.py)
        old_cfg = PoseConfig.build_for_superanimal_inference(
            sa.SUPER_ANIMAL, model_name=sa.POSE_MODEL, detector_name=DETECTOR,
            max_individuals=1, device="cpu").to_dict()
        old_runner, _ = get_inference_runners(
            old_cfg, snapshot_path=sa.snapshot_paths()[0], max_individuals=1,
            num_bodyparts=39, num_unique_bodyparts=0, device="cpu", detector_path=None)
        kp_old = run_pose(old_runner, 39)

    if args.smooth > 1:
        kp_new = smooth_series(kp_new, args.smooth)
        if kp_old is not None:
            kp_old = smooth_series(kp_old, args.smooth)

    # ---- render ----
    n_new = n_kpt - rend.n_sa
    out_w = W * 2 if args.compare else W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, fps, (out_w, H))
    vw_solo = cv2.VideoWriter(args.also_solo, fourcc, fps, (W, H)) if args.also_solo else None
    leg = [("SuperAnimal body (39)", SA_COLOR)] + [(f"{k} (DogFLW)", v) for k, v in REGION_COLOR.items()]
    inset_size = int(H * 0.42)
    for i, fr in enumerate(frames):
        right = fr.copy()
        kn = np.nan_to_num(kp_new[i], nan=-1)
        rend.draw(right, kn, pcut=args.pcutoff, r=3)
        if not args.no_inset:
            ins = face_inset(fr, kn, rend, args.pcutoff, size=inset_size)
            if ins is not None:
                right[H - inset_size - 12:H - 12, 12:12 + inset_size] = ins
        banner(right, "Fine-tuned: SuperAnimal-Quadruped + DogFLW face",
               f"{n_kpt} keypoints  =  39 body  +  {n_new} added facial landmarks")
        legend(right, leg, (W - 235, 60))
        if args.compare:
            left = fr.copy()
            k39 = np.nan_to_num(kp_old[i], nan=-1)
            pad = np.full((n_kpt, 3), -1.0); pad[:39] = k39
            rend.draw(left, pad, pcut=args.pcutoff, r=3)
            banner(left, "Original: SuperAnimal-Quadruped (HRNet-w32)", "39 keypoints, no facial landmarks")
            frame = np.hstack([left, right])
            cv2.line(frame, (W, 0), (W, H), (255, 255, 255), 2)
        else:
            frame = right
        vw.write(frame)
        if vw_solo is not None:
            vw_solo.write(right)
    vw.release()
    if vw_solo is not None:
        vw_solo.release()
        print("wrote", args.also_solo)
    for f in scratch.glob("*.jpg"):
        f.unlink()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
