"""Run the fine-tuned model on a video and write a labelled MP4.

Also supports rendering the original SuperAnimal-Quadruped model side by side, which is
what actually demonstrates the added keypoints: same frames, 39 keypoints on the left,
39 + the new DogFLW face keypoints on the right.
"""
from __future__ import annotations
import argparse, json, sys, tempfile, time
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


def head_boxes(kp, boxes, rend, shape, frac, pcut):
    """Re-cut each detector box so the predicted face fills `frac` of its long side.

    The pose head sees a top-down crop of whatever box it is given. On DogFLW the
    detector box puts the face at a median 56% of the box's long side; on video the
    same detector returns a whole-dog box and the face drops to ~25%, which measures
    about 13% worse NME on a ground-truth sweep. This re-crops to the measured
    optimum. Frames where pass 1 found too few face points keep their original box.
    """
    H, W = shape
    out = []
    for i, b in enumerate(boxes):
        pts = kp[i][rend.face_indices]
        v = np.isfinite(pts).all(1) & (pts[:, 2] >= pcut)
        if b is None or v.sum() < 6:
            out.append(b)
            continue
        p = pts[v, :2]
        cx, cy = p.mean(0)
        side = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) / frac
        a = float(np.clip(cx - side / 2, 0, W - 1))
        t = float(np.clip(cy - side / 2, 0, H - 1))
        out.append(np.array([a, t, min(side, W - a), min(side, H - t)]))
    return out


def gate_second_pass(kp, boxes, rend, criterion, conf_thresh, fill_thresh):
    """Which frames actually need the head-cropped second pose pass.

    The point of gating is compute: a second pass costs as much as the first, and on
    frames where the face already fills the box properly it changes almost nothing.

    Two criteria, and they do not agree:

    'scale'      - re-run when the face fills less than `fill_thresh` of the detector
                   box.  This is the quantity head-cropping actually corrects, and the
                   ground-truth sweep is monotonic in it: NME on the 30 reliable
                   landmarks runs 0.0411 / 0.0362 / 0.0328 at 25% / 45% / 55% fill.

    'confidence' - re-run when median confidence over the reliable landmarks is below
                   `conf_thresh`.  Requested, and supported, but be aware it is
                   inverted on this model: across that same sweep confidence went 0.807
                   -> 0.705 as accuracy IMPROVED.  High confidence indicates a bad
                   crop here, so this gate tends to skip the second pass exactly when
                   it would have helped most.

    Returns (bool array, stats dict).
    """
    from keypoint_scheme import RELIABLE
    rel = [rend.d2m[d] for d in RELIABLE if d in rend.d2m]
    n = len(kp)
    need = np.zeros(n, bool)
    fills = np.full(n, np.nan)
    confs = np.full(n, np.nan)
    for i in range(n):
        if boxes[i] is None or not np.isfinite(kp[i][:, :2]).any():
            continue
        pts = kp[i][rend.face_indices]
        v = np.isfinite(pts).all(1) & (pts[:, 2] >= 0.1)
        if v.sum() >= 6:
            p = pts[v, :2]
            span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
            fills[i] = span / max(boxes[i][2], boxes[i][3])
        confs[i] = np.nanmedian(kp[i][rel, 2])

    if criterion == "always":
        need[:] = [b is not None for b in boxes]
    elif criterion == "never":
        pass
    elif criterion == "confidence":
        need = np.nan_to_num(confs, nan=0.0) < conf_thresh
    else:                                    # 'scale'
        need = np.nan_to_num(fills, nan=0.0) < fill_thresh
    need &= np.array([b is not None for b in boxes])
    return need, {"fill": fills, "conf": confs}


def ema_fuse(k, lo, hi, alpha):
    """Causal exponential moving average on mid-confidence keypoints only.

    Points above `hi` are already steady - measured frame-to-frame jitter is 0.009 of
    the detector box - and smoothing them only adds lag when the dog moves.  Points
    below `lo` are not drawn at all.  The band between is where jitter is both visible
    and worth fixing.

    alpha is the weight on the current frame: 1.0 disables smoothing, lower is smoother
    and laggier.
    """
    if alpha >= 1.0 or len(k) < 2:
        return k
    out = k.copy()
    prev = None
    for i in range(len(out)):
        cur = out[i]
        if prev is not None:
            band = ((cur[:, 2] >= lo) & (cur[:, 2] < hi)
                    & np.isfinite(cur[:, :2]).all(1) & np.isfinite(prev[:, :2]).all(1))
            if band.any():
                cur[band, :2] = alpha * cur[band, :2] + (1 - alpha) * prev[band, :2]
        prev = cur.copy()
    return out


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
    ap.add_argument("--no-inset", action="store_true", help="drop the zoomed face panel")
    ap.add_argument("--bare", action="store_true",
                    help="keypoints only: no zoom inset, no title banner, no legend")
    ap.add_argument("--no-subpixel", action="store_true",
                    help="disable sub-cell peak fitting. Without it keypoints snap to "
                         "the 64x64 heatmap grid (4 px in the crop) and dense facial "
                         "landmarks decode to identical pixels - see subpixel.py")
    ap.add_argument("--no-lines", action="store_true",
                    help="draw bare points with no skeleton edges or face contours")
    ap.add_argument("--legend", action="store_true",
                    help="force the colour legend on even with --bare")
    ap.add_argument("--landmarks", choices=["all", "reliable"], default="all",
                    help="'all' draws every face landmark; 'reliable' draws only the 32 "
                         "the model can actually place (30 detected + 2 shape-derived) "
                         "and omits the 14 ear contour points measured unlearnable. "
                         "Deterministic - unlike a confidence threshold it does not "
                         "shift with how confident the model happens to be on a clip")
    ap.add_argument("--no-ear-correct", action="store_true",
                    help="skip the per-ear-type bias correction. It is on by default: "
                         "it classifies ear type from the model's own ear landmarks "
                         "and subtracts that type's systematic offset, measuring "
                         "ear NME 0.0882 -> 0.0631 on the test split")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the shape-model correction of the head-top landmarks. "
                         "The correction is on by default: it measures 43%% better than "
                         "the CNN on those two channels (NME 0.130 -> 0.074)")
    ap.add_argument("--head-crop", type=float, default=0.0, metavar="FRAC",
                    help="two-pass: after the first pose pass, re-cut the box so the face "
                         "fills FRAC of it and run pose again. 0 = off. 0.55 measured best "
                         "on the DogFLW test split; the whole-dog box gives ~0.25 on video")
    ap.add_argument("--gate", choices=["always", "scale", "confidence", "never"],
                    default="scale",
                    help="which frames get the second pass. 'scale' (default) re-runs "
                         "when the face fills less than --gate-fill of the box - the "
                         "quantity head-cropping corrects. 'confidence' re-runs below "
                         "--gate-conf, but confidence is inverted against accuracy on "
                         "this model, so it skips frames that needed the pass")
    ap.add_argument("--gate-conf", type=float, default=0.75,
                    help="confidence gate threshold (used with --gate confidence)")
    ap.add_argument("--gate-fill", type=float, default=0.45,
                    help="scale gate threshold: re-run when face/box is below this")
    ap.add_argument("--ema", type=float, default=1.0, metavar="ALPHA",
                    help="temporal EMA weight on the current frame, applied only to "
                         "keypoints inside the confidence band. 1.0 = off, 0.5 = "
                         "moderate smoothing. Suppresses jitter without lagging the "
                         "high-confidence points")
    ap.add_argument("--ema-band", type=float, nargs=2, default=[0.35, 0.75],
                    metavar=("LO", "HI"),
                    help="confidence band the EMA applies to (default 0.35 0.75)")
    ap.add_argument("--also-solo", default="", help="additionally write the fine-tuned panel alone")
    args = ap.parse_args()

    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

    if not args.no_subpixel:
        import subpixel
        if subpixel.enable():
            print("subpixel: parabolic peak fitting enabled (argmax would quantise "
                  "keypoints to the 4 px heatmap grid)", flush=True)

    rend = Renderer()
    n_kpt = len(rend.bodyparts)

    # ---- decode frames to a scratch dir (the DLC runners take file paths) ----
    scratch = Path(tempfile.gettempdir()) / "dlc_frames"
    scratch.mkdir(parents=True, exist_ok=True)
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

    def run_pose(runner, nk, bxs=None):
        bxs = boxes if bxs is None else bxs
        items = [(p, {"bboxes": np.array([b])}) for p, b in zip(paths, bxs) if b is not None]
        idxs = [i for i, b in enumerate(bxs) if b is not None]
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
    if args.head_crop > 0:
        need, st = gate_second_pass(kp_new, boxes, rend, args.gate,
                                    args.gate_conf, args.gate_fill)
        fill, conf = np.nanmedian(st["fill"]), np.nanmedian(st["conf"])
        print(f"gate '{args.gate}': median face/box {fill:.0%}, median confidence "
              f"{conf:.2f} -> second pass on {need.sum()}/{len(need)} frames "
              f"({need.sum()/max(len(need),1):.0%} of the compute a full pass would cost)",
              flush=True)
        if need.any():
            hb = head_boxes(kp_new, boxes, rend, (H, W), args.head_crop, args.pcutoff)
            # gated frames keep their pass-1 result: None boxes are skipped by run_pose
            hb = [b if need[i] else None for i, b in enumerate(hb)]
            kp2 = run_pose(pose_new, n_kpt, hb)
            ok = (need & np.isfinite(kp2[:, :, 0]).any(1)).nonzero()[0]
            # Take ONLY the face channels from the cropped pass. A head-tight crop puts
            # the neck, back, legs and tail outside the frame the pose head sees, so
            # pass 2 places just 9 of the 30 body-only keypoints against pass 1's 27.
            # Pass 1 saw the whole dog and keeps the body; pass 2 saw the face properly
            # and keeps the face.
            fch = rend.face_indices
            kp_new[np.ix_(ok, fch)] = kp2[np.ix_(ok, fch)]
            print(f"head crop: took {len(fch)} face channels from the cropped pass on "
                  f"{len(ok)} frames; body keypoints kept from pass 1", flush=True)

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

    # Ear-type bias correction, before the shape model so head_top is derived from a
    # settled face. One channel serves both erect and floppy ears, so the model averages
    # the two geometries and errs in a consistent direction; see ear_correct.py.
    if not args.no_ear_correct:
        try:
            from ear_correct import EarCorrector
            ec = EarCorrector()
            got = [ec.apply(kp_new[i]) for i in range(len(kp_new))
                   if np.isfinite(kp_new[i]).any()]
            got = [g for g in got if g]
            if got:
                counts = {t: got.count(t) for t in sorted(set(got))}
                print(f"ear correct: {len(got)}/{len(kp_new)} frames, ear type "
                      f"{counts}", flush=True)
        except FileNotFoundError:
            print("ear correct: data/ear_bias.pkl not found - skipping "
                  "(run: python src/ear_correct.py --fit)", flush=True)

    # Shape-model correction, before smoothing so the temporal median sees corrected
    # points. Only touches channels the CNN measurably cannot see; see shape_refine.py.
    if not args.no_refine:
        try:
            from shape_refine import Refiner
            ref = Refiner()
            n_ref = sum(ref.apply(kp_new[i]) > 0 for i in range(len(kp_new))
                        if np.isfinite(kp_new[i]).any())
            print(f"shape refine: corrected {', '.join(ref.names)} on {n_ref}/{len(kp_new)}"
                  f" frames", flush=True)
        except FileNotFoundError:
            print("shape refine: data/shape_model.npz not found - skipping "
                  "(run: python src/shape_refine.py --fit)", flush=True)

    # Drop the unlearnable channels by score, so both the points and every contour
    # segment touching them disappear through the existing visibility test.
    if args.landmarks == "reliable":
        from keypoint_scheme import UNRELIABLE
        drop = [rend.d2m[d] for d in UNRELIABLE if d in rend.d2m]
        kp_new[:, drop, 2] = -1.0
        print(f"landmarks: drawing the reliable {len(rend.face_indices) - len(drop)} of "
              f"{len(rend.face_indices)} face landmarks; {len(drop)} ear contour points "
              f"omitted", flush=True)

    if args.smooth > 1:
        kp_new = smooth_series(kp_new, args.smooth)
        if kp_old is not None:
            kp_old = smooth_series(kp_old, args.smooth)

    if args.ema < 1.0:
        lo, hi = args.ema_band
        before = np.nanmedian(np.linalg.norm(np.diff(kp_new[:, :, :2], axis=0), axis=2))
        kp_new = ema_fuse(kp_new, lo, hi, args.ema)
        after = np.nanmedian(np.linalg.norm(np.diff(kp_new[:, :, :2], axis=0), axis=2))
        print(f"temporal EMA alpha={args.ema} on the {lo}-{hi} confidence band: "
              f"median frame-to-frame motion {before:.2f} -> {after:.2f} px", flush=True)

    # What actually reaches the screen. NaN fails the comparison, so unplaced channels
    # are excluded automatically. Reported because every stage above can remove points
    # and the arithmetic across stages is not obvious.
    vis = kp_new[:, :, 2] >= args.pcutoff
    body_ch = [i for i in range(n_kpt) if i not in set(rend.face_indices)]
    print(f"DRAWN at pcutoff {args.pcutoff}: {vis.sum(1).mean():.1f} of {n_kpt} per frame"
          f"  ({vis[:, rend.face_indices].sum(1).mean():.1f}/{len(rend.face_indices)} face,"
          f" {vis[:, body_ch].sum(1).mean():.1f}/{len(body_ch)} body-only)", flush=True)

    # ---- render ----
    n_new = n_kpt - rend.n_sa
    out_w = W * 2 if args.compare else W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, fps, (out_w, H))
    vw_solo = cv2.VideoWriter(args.also_solo, fourcc, fps, (W, H)) if args.also_solo else None
    # Region names as the legend, ordered head-to-tail so it reads like the animal.
    leg = ([(k, REGION_COLOR[k]) for k in
            ("ear", "head", "eye", "nose", "muzzle", "mouth") if k in REGION_COLOR]
           + [("body / skeleton", SA_COLOR)])
    show_legend = args.legend or not args.bare
    lines = not args.no_lines
    inset_size = int(H * 0.42)
    for i, fr in enumerate(frames):
        right = fr.copy()
        kn = np.nan_to_num(kp_new[i], nan=-1)
        rend.draw(right, kn, pcut=args.pcutoff, r=3, lines=lines)
        if not args.no_inset and not args.bare:
            ins = face_inset(fr, kn, rend, args.pcutoff, size=inset_size)
            if ins is not None:
                right[H - inset_size - 12:H - 12, 12:12 + inset_size] = ins
        if not args.bare:
            banner(right, "Fine-tuned: SuperAnimal-Quadruped + DogFLW face",
                   f"{n_kpt} keypoints  =  39 body  +  {n_new} added facial landmarks")
        if show_legend:
            legend(right, leg, (W - 235, 34 if args.bare else 60))
        if args.compare:
            left = fr.copy()
            k39 = np.nan_to_num(kp_old[i], nan=-1)
            pad = np.full((n_kpt, 3), -1.0); pad[:39] = k39
            rend.draw(left, pad, pcut=args.pcutoff, r=3, lines=lines)
            if not args.bare:
                banner(left, "Original: SuperAnimal-Quadruped (HRNet-w32)",
                       "39 keypoints, no facial landmarks")
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
