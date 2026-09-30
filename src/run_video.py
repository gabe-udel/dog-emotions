"""Run the two-stage cascade on a video and write a labelled MP4.

Rewritten for the cascade. Gone with the unified model: the two-pass head crop and its
gating (`--head-crop`, `--gate`, `--gate-fill`) and the EMA fusion that existed to
stabilise the second pass. The cascade *is* the head crop, applied every frame from a
box derived by the same function training used, so none of that machinery has a job.

    python src/run_video.py --video "Happy lab.mov" \
        --snapshot face_project/face1/snapshot-004.pt --out outputs/cascade.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import superanimal as sa
from aggregate import aggregate, load_noise_floors, motion_energy
from cascade import Cascade, FrameResult, config_beside
from draw import REGION_COLOR, SA_COLOR, Renderer, banner, legend
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from features import feature_names, frame_features
from keypoint_scheme import ACTIVE, ACTIVE_NAMES, REGION_OF
from temporal import ransac_filter, rolling_median, summarise, window_frames


def extract_frames(video: str, scratch: Path, width: int, start: int,
                   limit: int) -> tuple[list[str], float]:
    """Decode to JPEGs; the detector runner takes paths, not arrays."""
    scratch.mkdir(parents=True, exist_ok=True)
    for f in scratch.glob("*.jpg"):
        f.unlink()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    paths, i = [], 0
    while len(paths) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start:
            h, w = frame.shape[:2]
            if width and w != width:
                frame = cv2.resize(frame, (width, int(round(h * width / w))),
                                   interpolation=cv2.INTER_AREA)
            p = scratch / f"f{len(paths):05d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(p))
        i += 1
    cap.release()
    if not paths:
        raise SystemExit(f"no frames read from {video} (start-frame {start}?)")
    return paths, fps


def face_legend_entries() -> list[tuple[str, tuple[int, int, int]]]:
    """Counts come from ACTIVE, so the legend cannot claim points the model no longer
    predicts - the ear row reads 6, not 14."""
    counts: dict[str, int] = {}
    for d in ACTIVE:
        counts[REGION_OF[d]] = counts.get(REGION_OF[d], 0) + 1
    entries = [(f"{r} ({counts[r]})", REGION_COLOR[r])
               for r in ("ear", "head", "eye", "nose", "muzzle", "mouth")
               if counts.get(r)]
    return entries + [("body (39)", SA_COLOR)]


def clip_features(results: list[FrameResult], bodies: list, faces: list,
                  sa_names: list[str], fps: float, pcut: float
                  ) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Per-frame feature rows, plus the one-row clip summary `aggregate()` produces.

    Called with the keypoints as they are AFTER outlier rejection and BEFORE the
    rendering median. That ordering is load-bearing, not incidental:

    * `aggregate.oscillation` deliberately reads the RAW series, because smoothing is a
      low-pass filter and a wag or a pant is exactly the high-frequency content it
      removes. Outlier rejection is what protects that stage, so RANSAC belongs upstream
      of here and the median does not.
    * `aggregate()` then smooths again at its own fps-aware window. Handing it a series
      that has already been median-filtered low-passes it twice and quietly flattens
      every rate and oscillation feature - with nothing in the output to say so.

    The rolling median in `main()` exists to stop the drawn skeleton jittering. It is a
    rendering choice, and rendering choices must not reach the measurements.
    """
    rows = [
        frame_features(bodies[i], faces[i],
                       r.face_box.side if r.face_box is not None else None,
                       sa_names, ACTIVE_NAMES, pcut)
        for i, r in enumerate(results)
    ]

    # motion_energy wants whole arrays and a scale per frame, so it runs after the
    # per-frame pass that produced body_L. Frames with no dog stay NaN rather than 0:
    # a still dog and an undetected dog are different facts.
    n = len(results)
    body_arr = np.full((n, len(sa_names), 3), np.nan)
    face_arr = np.full((n, len(ACTIVE_NAMES), 3), np.nan)
    for i in range(n):
        if bodies[i] is not None:
            body_arr[i] = bodies[i]
        if faces[i] is not None:
            face_arr[i] = faces[i]
    lengths = np.array([r.get("body_L", np.nan) for r in rows], dtype=float)
    sides = np.array([r.face_box.side if r.face_box is not None else np.nan
                      for r in results], dtype=float)
    for i, energy in enumerate(motion_energy(body_arr, face_arr, lengths, sides,
                                             fps, pcut)):
        rows[i]["motion_energy"] = float(energy)

    return rows, aggregate(rows, fps, noise_floor=load_noise_floors())


def _fmt(v: float | None) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:.3f}"


def print_feature_summary(clip: dict[str, float], cols: list[str]) -> None:
    """Coverage first, because a feature's level is meaningless without it.

    A row at 4% coverage is not a measurement of the dog, it is a measurement of three
    lucky frames, and printing its level beside a fully-covered one invites the two to
    be read as equals.
    """
    print(f"\n{'=' * 72}\nFEATURES\n{'=' * 72}")
    print(f"{'feature':<26} {'coverage':>9} {'level':>9} {'spread':>9} {'rate/s':>9}")
    for c in cols:
        cov = clip.get(f"{c}__coverage")
        if cov is None:
            continue
        flag = "" if cov > 0.5 else ("   <- unusable" if cov < 0.05 else "   <- sparse")
        print(f"{c:<26} {cov * 100:8.0f}% {_fmt(clip.get(f'{c}__level')):>9} "
              f"{_fmt(clip.get(f'{c}__spread')):>9} "
              f"{_fmt(clip.get(f'{c}__rate')):>9}{flag}")
    for tag in ("wag", "mouth"):
        if f"{tag}_rate_hz" in clip:
            print(f"{tag + '_rate_hz':<26} {'':>9} "
                  f"{_fmt(clip.get(f'{tag}_rate_hz')):>9}     "
                  f"amp {_fmt(clip.get(f'{tag}_amp'))}")


def write_features(rows: list[dict[str, float]], clip: dict[str, float],
                   cols: list[str], csv_path: Path, json_path: Path,
                   video: str, pcut: float) -> None:
    """Per-frame CSV for inspection, one-row JSON for a classifier to consume."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame"] + cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"frame": i, **{c: r.get(c, "") for c in cols}})
    json_path.write_text(
        json.dumps({"video": video, "feature_pcut": pcut, **clip}, indent=2),
        encoding="utf-8")
    print(f"wrote {csv_path}  ({len(rows)} frames x {len(cols)} features)")
    print(f"wrote {json_path}  ({len(clip)} clip-level columns)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/cascade.mp4")
    ap.add_argument("--config", default=None,
                    help="defaults to the pytorch_config.yaml beside --snapshot")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--smooth-ms", type=float, default=100.0,
                    help="temporal median window in MILLISECONDS, converted with the "
                         "clip's own fps. 0 shows raw per-frame output")
    ap.add_argument("--smooth", type=int, default=None,
                    help="override --smooth-ms with an explicit frame count")
    ap.add_argument("--ransac", action="store_true",
                    help="reject keypoints inconsistent with short-term motion")
    ap.add_argument("--ransac-ms", type=float, default=230.0,
                    help="RANSAC fit window in milliseconds")
    ap.add_argument("--ransac-thresh", type=float, default=0.05,
                    help="inlier tolerance as a fraction of the face box side")
    ap.add_argument("--ransac-degree", type=int, default=1, choices=(1, 2),
                    help="1 = constant velocity, 2 = constant acceleration")
    ap.add_argument("--ransac-repair", default="model", choices=("model", "drop"),
                    help="replace an outlier with the fitted value, or zero its score")
    ap.add_argument("--pcut", type=float, default=PostConfig.min_confidence)
    ap.add_argument("--pad", type=float, default=FaceBoxConfig.pad)
    ap.add_argument("--no-lines", action="store_true",
                    help="points only; contours exaggerate a single bad landmark")
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--no-body", action="store_true", help="draw the face only")
    ap.add_argument("--ear-correct", action="store_true",
                    help="quarantined: measure on val before trusting it")
    ap.add_argument("--shape-refine", action="store_true",
                    help="quarantined: measure on val before trusting it")
    ap.add_argument("--no-subpixel", action="store_true")
    ap.add_argument("--features", action="store_true",
                    help="also compute per-frame features and the clip summary row")
    ap.add_argument("--features-out", default=None,
                    help="per-frame CSV; defaults to <out>_features.csv")
    ap.add_argument("--clip-out", default=None,
                    help="clip summary JSON; defaults to <out>_clip.json")
    ap.add_argument("--feature-pcut", type=float, default=None,
                    help="confidence cut for FEATURES only; defaults to --pcut. Kept "
                         "separate because --pcut is a display control, and raising it "
                         "to tidy up a video must not silently change a measurement")
    a = ap.parse_args()

    cfg = CascadeConfig(
        facebox=FaceBoxConfig(pad=a.pad),
        post=PostConfig(ear_correct=a.ear_correct, shape_refine=a.shape_refine,
                        subpixel=not a.no_subpixel, min_confidence=a.pcut),
    )

    scratch = Path(tempfile.gettempdir()) / "dlc_cascade_frames"
    paths, fps = extract_frames(a.video, scratch, a.width, a.start_frame, a.max_frames)
    print(f"{len(paths)} frames at {fps:.1f} fps", flush=True)

    cascade = Cascade(a.config or config_beside(a.snapshot), a.snapshot, cfg=cfg)
    if cascade.ear_corrector is not None:
        print("[post] ear correction ON (quarantined - measure it on val)", flush=True)
    if cascade.refiner is not None:
        print("[post] shape refine ON (quarantined - measure it on val)", flush=True)

    t0 = time.time()
    boxes, poses = cascade.stage1(paths)
    print(f"stage 1 (detector + stock SuperAnimal): {time.time() - t0:.1f}s", flush=True)

    results: list[FrameResult] = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        fb = cascade.face_box_for(poses[i], boxes[i])
        # stage2 applies whatever post-processing the config enables, so evaluation
        # and rendering cannot drift apart.
        face = cascade.stage2(img, fb) if fb is not None else None
        results.append(FrameResult(boxes[i], poses[i], fb, face))
        if (i + 1) % 25 == 0:
            print(f"  stage 2: {i + 1}/{len(paths)}", flush=True)

    rep = cascade.report(results)
    print(f"face box: derived {rep['derived']}  fallback {rep['fallback']}  "
          f"failed {rep['failed']}  no dog {rep['no_dog']}", flush=True)

    faces = [r.face for r in results]
    bodies = [r.body for r in results]

    if a.ransac:
        # Scale the tolerance by the face box so the same threshold means the same
        # thing whether the dog is near or far.
        side = np.array([r.face_box.side if r.face_box is not None else np.nan
                         for r in results])
        win = window_frames(a.ransac_ms, fps, minimum=a.ransac_degree + 2)
        faces, mask = ransac_filter(
            faces, window=win, thresh=a.ransac_thresh, degree=a.ransac_degree,
            conf_min=a.pcut, scale=side, repair=a.ransac_repair)
        print(f"[ransac] window {win} frames ({a.ransac_ms:.0f} ms at {fps:.1f} fps), "
              f"tol {a.ransac_thresh:g} x face box, degree {a.ransac_degree}", flush=True)
        print(f"[ransac] {summarise(mask, ACTIVE_NAMES)}", flush=True)

    # Features come off the keypoints HERE - after outlier rejection, before the
    # rendering median. See clip_features() for why that ordering is not negotiable.
    feat_rows = clip_row = None
    fpcut = a.pcut if a.feature_pcut is None else a.feature_pcut
    if a.features:
        print(f"[features] confidence cut {fpcut:g}"
              f"{'' if a.feature_pcut is None else ' (--feature-pcut)'}", flush=True)
        feat_rows, clip_row = clip_features(results, bodies, faces,
                                            cascade.sa_bodyparts, fps, fpcut)

    # Reject before smoothing: a median over a window that still contains an outlier
    # spreads its error into the neighbours instead of removing it.
    k = a.smooth if a.smooth is not None else window_frames(a.smooth_ms, fps)
    if k > 1:
        print(f"[smooth] median over {k} frames "
              f"({k / fps * 1000:.0f} ms at {fps:.1f} fps)", flush=True)
    faces = rolling_median(faces, k)
    bodies = rolling_median(bodies, k)

    rend = Renderer(cascade.sa_bodyparts)
    first = cv2.imread(paths[0])
    h, w = first.shape[:2]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    drawn = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        rend.draw(img, body=None if a.no_body else bodies[i], face=faces[i],
                  pcut=a.pcut, lines=not a.no_lines)
        if faces[i] is not None:
            drawn.append(int((faces[i][:, 2] >= a.pcut).sum()))
        banner(img, "SuperAnimal body + DogFLW face cascade",
               sub=f"39 body + {len(ACTIVE)} face  |  conf >= {a.pcut}")
        if not a.no_legend:
            legend(img, face_legend_entries(), org=(18, h - 170))
        vw.write(img)
    vw.release()

    if drawn:
        print(f"DRAWN: {np.mean(drawn):.1f}/{len(ACTIVE)} face landmarks per frame "
              f"(min {min(drawn)}, max {max(drawn)}) at conf >= {a.pcut}")
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s total)")

    if feat_rows is not None:
        cols = feature_names(cascade.sa_bodyparts, ACTIVE_NAMES) + ["motion_energy"]
        stem = Path(a.out).with_suffix("")
        write_features(
            feat_rows, clip_row, cols,
            Path(a.features_out) if a.features_out else Path(f"{stem}_features.csv"),
            Path(a.clip_out) if a.clip_out else Path(f"{stem}_clip.json"),
            a.video, fpcut)
        print_feature_summary(clip_row, cols)


if __name__ == "__main__":
    main()
