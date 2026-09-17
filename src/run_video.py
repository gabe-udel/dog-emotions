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
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import superanimal as sa
from cascade import Cascade, FrameResult, config_beside
from draw import REGION_COLOR, SA_COLOR, Renderer, banner, legend
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
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


if __name__ == "__main__":
    main()
