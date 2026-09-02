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
from cascade import Cascade, FrameResult
from draw import REGION_COLOR, SA_COLOR, Renderer, banner, legend
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from keypoint_scheme import REGION_OF


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


def temporal_median(seq: list[np.ndarray | None], k: int) -> list[np.ndarray | None]:
    """Cosmetic k-frame median over each channel. k=1 leaves the raw output alone."""
    if k <= 1:
        return seq
    out: list[np.ndarray | None] = []
    half = k // 2
    for i, cur in enumerate(seq):
        if cur is None:
            out.append(None)
            continue
        window = [s for s in seq[max(0, i - half):i + half + 1] if s is not None
                  and s.shape == cur.shape]
        out.append(np.median(np.stack(window), axis=0) if len(window) > 1 else cur)
    return out


def face_legend_entries() -> list[tuple[str, tuple[int, int, int]]]:
    counts: dict[str, int] = {}
    for d in range(46):
        counts[REGION_OF[d]] = counts.get(REGION_OF[d], 0) + 1
    entries = [(f"{r} ({counts[r]})", REGION_COLOR[r])
               for r in ("ear", "head", "eye", "nose", "muzzle", "mouth")]
    return entries + [("body (39)", SA_COLOR)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/cascade.mp4")
    ap.add_argument("--config", default="face_project/face1/pytorch_config.yaml")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--smooth", type=int, default=3,
                    help="temporal median window; 1 shows raw per-frame output")
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

    cascade = Cascade(a.config, a.snapshot, cfg=cfg)
    ear, refiner = _load_post(cfg)

    t0 = time.time()
    boxes, poses = cascade.stage1(paths)
    print(f"stage 1 (detector + stock SuperAnimal): {time.time() - t0:.1f}s", flush=True)

    results: list[FrameResult] = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        fb = cascade.face_box_for(poses[i], boxes[i])
        face = cascade.stage2(img, fb) if fb is not None else None
        if face is not None:
            if ear is not None:
                ear.apply(face)
            if refiner is not None:
                refiner.apply(face)
        results.append(FrameResult(boxes[i], poses[i], fb, face))
        if (i + 1) % 25 == 0:
            print(f"  stage 2: {i + 1}/{len(paths)}", flush=True)

    rep = cascade.report(results)
    print(f"face box: derived {rep['derived']}  fallback {rep['fallback']}  "
          f"failed {rep['failed']}  no dog {rep['no_dog']}", flush=True)

    faces = temporal_median([r.face for r in results], a.smooth)
    bodies = temporal_median([r.body for r in results], a.smooth)

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
               sub=f"39 body + 46 face  |  conf >= {a.pcut}")
        if not a.no_legend:
            legend(img, face_legend_entries(), org=(18, h - 170))
        vw.write(img)
    vw.release()

    if drawn:
        print(f"DRAWN: {np.mean(drawn):.1f}/46 face landmarks per frame "
              f"(min {min(drawn)}, max {max(drawn)}) at conf >= {a.pcut}")
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s total)")


def _load_post(cfg: CascadeConfig):
    """Load the quarantined corrections only when explicitly enabled."""
    ear = refiner = None
    if cfg.post.ear_correct:
        from ear_correct import EarCorrector
        ear = EarCorrector()
        print("[post] ear correction ON (quarantined - measure it on val)", flush=True)
    if cfg.post.shape_refine:
        from shape_refine import Refiner
        refiner = Refiner()
        print("[post] shape refine ON (quarantined - measure it on val)", flush=True)
    return ear, refiner


if __name__ == "__main__":
    main()
