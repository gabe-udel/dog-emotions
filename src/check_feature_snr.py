"""Experiment C: which features carry more signal than measurement noise?

A feature is worth keeping only if it moves more when the DOG changes than when the
KEYPOINTS are wrong. This measures both and reports the ratio, so the list is pruned on
evidence rather than on how good the behavioural argument sounds.

Three things the first version of this script got wrong, each of which inflated every
score into the thousands:

**Noise came from temporal residuals.** Fitting a local quadratic and taking the residual
measures JITTER, not ERROR. A landmark sitting 20 px wrong in the same direction for
thirty frames fits a smooth curve perfectly and reports ~1 px of noise. Measured against
the val-set NME the true figure was ~17x larger. Face landmarks now take their sigma from
the measured per-landmark NME instead.

**Observed spread was pooled across clips.** Four clips means four dogs at four distances
through two cameras. That variance is real but the dog's state did not cause it, so it is
not signal. Spread is now measured within each clip, then pooled.

**Angles were treated as linear.** A feature resting near +/-180 wraps, and a plain
standard deviation reports a huge spread for a perfectly steady value - which is exactly
how head_tilt scored as the third-strongest feature while being constant. Angular
features now use circular statistics.

    python src/check_feature_snr.py --video "Happy lab.mov" --snapshot <ckpt>
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

import superanimal as sa
from cascade import Cascade, config_beside
from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
from features import feature_names, frame_features
from keypoint_scheme import ACTIVE_NAMES
from run_video import extract_frames

# Features measured in degrees, which need circular statistics.
ANGULAR = {
    "ear_angle_L_eyeframe", "ear_angle_R_eyeframe",
    "ear_angle_L_basevec", "ear_angle_R_basevec",
    "tail_height", "head_tilt", "head_pitch",
}

# Bookkeeping columns, not measurements - excluded from the verdict.
NOT_FEATURES = {"tail_conf", "head_tilt_used_ground", "face_px", "body_L", "n_paws"}

# A feature swinging further than the anatomy can is unstable, not informative, and raw
# variance would happily score that instability as signal.
IMPLAUSIBLE_ANGLE = 60.0
IMPLAUSIBLE = {"tail_tuck": 1.0, "posture": 1.0, "head_height": 1.5,
               "ear_height": 1.0, "tail_len": 1.0}

# DogFLW's NME normaliser is its annotated face-box diagonal. Our derived box has
# pad=1.8 and leaves the face filling ~67% of it, so that diagonal is about
# 0.67 * S * sqrt(2) ~= 0.95 * S. An approximation, and flagged as one.
FACE_DIAG_PER_S = 0.95


def circular_sd(deg: np.ndarray) -> float:
    """Circular standard deviation in degrees. Steady-near-180 reads as ~0, not ~145."""
    r = np.exp(1j * np.radians(deg))
    R = float(np.abs(r.mean()))
    if R <= 1e-12:
        return 180.0
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R)))))


def local_noise(series: np.ndarray, window: int = 9, degree: int = 2) -> np.ndarray:
    """Per-keypoint JITTER from residuals to a local polynomial fit.

    Precision, not accuracy: it cannot see an error that persists across the window.
    Used only for body keypoints, which have no ground truth on this project, and then
    inflated by the ratio the face landmarks reveal between jitter and true error.
    """
    t, k, _ = series.shape
    half = window // 2
    resid: list[list[float]] = [[] for _ in range(k)]
    for c in range(half, t - half):
        idx = np.arange(c - half, c + half + 1)
        tt = (idx - c).astype(float)
        for j in range(k):
            seg = series[idx, j, :2]
            ok = np.isfinite(seg).all(axis=1) & (series[idx, j, 2] > 0)
            if ok.sum() < degree + 2:
                continue
            A = np.vander(tt[ok], degree + 1)
            coef, *_ = np.linalg.lstsq(A, seg[ok], rcond=None)
            pred = (np.vander(np.array([0.0]), degree + 1) @ coef)[0]
            if series[c, j, 2] > 0 and np.isfinite(series[c, j, :2]).all():
                resid[j].append(float(np.linalg.norm(series[c, j, :2] - pred)))

    out = np.full(k, np.nan)
    for j in range(k):
        if len(resid[j]) >= 5:
            r = np.asarray(resid[j])
            out[j] = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
    return out


def face_sigma_px(nme_by_name: dict[str, float], S: float) -> np.ndarray:
    """Per-face-landmark sigma in pixels, from the measured val-set NME."""
    diag = FACE_DIAG_PER_S * S
    return np.array([nme_by_name.get(n, np.nan) * diag for n in ACTIVE_NAMES])


def run_clip(cascade: Cascade, video: str, max_frames: int, width: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scratch = Path(tempfile.gettempdir()) / "dlc_snr_frames"
    paths, fps = extract_frames(video, scratch, width, 0, max_frames)
    boxes, poses = cascade.stage1(paths)
    bodies = np.full((len(paths), 39, 3), np.nan)
    faces = np.full((len(paths), len(ACTIVE_NAMES), 3), np.nan)
    sides = np.full(len(paths), np.nan)
    for i, p in enumerate(paths):
        if poses[i] is not None:
            bodies[i] = poses[i]
        fb = cascade.face_box_for(poses[i], boxes[i])
        if fb is not None:
            sides[i] = fb.side
            faces[i] = cascade.stage2(cv2.imread(p), fb)
        if (i + 1) % 40 == 0:
            print(f"    {i + 1}/{len(paths)}", flush=True)
    return bodies, faces, sides, fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-json", default="outputs/face38_eval_val.json",
                    help="source of the measured per-landmark NME")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--pcut", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/feature_snr.json")
    a = ap.parse_args()

    nme = json.loads(Path(a.eval_json).read_text())["per_landmark"]
    cfg = CascadeConfig(facebox=FaceBoxConfig(),
                        post=PostConfig(ear_correct=False, shape_refine=False,
                                        min_confidence=a.pcut))
    cascade = Cascade(a.config or config_beside(a.snapshot), a.snapshot, cfg=cfg)
    SA = sa.sa_bodyparts()
    rng = np.random.default_rng(a.seed)
    keys = feature_names(SA, ACTIVE_NAMES)

    clip_rows: list[list[dict]] = []
    noise_sd: dict[str, list[float]] = {}
    per_clip_meta: dict[str, dict] = {}

    for video in a.video:
        print(f"\n[{Path(video).name}]", flush=True)
        bodies, faces, sides, fps = run_clip(cascade, video, a.max_frames, a.width)
        med_S = float(np.nanmedian(sides))

        jitter_body = local_noise(bodies)
        jitter_face = local_noise(faces)
        sig_face_ref = face_sigma_px(nme, med_S)
        # How far the jitter estimate falls short of true error, measured on the face
        # where both numbers exist, then applied to the body where only jitter does.
        ratio = np.nanmedian(sig_face_ref / np.where(jitter_face > 0, jitter_face, np.nan))
        ratio = float(ratio) if np.isfinite(ratio) and ratio > 1 else 1.0
        sigma_body = jitter_body * ratio

        per_clip_meta[Path(video).name] = {
            "fps": fps, "face_box_px_median": med_S,
            "face_jitter_px_median": float(np.nanmedian(jitter_face)),
            "face_sigma_px_median": float(np.nanmedian(sig_face_ref)),
            "jitter_to_error_ratio": ratio,
            "body_jitter_px_median": float(np.nanmedian(jitter_body)),
            "body_sigma_px_median_INFERRED": float(np.nanmedian(sigma_body)),
        }
        print(f"  face: jitter {np.nanmedian(jitter_face):.2f}px vs measured error "
              f"{np.nanmedian(sig_face_ref):.2f}px  -> x{ratio:.1f}")
        print(f"  body: jitter {np.nanmedian(jitter_body):.2f}px -> inferred "
              f"{np.nanmedian(sigma_body):.2f}px (accuracy never measured)")

        rows = []
        for i in range(len(bodies)):
            rows.append(frame_features(bodies[i], faces[i], sides[i], SA,
                                       ACTIVE_NAMES, a.pcut))
            if not np.isfinite(sides[i]):
                continue
            sigma_face = face_sigma_px(nme, float(sides[i]))
            draws: dict[str, list[float]] = {k: [] for k in keys}
            for _ in range(a.draws):
                b2, f2 = bodies[i].copy(), faces[i].copy()
                for j in range(b2.shape[0]):
                    if np.isfinite(sigma_body[j]):
                        b2[j, :2] += rng.normal(0.0, sigma_body[j], size=2)
                for j in range(f2.shape[0]):
                    if np.isfinite(sigma_face[j]):
                        f2[j, :2] += rng.normal(0.0, sigma_face[j], size=2)
                for k, v in frame_features(b2, f2, sides[i], SA, ACTIVE_NAMES,
                                           a.pcut).items():
                    draws[k].append(v)
            for k, vals in draws.items():
                v = np.asarray(vals, dtype=float)
                v = v[np.isfinite(v)]
                if len(v) >= max(5, a.draws // 4):
                    noise_sd.setdefault(k, []).append(
                        circular_sd(v) if k in ANGULAR else float(v.std()))
        clip_rows.append(rows)

    # ------------------------------------------------------------------ summarise
    n_frames = sum(len(r) for r in clip_rows)
    summary: dict[str, dict] = {}
    for k in keys:
        flat = np.asarray([r[k] for rows in clip_rows for r in rows], dtype=float)
        cov = float(np.isfinite(flat).mean())
        ns = np.asarray(noise_sd.get(k, []), dtype=float)
        ns = ns[np.isfinite(ns)]

        per_clip = []
        for rows in clip_rows:
            v = np.asarray([r[k] for r in rows], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 10:
                per_clip.append(circular_sd(v) if k in ANGULAR else float(v.std()))
        if not per_clip or len(ns) == 0:
            summary[k] = {"coverage": cov, "observed_sd": float("nan"),
                          "noise_sd": float("nan"), "snr": float("nan"),
                          "implausible": False}
            continue

        o_sd = float(np.median(per_clip))
        n_sd = float(np.median(ns))
        snr = max(0.0, o_sd ** 2 - n_sd ** 2) / (n_sd ** 2) if n_sd > 1e-12 else float("nan")
        limit = IMPLAUSIBLE.get(k, IMPLAUSIBLE_ANGLE if k in ANGULAR else None)
        summary[k] = {"coverage": cov, "observed_sd": o_sd, "noise_sd": n_sd,
                      "snr": snr, "implausible": bool(limit and o_sd > limit)}

    out = {"videos": a.video, "draws": a.draws, "n_frames": n_frames, "pcut": a.pcut,
           "clips": per_clip_meta, "features": summary}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"\n{'='*82}\nFEATURE SIGNAL-TO-NOISE  ({n_frames} frames, {a.draws} draws)"
          f"\n{'='*82}")
    print(f"{'feature':26s} {'cover':>6s} {'obs sd':>9s} {'noise sd':>9s} {'SNR':>7s}  verdict")
    rank = sorted(((k, v) for k, v in summary.items() if k not in NOT_FEATURES),
                  key=lambda kv: (np.isnan(kv[1]["snr"]), -(kv[1]["snr"] or 0)))
    for k, v in rank:
        snr = v["snr"]
        if np.isnan(snr):
            verdict = "NO DATA"
        elif v["implausible"]:
            verdict = "UNSTABLE - spread exceeds anatomy"
        elif snr >= 4:
            verdict = "strong"
        elif snr >= 1:
            verdict = "usable"
        elif snr >= 0.25:
            verdict = "marginal"
        else:
            verdict = "DROP - noise-dominated"
        print(f"  {k:24s} {v['coverage']*100:5.0f}% {v['observed_sd']:9.4f} "
              f"{v['noise_sd']:9.4f} {snr:7.2f}  {verdict}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
