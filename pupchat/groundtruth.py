"""Proving accuracy on YOUR footage, not on DogFLW.

Everything the project knows about accuracy was measured on still images from DogFLW and
StanfordExtra. That is the right baseline and it does not transfer: the pom clip showed
`mouth_open_jaw` at 94% coverage on DogFLW stills and 10% on one video. A number
measured on someone else's photographs is not a claim about your camera, your breed,
your lighting or your framing.

The only way to make that claim is the way the project already does it elsewhere - label
ground truth, difference it against the prediction, and report the spread:

    1. sample frames from your clip
    2. click the keypoints by hand           -> label_frames()
    3. difference predicted against clicked  -> score_keypoints()
    4. recompute features from BOTH          -> score_features()

Step 4 is the part that matters and the part people skip. A keypoint error of 3 px means
nothing until you know what it does to `ear_angle`, and the answer is not proportional:
features built as a difference between two nearby landmarks cancel most of a shared
error, and features built as a ratio can blow up when the denominator is small.

Two rules this tool follows, and they decide whether the result means anything:

**The model's prediction is never drawn while you label.** Showing the guess for the
point being labelled anchors the click to it, and the run then measures agreement rather
than accuracy. `src/label_withers.py` follows the same rule for the same reason.

**Frames are sampled evenly, not chosen.** Picking the frames where the dog looks clear
measures the easy cases, which is exactly the bias that makes a number flattering.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LABELS_NAME = "ground_truth.json"


# ------------------------------------------------------------------ sampling

def sample_indices(n_frames: int, k: int, seed: int = 0) -> list[int]:
    """Evenly spaced frames. Deterministic, so a second labelling session covers the
    same frames and the two can be compared."""
    if k >= n_frames:
        return list(range(n_frames))
    return [int(round(i)) for i in np.linspace(0, n_frames - 1, k)]


# ------------------------------------------------------------------ scoring

def score_keypoints(labelled: dict[str, dict[str, list[float]]],
                    predicted: np.ndarray, names: list[str], scores: np.ndarray,
                    frame_index: dict[str, int], normaliser: dict[str, float],
                    pcut: float = 0.35) -> dict:
    """Per-keypoint error against hand labels, normalised the way the project does.

    `normaliser` is per frame - the face-box or dog-box diagonal - so the error is a
    fraction of the subject's size and a 10 px miss on a small dog is not counted the
    same as on a large one. Same construction as evaluate_face and check_body_accuracy,
    so the numbers can sit beside theirs.
    """
    per_kp: dict[str, list[float]] = {n: [] for n in names}
    skipped_low_conf = 0

    for key, points in labelled.items():
        i = frame_index.get(key)
        if i is None:
            continue
        diag = normaliser.get(key)
        if not diag or not math.isfinite(diag) or diag <= 0:
            continue
        for name, xy in points.items():
            if name not in per_kp or xy is None:
                continue
            j = names.index(name)
            if scores[i, j] < pcut:
                skipped_low_conf += 1
                continue
            p = predicted[i, j, :2]
            if not np.isfinite(p).all():
                continue
            per_kp[name].append(float(np.linalg.norm(p - np.asarray(xy, float)) / diag))

    rows = {}
    for name, errs in per_kp.items():
        if not errs:
            continue
        e = np.asarray(errs)
        rows[name] = {
            "n": int(len(e)),
            "nme": float(e.mean()),
            "median": float(np.median(e)),
            "pck5": float((e < 0.05).mean()),
            "pck10": float((e < 0.10).mean()),
        }
    pooled = np.concatenate([np.asarray(v) for v in per_kp.values() if v]) \
        if any(per_kp.values()) else np.array([])
    return {
        "per_keypoint": dict(sorted(rows.items(), key=lambda kv: -kv[1]["nme"])),
        "overall": ({"n": int(pooled.size), "nme": float(pooled.mean()),
                     "median": float(np.median(pooled)),
                     "pck5": float((pooled < 0.05).mean()),
                     "pck10": float((pooled < 0.10).mean())}
                    if pooled.size else {}),
        "skipped_below_confidence_cut": skipped_low_conf,
        "normaliser": "per-frame box diagonal",
        "note": "measured on this footage, not on DogFLW",
    }


def score_features(pred_rows: list[dict], truth_rows: list[dict],
                   angular: set[str]) -> dict:
    """Feature error: the same feature computed from predicted and from labelled
    keypoints, differenced.

    This is the number that matters, and it is not a rescaling of the keypoint error.
    A difference feature cancels most of a shared shift; a ratio feature can blow up.
    Identical method to src/check_feature_error.py, applied to your clip instead of
    DogFLW's validation split.
    """
    if not pred_rows or not truth_rows:
        return {}
    out = {}
    for name in truth_rows[0]:
        d = []
        for p, t in zip(pred_rows, truth_rows):
            a, b = p.get(name), t.get(name)
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            diff = a - b
            if name in angular:
                diff = (diff + 180.0) % 360.0 - 180.0
            d.append(diff)
        if len(d) < 3:
            continue
        arr = np.asarray(d)
        out[name] = {
            "n": int(arr.size),
            "error_sd": float(arr.std()),
            "bias": float(arr.mean()),
            "max_abs": float(np.max(np.abs(arr))),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["error_sd"]))


def compare_to_dogflw(local: dict, reference: dict) -> list[dict]:
    """Your error against the DogFLW-measured error, per feature.

    A ratio far above 1 means the published figure does not describe your footage, and
    whatever was built on it needs re-checking against the local number.
    """
    rows = []
    for name, v in local.items():
        ref = reference.get(name, {})
        ref_sd = ref.get("error_sd")
        rows.append({
            "feature": name,
            "local_error_sd": round(v["error_sd"], 5),
            "dogflw_error_sd": round(ref_sd, 5) if ref_sd else None,
            "ratio": (round(v["error_sd"] / ref_sd, 2)
                      if ref_sd and ref_sd > 1e-12 else None),
            "local_bias": round(v["bias"], 5),
            "n": v["n"],
        })
    rows.sort(key=lambda r: -(r["ratio"] or 0))
    return rows


# ------------------------------------------------------------------ storage

def save_labels(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_labels(path: Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def labelling_progress(payload: dict, wanted: list[int], per_frame: int) -> dict:
    """How much of the job is done, so a session can be stopped and resumed."""
    done = payload.get("frames", {})
    complete = sum(1 for k in done if len(done[k]) >= per_frame)
    return {
        "frames_labelled": len(done),
        "frames_complete": complete,
        "frames_wanted": len(wanted),
        "points_placed": sum(len(v) for v in done.values()),
        "fraction": round(len(done) / max(1, len(wanted)), 3),
    }
