"""Why a feature came out the way it did.

`frames.json` shows that `tail_height` was NaN. It does not show that `tail_base` sat
at confidence 0.09 against a 0.35 cut, which is the thing you need before you can decide
whether to lower the cut, improve the model, or drop the feature.

This module joins the two: the raw keypoint confidences and the feature values for the
same frame, side by side, with the failures marked. It reads the cache and recomputes
nothing, so it is instant and cannot disagree with what the pipeline produced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OK, BAD, MISS = "ok", "below cut", "missing"


def load_keypoints(npz_path: str | Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    return {"body": z["body"], "face": z["face"],
            "sa_names": [str(n) for n in z["sa_names"]],
            "face_names": [str(n) for n in z["face_names"]]}


def _state(score: float, pcut: float) -> str:
    if not np.isfinite(score):
        return MISS
    return OK if score >= pcut else BAD


def keypoint_coverage(kp: dict, pcut: float) -> list[tuple[str, str, float, float]]:
    """(group, name, mean confidence, fraction of frames above the cut), worst first.

    This is the table that explains a coverage number. A feature is NaN unless every
    keypoint it needs clears the cut, so the weakest keypoint sets the ceiling for every
    feature that touches it.
    """
    rows = []
    for group, arr, names in (("body", kp["body"], kp["sa_names"]),
                              ("face", kp["face"], kp["face_names"])):
        for i, name in enumerate(names):
            s = arr[:, i, 2]
            s = s[np.isfinite(s)]
            if len(s) == 0:
                rows.append((group, name, float("nan"), 0.0))
                continue
            rows.append((group, name, float(s.mean()), float((s >= pcut).mean())))
    rows.sort(key=lambda r: r[3])
    return rows


def frame_trace(kp: dict, frames: list[dict], index: int, pcut: float) -> dict:
    """One frame: every keypoint's confidence, and every feature's value."""
    body, face = kp["body"][index], kp["face"][index]
    out = {"frame": index, "keypoints": [], "features": []}
    for group, arr, names in (("body", body, kp["sa_names"]),
                              ("face", face, kp["face_names"])):
        for i, name in enumerate(names):
            out["keypoints"].append({
                "group": group, "name": name, "score": float(arr[i, 2]),
                "state": _state(float(arr[i, 2]), pcut)})
    for name, value in frames[index].items():
        out["features"].append({
            "name": name, "value": float(value),
            "ok": bool(np.isfinite(value))})
    return out


def render_coverage(rows, pcut: float, limit: int = 18) -> str:
    out = [f"KEYPOINT RELIABILITY  (confidence cut {pcut:g}; weakest first)",
           f"{'':2}{'keypoint':<24}{'group':>6}{'mean conf':>11}{'% above cut':>13}",
           "-" * 56]
    for group, name, mean, frac in rows[:limit]:
        mark = "!!" if frac < 0.4 else ("! " if frac < 0.9 else "  ")
        mean_s = "   n/a" if not np.isfinite(mean) else f"{mean:9.3f}"
        out.append(f"{mark}{name:<24}{group:>6}{mean_s}{frac * 100:12.0f}%")
    if len(rows) > limit:
        strong = sum(1 for r in rows[limit:] if r[3] >= 0.9)
        out.append(f"  ... {len(rows) - limit} more, {strong} of them above 90%")
    out.append("")
    out.append("A feature is NaN unless EVERY keypoint it needs clears the cut, so the")
    out.append("weakest row above caps every feature that touches it.")
    return "\n".join(out)


def render_frame(trace: dict, pcut: float, only_failed: bool = False) -> str:
    out = [f"FRAME {trace['frame']}  (confidence cut {pcut:g})", ""]

    bad = [k for k in trace["keypoints"] if k["state"] != OK]
    out.append(f"keypoints below the cut: {len(bad)} of {len(trace['keypoints'])}")
    for k in bad:
        out.append(f"  !! {k['name']:<24} {k['group']:>5}  "
                   f"{k['score']:.3f}  {k['state']}")
    if not bad:
        out.append("  (none)")

    out.append("")
    nan = [f for f in trace["features"] if not f["ok"]]
    out.append(f"features that could not be computed: {len(nan)} of "
               f"{len(trace['features'])}")
    for f in nan:
        out.append(f"  !! {f['name']}")

    if not only_failed:
        out.append("")
        out.append("computed values:")
        for f in trace["features"]:
            if f["ok"]:
                out.append(f"     {f['name']:<26} {f['value']:+.4f}")
    return "\n".join(out)
