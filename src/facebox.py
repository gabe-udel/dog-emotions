"""Derive a face box from SuperAnimal's body-pose output.

THE INVARIANT OF THIS PROJECT: this module is the only place a face box is produced,
and both the training data builder (`build_face_coco.py`) and video inference
(`cascade.py`) call `derive_face_box` with the same `FaceBoxConfig`. Training on
DogFLW's shipped face bounding boxes and inferring on derived ones would mean the
face model never sees, at training time, the box distribution it gets in production -
tightness, centring and aspect all differ. `tests/test_facebox.py` asserts that both
call sites resolve to this function.

The box is deliberately derived from the *predicted* pose rather than ground truth,
including at training time, so first-stage error is part of the training distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceconfig import FACE_ANCHORS, FaceBoxConfig


@dataclass(frozen=True)
class FaceBox:
    """A square face box in image coordinates.

    Not clipped to the image: a box may extend past an edge, and `crop.py` pads rather
    than clips so the aspect ratio - and therefore the coordinate transform - stays
    exact. Use `clipped_to` for drawing.
    """

    cx: float
    cy: float
    side: float
    source: str
    """'anchors' - derived from >= min_anchors pose keypoints.
    'fallback' - derived from the dog box because too few anchors were visible.
    """
    n_anchors: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        h = self.side / 2.0
        return (self.cx - h, self.cy - h, self.cx + h, self.cy + h)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, _, _ = self.xyxy
        return (x1, y1, self.side, self.side)

    def clipped_to(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Integer xyxy clipped into the image, for drawing only."""
        x1, y1, x2, y2 = self.xyxy
        return (max(0, int(round(x1))), max(0, int(round(y1))),
                min(width, int(round(x2))), min(height, int(round(y2))))


def anchor_indices(superanimal_bodyparts: list[str]) -> list[int]:
    """Positions of FACE_ANCHORS within SuperAnimal's 39-keypoint ordering.

    Raises rather than silently dropping: a missing anchor name means the bodypart list
    is not SuperAnimal-Quadruped's and every downstream box would be quietly wrong.
    """
    missing = [n for n in FACE_ANCHORS if n not in superanimal_bodyparts]
    if missing:
        raise ValueError(
            f"bodypart list is missing SuperAnimal face anchors {missing}; "
            f"got {len(superanimal_bodyparts)} bodyparts"
        )
    return [superanimal_bodyparts.index(n) for n in FACE_ANCHORS]


def derive_face_box(
    pose: np.ndarray,
    anchor_idx: list[int],
    cfg: FaceBoxConfig,
    dog_box_xywh: np.ndarray | None = None,
) -> FaceBox | None:
    """Face box from one animal's SuperAnimal pose.

    Args:
        pose: (39, 3) array of x, y, score in image coordinates.
        anchor_idx: output of `anchor_indices`, the positions of the 11 head anchors.
        cfg: geometry and thresholds. See FaceBoxConfig for how pad and anchor_conf
            were measured.
        dog_box_xywh: the detector box, needed only for the 'dog_box_upper' fallback.

    Returns:
        A FaceBox, or None when derivation fails and the configured fallback is 'skip'
        or is unavailable. Callers must handle None - that is the failure path the
        cascade reports a rate for.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"pose must be (n, 3); got {pose.shape}")

    a = pose[anchor_idx]
    ok = np.isfinite(a[:, :2]).all(axis=1) & (a[:, 2] >= cfg.anchor_conf)
    n_ok = int(ok.sum())

    if n_ok >= cfg.min_anchors:
        p = a[ok, :2]
        x1, x2 = float(p[:, 0].min()), float(p[:, 0].max())
        y1, y2 = float(p[:, 1].min()), float(p[:, 1].max())
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hull = max(x2 - x1, y2 - y1) if cfg.square else 0.0
        side = hull * cfg.pad
        if side >= cfg.min_side_px:
            return FaceBox(cx=cx, cy=cy, side=side, source="anchors", n_anchors=n_ok)

    return _fallback(cfg, dog_box_xywh, n_ok)


def _fallback(cfg: FaceBoxConfig, dog_box_xywh: np.ndarray | None,
              n_ok: int) -> FaceBox | None:
    """Face box when the anchors are unusable.

    Measured rate at the default anchor_conf=0.1: 3 of 4,335 DogFLW images. It is rarer
    than it looks because the threshold is low by design - but on video, where the dog
    may be turned away, it is the normal case rather than the exception, so it has to
    degrade gracefully instead of raising.
    """
    if cfg.fallback == "skip" or dog_box_xywh is None:
        return None
    if cfg.fallback != "dog_box_upper":
        raise ValueError(f"unknown fallback {cfg.fallback!r}")

    x, y, w, h = (float(v) for v in np.asarray(dog_box_xywh, dtype=float).reshape(4))
    if w <= 0 or h <= 0:
        return None
    # The head sits at the top of a standing dog's box. Take the upper band, square it
    # off on the smaller dimension so the crop is not wildly anisotropic.
    band = h * cfg.fallback_frac
    side = min(w, band)
    if side < cfg.min_side_px:
        return None
    return FaceBox(cx=x + w / 2.0, cy=y + side / 2.0, side=side,
                   source="fallback", n_anchors=n_ok)


def derive_many(
    poses: np.ndarray,
    anchor_idx: list[int],
    cfg: FaceBoxConfig,
    dog_boxes_xywh: np.ndarray | None = None,
) -> list[FaceBox | None]:
    """Vectorised-caller convenience: one FaceBox (or None) per row of `poses`."""
    out: list[FaceBox | None] = []
    for i in range(len(poses)):
        db = None if dog_boxes_xywh is None else dog_boxes_xywh[i]
        out.append(derive_face_box(poses[i], anchor_idx, cfg, db))
    return out


def failure_rate(boxes: list[FaceBox | None]) -> dict[str, float | int]:
    """Summarise how box derivation went over a dataset or a video."""
    n = len(boxes)
    none = sum(b is None for b in boxes)
    fb = sum(b is not None and b.source == "fallback" for b in boxes)
    return {
        "n": n,
        "derived": n - none - fb,
        "fallback": fb,
        "failed": none,
        "fallback_rate": fb / n if n else 0.0,
        "failure_rate": none / n if n else 0.0,
    }
