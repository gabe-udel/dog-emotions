"""Face-box derivation, including the invariant this whole refactor exists to enforce.

`test_training_and_inference_share_one_function` is the important one: it fails if
anyone reintroduces a second box definition for training. The old architecture's core
defect was that training crops came from DogFLW's shipped face boxes while inference
crops came from somewhere else entirely, so the model never saw its deployment
distribution.
"""
from __future__ import annotations

import numpy as np
import pytest

import facebox
from facebox import FaceBox, anchor_indices, derive_face_box, derive_many, failure_rate
from faceconfig import FACE_ANCHORS, FaceBoxConfig

SA_BODYPARTS = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base",
    "right_antler_end", "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base", "neck_end",
    "throat_base", "throat_end", "back_base", "back_middle", "back_end",
    "belly_bottom", "body_middle_right", "body_middle_left",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_thai", "back_left_knee", "back_left_paw",
    "back_right_thai", "back_right_knee", "back_right_paw",
    "tail_base", "tail_end",
]
CFG = FaceBoxConfig()


def _pose(conf: float = 0.9, spread: float = 40.0, cx: float = 100.0,
          cy: float = 100.0) -> np.ndarray:
    """A synthetic 39-keypoint pose with the 11 face anchors spread over a square."""
    p = np.zeros((39, 3))
    p[:, 2] = 0.0
    idx = anchor_indices(SA_BODYPARTS)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-spread / 2, spread / 2, size=(len(idx), 2))
    pts[0] = [-spread / 2, -spread / 2]
    pts[1] = [spread / 2, spread / 2]
    for k, i in enumerate(idx):
        p[i] = [cx + pts[k, 0], cy + pts[k, 1], conf]
    return p


# ------------------------------------------------------------------ the invariant

def test_training_and_inference_share_one_function() -> None:
    """Both call sites must resolve to facebox.derive_face_box - the same object."""
    import build_face_coco
    import cascade
    assert build_face_coco.derive_face_box is facebox.derive_face_box
    assert cascade.derive_face_box is facebox.derive_face_box


def test_only_one_module_defines_a_box() -> None:
    """No second implementation may reappear elsewhere in src/."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        p.name for p in src.glob("*.py")
        if p.name != "facebox.py" and "def derive_face_box" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"a second derive_face_box exists in {offenders}"


# ------------------------------------------------------------------ anchors

def test_anchor_indices_resolve() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    assert len(idx) == len(FACE_ANCHORS)
    assert [SA_BODYPARTS[i] for i in idx] == list(FACE_ANCHORS)


def test_anchor_indices_reject_wrong_bodypart_list() -> None:
    """A silently-wrong bodypart list would make every box wrong; fail loudly."""
    with pytest.raises(ValueError, match="missing SuperAnimal face anchors"):
        anchor_indices(["nose", "tail_base"])


def test_antlers_are_not_anchors() -> None:
    assert not any("antler" in a for a in FACE_ANCHORS)


# ------------------------------------------------------------------ geometry

def test_box_is_square_and_padded() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    box = derive_face_box(_pose(spread=40.0), idx, CFG)
    assert box is not None and box.source == "anchors"
    assert box.side == pytest.approx(40.0 * CFG.pad)
    x1, y1, x2, y2 = box.xyxy
    assert (x2 - x1) == pytest.approx(y2 - y1)


def test_box_is_centred_on_the_anchor_hull() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    box = derive_face_box(_pose(cx=250.0, cy=175.0), idx, CFG)
    assert box is not None
    assert box.cx == pytest.approx(250.0, abs=1e-9)
    assert box.cy == pytest.approx(175.0, abs=1e-9)


def test_low_confidence_anchors_are_excluded() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    p = _pose(conf=0.05)                       # below anchor_conf 0.1
    assert derive_face_box(p, idx, CFG, dog_box_xywh=None) is None


def test_falls_back_to_dog_box_when_anchors_missing() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    p = _pose(conf=0.0)
    box = derive_face_box(p, idx, CFG, dog_box_xywh=np.array([10.0, 20.0, 100.0, 200.0]))
    assert box is not None and box.source == "fallback"
    assert box.cx == pytest.approx(60.0)        # centred on the dog box horizontally
    assert box.side == pytest.approx(min(100.0, 200.0 * CFG.fallback_frac))


def test_skip_fallback_returns_none() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    cfg = FaceBoxConfig(fallback="skip")
    box = derive_face_box(_pose(conf=0.0), idx, cfg,
                          dog_box_xywh=np.array([0.0, 0.0, 100.0, 100.0]))
    assert box is None


def test_unknown_fallback_raises() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    cfg = FaceBoxConfig(fallback="teleport")
    with pytest.raises(ValueError, match="unknown fallback"):
        derive_face_box(_pose(conf=0.0), idx, cfg,
                        dog_box_xywh=np.array([0.0, 0.0, 100.0, 100.0]))


def test_degenerate_hull_is_refused() -> None:
    """All anchors on one point: side would be ~0 and the crop meaningless."""
    idx = anchor_indices(SA_BODYPARTS)
    p = np.zeros((39, 3))
    for i in idx:
        p[i] = [50.0, 50.0, 0.9]
    assert derive_face_box(p, idx, CFG, dog_box_xywh=None) is None


def test_nan_anchors_are_ignored_not_propagated() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    p = _pose()
    p[idx[0]] = [np.nan, np.nan, 0.9]
    box = derive_face_box(p, idx, CFG)
    assert box is not None and np.isfinite([box.cx, box.cy, box.side]).all()


def test_malformed_pose_raises() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    with pytest.raises(ValueError, match=r"pose must be"):
        derive_face_box(np.zeros((39,)), idx, CFG)


# ------------------------------------------------------------------ batch + reporting

def test_derive_many_and_failure_rate() -> None:
    idx = anchor_indices(SA_BODYPARTS)
    poses = np.stack([_pose(), _pose(conf=0.0), _pose()])
    boxes = derive_many(poses, idx, FaceBoxConfig(fallback="skip"))
    assert [b is None for b in boxes] == [False, True, False]
    r = failure_rate(boxes)
    assert r == {"n": 3, "derived": 2, "fallback": 0, "failed": 1,
                 "fallback_rate": 0.0, "failure_rate": pytest.approx(1 / 3)}


def test_clipped_to_stays_in_image() -> None:
    box = FaceBox(5.0, 5.0, 100.0, "anchors", 11)
    x1, y1, x2, y2 = box.clipped_to(320, 240)
    assert (x1, y1) == (0, 0)
    assert x2 <= 320 and y2 <= 240
