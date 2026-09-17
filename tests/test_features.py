"""Per-frame feature extraction.

Two of these pin bugs that a synthetic-geometry smoke test caught and that no amount of
reading the code would have shown:

* `test_ground_direction_aligns_with_body_axis` - the ground line comes from an SVD,
  which returns +/-v with no preferred sign. Unpinned, head_tilt flips 180 degrees
  between frames for no physical reason, and differentiating that flip manufactures an
  enormous rate.
* `test_head_pitch_neutral_is_near_zero` - referenced to the raw body axis (which runs
  toward the TAIL) a neutral dog read 158 degrees, parked against the +/-180 wrap where
  every derivative goes discontinuous.

`test_lookup_is_by_name_not_index` is the invariant the module exists to protect: the
SuperAnimal channel order at runtime is not the order the fixtures list.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import superanimal as sa
from features import body_frame, frame_features, Points
from keypoint_scheme import ACTIVE_NAMES

SA_NAMES = sa.sa_bodyparts()

# Image coordinates, y DOWN. A dog standing level, facing +x, paws on the line y=400.
BODY_PTS = {
    "nose": (320, 180), "neck_end": (300, 190), "neck_base": (270, 200),
    "back_base": (260, 200), "back_middle": (200, 205), "back_end": (140, 210),
    "tail_base": (135, 210), "tail_end": (95, 150),          # tail raised
    "front_left_thai": (255, 300), "front_right_thai": (258, 300),
    "back_left_thai": (145, 300), "back_right_thai": (148, 300),
    "front_left_paw": (255, 400), "front_right_paw": (258, 400),
    "back_left_paw": (145, 400), "back_right_paw": (148, 400),
}

FACE_PTS = {
    "eye_right_medial": (330, 170), "eye_right_lateral": (318, 170),
    "eye_left_medial": (330, 190), "eye_left_lateral": (318, 190),
    "eye_right_upper_lid": (324, 166), "eye_right_lower_lid": (324, 174),
    "eye_left_upper_lid": (324, 186), "eye_left_lower_lid": (324, 194),
    "head_top_right": (300, 168), "head_top_left": (300, 192),
    "ear_right_tip": (280, 155), "ear_left_tip": (280, 205),
    "ear_right_inner_base": (300, 172), "ear_left_inner_base": (300, 188),
    "nose_top": (350, 180), "nose_bottom": (354, 180), "philtrum": (357, 180),
    "lip_upper_right": (356, 176), "lip_upper_left": (356, 184),
    "lip_lower_center": (357, 186), "chin_upper": (352, 188),
}


def make_body(**overrides: tuple[float, float]) -> np.ndarray:
    pts = dict(BODY_PTS) | overrides
    arr = np.zeros((len(SA_NAMES), 3))
    for n, (x, y) in pts.items():
        arr[SA_NAMES.index(n)] = [x, y, 0.95]
    return arr


def make_face(**overrides: tuple[float, float]) -> np.ndarray:
    pts = dict(FACE_PTS) | overrides
    arr = np.zeros((len(ACTIVE_NAMES), 3))
    for n, (x, y) in pts.items():
        if n in ACTIVE_NAMES:
            arr[ACTIVE_NAMES.index(n)] = [x, y, 0.95]
    return arr


def feats(body: np.ndarray | None = None, face: np.ndarray | None = None,
          side: float | None = 80.0) -> dict[str, float]:
    return frame_features(make_body() if body is None else body,
                          make_face() if face is None else face,
                          side, SA_NAMES, ACTIVE_NAMES)


# ------------------------------------------------------------------ the invariant

def test_lookup_is_by_name_not_index() -> None:
    """Permuting the channel order and the name list together must change nothing.

    At runtime SuperAnimal orders back_base 19, back_end 20, back_middle 21, and
    interleaves the rear legs. Any hard-coded index computes a different feature while
    still returning a plausible number.
    """
    body, face = make_body(), make_face()
    ref = frame_features(body, face, 80.0, SA_NAMES, ACTIVE_NAMES)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(SA_NAMES))
    shuffled_names = [SA_NAMES[i] for i in perm]
    shuffled_body = body[perm]

    got = frame_features(shuffled_body, face, 80.0, shuffled_names, ACTIVE_NAMES)
    for k in ref:
        a, b = ref[k], got[k]
        if math.isnan(a):
            assert math.isnan(b), f"{k}: {a} vs {b}"
        else:
            assert b == pytest.approx(a, abs=1e-9), f"{k}: {a} vs {b}"


# ------------------------------------------------------------------ regressions

def test_ground_direction_aligns_with_body_axis() -> None:
    bf = body_frame(Points(make_body(), SA_NAMES, 0.35))
    assert bf is not None and bf.ground_d is not None
    assert float(np.dot(bf.ground_d, bf.b)) >= 0.0


def test_ground_direction_is_stable_when_paws_drop_out() -> None:
    """Losing a paw must not flip the ground line and swing head_tilt by 180 degrees."""
    full = feats()
    body = make_body()
    body[SA_NAMES.index("back_right_paw"), 2] = 0.0      # below the confidence cut
    partial = frame_features(body, make_face(), 80.0, SA_NAMES, ACTIVE_NAMES)
    assert abs(full["head_tilt"] - partial["head_tilt"]) < 5.0


def test_head_frame_orientation_does_not_depend_on_head_top() -> None:
    """head_top's mean confidence is 0.39, so it drops out often. When the v-axis sign
    keyed off it, those frames kept R90's arbitrary sign and EVERY ear angle flipped
    180 degrees - on real video, alternating between -57 and +122 frame to frame."""
    ref = feats()
    face = make_face()
    for n in ("head_top_right", "head_top_left"):
        face[ACTIVE_NAMES.index(n), 2] = 0.0
    got = frame_features(make_body(), face, 80.0, SA_NAMES, ACTIVE_NAMES)
    for k in ("ear_angle_L_eyeframe", "ear_angle_R_eyeframe", "ear_height"):
        assert got[k] == pytest.approx(ref[k], abs=1e-6), (
            f"{k} moved to {got[k]} from {ref[k]} when head_top dropped out")


def test_head_frame_refuses_without_a_nose_reference() -> None:
    """No dependable orientation means v's sign is a coin flip. A missing feature beats
    one that may be mirrored."""
    face = make_face()
    for n in ("nose_top", "nose_bottom"):
        if n in ACTIVE_NAMES:
            face[ACTIVE_NAMES.index(n), 2] = 0.0
    out = frame_features(make_body(), face, 80.0, SA_NAMES, ACTIVE_NAMES)
    assert math.isnan(out["ear_angle_L_eyeframe"])
    assert math.isnan(out["ear_height"])


def test_head_tilt_level_dog_is_near_zero_not_180() -> None:
    """The ground direction is pinned to the body axis, which runs toward the TAIL.
    Referenced to it a level dog read +/-180 and alternated sign across the wrap."""
    # A coherent head-ON face: the eyes must be separated HORIZONTALLY for the eye line
    # to be parallel to the ground. The module-level fixture separates them in y (a
    # profile-ish layout), where 90 degrees is the geometrically correct answer.
    # The nose sits below the eyes so "up the skull" resolves up-image.
    body = make_body()
    face = make_face(eye_right_medial=(318, 180), eye_right_lateral=(306, 180),
                     eye_left_medial=(332, 180), eye_left_lateral=(344, 180),
                     nose_top=(325, 205), nose_bottom=(325, 210))
    tilt = frame_features(body, face, 80.0, SA_NAMES, ACTIVE_NAMES)["head_tilt"]
    assert abs(tilt) < 15.0, f"head_tilt {tilt} should be ~0 for a level head"


def test_head_pitch_neutral_is_near_zero() -> None:
    """Not 158 degrees, and nowhere near the +/-180 wrap."""
    assert abs(feats()["head_pitch"]) < 90.0


def test_head_pitch_sign_follows_the_nose() -> None:
    up = feats(body=make_body(nose=(320, 140)))["head_pitch"]
    down = feats(body=make_body(nose=(320, 260)))["head_pitch"]
    assert up > 0 > down, f"nose up {up}, nose down {down}"


# ------------------------------------------------------------------ missing data

def test_everything_is_nan_when_nothing_is_given() -> None:
    # Counts and flags are not measurements: 0 paws is true, and
    # head_tilt_used_ground=0 correctly records that no ground line was available.
    # Only quantities that could be mistaken for a reading must be NaN.
    NOT_MEASUREMENTS = {"n_paws", "tail_conf", "head_tilt_used_ground"}
    out = frame_features(None, None, None, SA_NAMES, ACTIVE_NAMES)
    for k, v in out.items():
        if k in NOT_MEASUREMENTS:
            continue
        assert math.isnan(v), f"{k} returned {v}, expected NaN"


def test_missing_tail_gives_nan_not_zero() -> None:
    """Zero tuck means the tail is touching the hips - maximum fear. A tail that is
    merely out of shot must never be written as that."""
    body = make_body()
    for n in ("tail_base", "tail_end"):
        body[SA_NAMES.index(n), 2] = 0.0
    out = frame_features(body, make_face(), 80.0, SA_NAMES, ACTIVE_NAMES)
    for k in ("tail_tuck", "tail_height", "tail_lateral", "tail_len"):
        assert math.isnan(out[k]), f"{k} returned {out[k]}"


def test_low_confidence_point_is_not_used() -> None:
    body = make_body()
    body[SA_NAMES.index("nose"), 2] = 0.10
    out = frame_features(body, make_face(), 80.0, SA_NAMES, ACTIVE_NAMES)
    assert math.isnan(out["head_height"])
    assert math.isnan(out["head_pitch"])


# ------------------------------------------------------------------ signs

def test_posture_is_negative_for_a_play_bow() -> None:
    """Front end down, rear end up."""
    body = make_body(front_left_thai=(255, 370), front_right_thai=(258, 370))
    assert frame_features(body, make_face(), 80.0, SA_NAMES, ACTIVE_NAMES)["posture"] < 0


def test_posture_is_positive_when_sitting() -> None:
    """Hips on the ground, front end up - the case a ratio would have divided by zero."""
    body = make_body(back_left_thai=(145, 398), back_right_thai=(148, 398))
    out = frame_features(body, make_face(), 80.0, SA_NAMES, ACTIVE_NAMES)
    assert out["posture"] > 0
    assert math.isfinite(out["posture"])


def test_level_dog_has_posture_near_zero() -> None:
    assert abs(feats()["posture"]) < 0.05


def test_tail_above_the_spine_is_positive() -> None:
    assert feats()["tail_height"] > 0
    low = feats(body=make_body(tail_end=(95, 290)))["tail_height"]
    assert low < 0


def test_both_ear_formulations_agree_on_a_symmetric_head() -> None:
    """Mirrored sign conventions must make left and right read alike on a symmetric
    dog - otherwise the left/right difference is pure convention error."""
    o = feats()
    assert o["ear_angle_L_eyeframe"] == pytest.approx(o["ear_angle_R_eyeframe"], abs=1e-6)
    assert o["ear_angle_L_basevec"] == pytest.approx(o["ear_angle_R_basevec"], abs=1e-6)


def test_scale_invariance() -> None:
    """Doubling every coordinate is the dog moving closer to the camera. Normalised
    features must not move; this is the property the whole cascade exists to preserve."""
    ref = feats()
    big = frame_features(make_body() * np.array([2, 2, 1]),
                         make_face() * np.array([2, 2, 1]),
                         160.0, SA_NAMES, ACTIVE_NAMES)
    for k in ("posture", "head_height", "tail_tuck", "tail_len",
              "mouth_open_jaw", "ear_height", "ear_closeness_iod", "view_yaw"):
        assert big[k] == pytest.approx(ref[k], rel=1e-6), k
