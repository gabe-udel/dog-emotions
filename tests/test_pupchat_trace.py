"""The trace layer: joining keypoint confidences to feature failures.

`frames.json` shows that a feature was NaN. These functions show WHICH keypoint fell
below the confidence cut, which is the only form of that information anyone can act on.
Pure numpy over the cache - no model, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

from pupchat.trace import (BAD, MISS, OK, frame_trace, keypoint_coverage,
                           render_coverage, render_frame)

PCUT = 0.35
SA = ["nose", "tail_base", "tail_end", "front_left_paw"]
FACE = ["ear_left_tip", "ear_right_tip", "philtrum"]
N = 10


@pytest.fixture
def kp():
    body = np.zeros((N, len(SA), 3))
    face = np.zeros((N, len(FACE), 3))
    body[:, 0, 2] = 0.95                 # nose: always good
    body[:, 1, 2] = 0.10                 # tail_base: always below the cut
    body[:, 2, 2] = 0.08                 # tail_end: always below
    body[:, 3, 2] = 0.90                 # front_left_paw: good
    body[3, 3, 2] = 0.05                 # ... except on one frame
    face[:, 0, 2] = 0.80
    face[:, 1, 2] = 0.80
    face[:, 2, 2] = 0.70
    return {"body": body, "face": face, "sa_names": SA, "face_names": FACE}


@pytest.fixture
def frames():
    return [{"tail_height": float("nan"), "ear_closeness_iod": 1.5,
             "mouth_open_jaw": 0.2} for _ in range(N)]


def test_coverage_is_sorted_weakest_first(kp):
    rows = keypoint_coverage(kp, PCUT)
    fracs = [r[3] for r in rows]
    assert fracs == sorted(fracs)
    assert rows[0][1] in {"tail_base", "tail_end"}


def test_coverage_reports_the_fraction_above_the_cut(kp):
    by_name = {r[1]: r for r in keypoint_coverage(kp, PCUT)}
    assert by_name["nose"][3] == 1.0
    assert by_name["tail_base"][3] == 0.0
    assert by_name["front_left_paw"][3] == pytest.approx(0.9)


def test_coverage_covers_body_and_face_together(kp):
    rows = keypoint_coverage(kp, PCUT)
    assert len(rows) == len(SA) + len(FACE)
    assert {r[0] for r in rows} == {"body", "face"}


def test_a_keypoint_with_no_finite_score_is_not_silently_perfect():
    kp = {"body": np.full((3, 1, 3), np.nan), "face": np.zeros((3, 0, 3)),
          "sa_names": ["ghost"], "face_names": []}
    rows = keypoint_coverage(kp, PCUT)
    assert rows[0][3] == 0.0
    assert not np.isfinite(rows[0][2])


def test_frame_trace_marks_each_keypoint_state(kp, frames):
    t = frame_trace(kp, frames, 3, PCUT)
    states = {k["name"]: k["state"] for k in t["keypoints"]}
    assert states["nose"] == OK
    assert states["tail_base"] == BAD
    assert states["front_left_paw"] == BAD          # the one bad frame
    t0 = frame_trace(kp, frames, 0, PCUT)
    assert {k["name"]: k["state"] for k in t0["keypoints"]}["front_left_paw"] == OK


def test_frame_trace_marks_nan_features(kp, frames):
    t = frame_trace(kp, frames, 0, PCUT)
    ok = {f["name"]: f["ok"] for f in t["features"]}
    assert ok["tail_height"] is False
    assert ok["ear_closeness_iod"] is True


def test_missing_is_distinct_from_below_cut():
    """A keypoint the model never emitted is a different fact from one it emitted with
    low confidence, and the trace must not merge them."""
    kp = {"body": np.array([[[0, 0, np.nan], [0, 0, 0.1]]]),
          "face": np.zeros((1, 0, 3)),
          "sa_names": ["absent", "weak"], "face_names": []}
    states = {k["name"]: k["state"]
              for k in frame_trace(kp, [{}], 0, PCUT)["keypoints"]}
    assert states["absent"] == MISS
    assert states["weak"] == BAD


def test_render_coverage_flags_the_weak_ones(kp):
    text = render_coverage(keypoint_coverage(kp, PCUT), PCUT)
    assert "tail_base" in text and "!!" in text
    assert "confidence cut 0.35" in text


def test_render_frame_lists_failures_first(kp, frames):
    text = render_frame(frame_trace(kp, frames, 0, PCUT), PCUT)
    assert "tail_base" in text
    assert "tail_height" in text
    assert text.index("keypoints below the cut") < text.index("could not be computed")


def test_failed_only_suppresses_the_values_table(kp, frames):
    t = frame_trace(kp, frames, 0, PCUT)
    assert "computed values" in render_frame(t, PCUT, only_failed=False)
    assert "computed values" not in render_frame(t, PCUT, only_failed=True)
