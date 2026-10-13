"""Proving accuracy on local footage.

The method is the project's own, applied to a different set of images: label, difference
against the prediction, report the spread. These tests pin the arithmetic and the two
properties that decide whether the result means anything - even sampling, and keypoint
error that does not pretend to predict feature error.
"""
from __future__ import annotations

import numpy as np
import pytest

from pupchat.groundtruth import (compare_to_dogflw, labelling_progress,
                                 load_labels, sample_indices, save_labels,
                                 score_features, score_keypoints)

NAMES = ["nose", "tail_base", "left_eye"]


# ------------------------------------------------------------------ sampling

def test_sampling_is_even_across_the_clip():
    """Choosing the frames where the dog looks clear measures the easy cases, which is
    exactly the bias that makes an accuracy number flattering."""
    idx = sample_indices(100, 5)
    gaps = np.diff(idx)
    assert idx[0] == 0 and idx[-1] == 99
    assert gaps.max() - gaps.min() <= 1


def test_sampling_is_deterministic():
    """A second labelling session must cover the same frames, or the two cannot be
    compared."""
    assert sample_indices(100, 7) == sample_indices(100, 7)


def test_asking_for_more_frames_than_exist_returns_all():
    assert sample_indices(4, 10) == [0, 1, 2, 3]


# ------------------------------------------------------------------ keypoint error

def simple_case(offset_px=5.0, score=0.9):
    pred = np.zeros((2, len(NAMES), 3))
    pred[:, :, 2] = score
    pred[:, 0, :2] = [100.0 + offset_px, 100.0]
    pred[:, 1, :2] = [200.0, 200.0]
    pred[:, 2, :2] = [50.0, 50.0]
    labelled = {
        "f0": {"nose": [100.0, 100.0], "tail_base": [200.0, 200.0]},
        "f1": {"nose": [100.0, 100.0], "tail_base": [200.0, 200.0]},
    }
    return pred, labelled, {"f0": 0, "f1": 1}, {"f0": 100.0, "f1": 100.0}


def test_error_is_normalised_by_the_box_diagonal():
    """5 px on a 100 px diagonal is 0.05, the same as 10 px on 200 - which is the whole
    point of normalising."""
    pred, lab, fi, norm = simple_case(offset_px=5.0)
    r = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi, norm)
    assert r["per_keypoint"]["nose"]["nme"] == pytest.approx(0.05)
    assert r["per_keypoint"]["tail_base"]["nme"] == pytest.approx(0.0)


def test_a_bigger_subject_gives_a_smaller_normalised_error():
    pred, lab, fi, _ = simple_case(offset_px=5.0)
    small = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi,
                            {"f0": 100.0, "f1": 100.0})
    large = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi,
                            {"f0": 500.0, "f1": 500.0})
    assert large["per_keypoint"]["nose"]["nme"] < small["per_keypoint"]["nose"]["nme"]


def test_pck_counts_the_fraction_inside_the_threshold():
    pred, lab, fi, norm = simple_case(offset_px=4.0)
    r = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi, norm)
    assert r["per_keypoint"]["nose"]["pck5"] == 1.0
    pred2, lab2, fi2, norm2 = simple_case(offset_px=9.0)
    r2 = score_keypoints(lab2, pred2, NAMES, pred2[:, :, 2], fi2, norm2)
    assert r2["per_keypoint"]["nose"]["pck5"] == 0.0
    assert r2["per_keypoint"]["nose"]["pck10"] == 1.0


def test_predictions_below_the_confidence_cut_are_not_scored():
    """Scoring a point the pipeline already refused to report would measure something
    the pipeline never claimed."""
    pred, lab, fi, norm = simple_case(offset_px=5.0, score=0.1)
    r = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi, norm, pcut=0.35)
    assert r["per_keypoint"] == {}
    assert r["skipped_below_confidence_cut"] == 4


def test_unlabelled_keypoints_are_simply_absent():
    pred, lab, fi, norm = simple_case()
    r = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi, norm)
    assert "left_eye" not in r["per_keypoint"]


def test_worst_keypoint_is_listed_first():
    pred, lab, fi, norm = simple_case(offset_px=5.0)
    r = score_keypoints(lab, pred, NAMES, pred[:, :, 2], fi, norm)
    assert list(r["per_keypoint"])[0] == "nose"


# ------------------------------------------------------------------ feature error

def test_feature_error_is_the_difference_of_the_two_computations():
    pred = [{"ear_len": 0.20}, {"ear_len": 0.22}, {"ear_len": 0.19}, {"ear_len": 0.21}]
    truth = [{"ear_len": 0.20}, {"ear_len": 0.20}, {"ear_len": 0.20}, {"ear_len": 0.20}]
    r = score_features(pred, truth, angular=set())
    assert r["ear_len"]["n"] == 4
    assert r["ear_len"]["error_sd"] == pytest.approx(np.std([0, .02, -.01, .01]))


def test_bias_is_reported_separately_from_scatter():
    """A feature can be precise and wrong. Reporting only the spread hides that."""
    pred = [{"x": 1.5} for _ in range(5)]
    truth = [{"x": 1.0} for _ in range(5)]
    r = score_features(pred, truth, angular=set())
    assert r["x"]["bias"] == pytest.approx(0.5)
    assert r["x"]["error_sd"] == pytest.approx(0.0)


def test_angular_features_wrap_so_a_crossing_is_not_a_huge_error():
    pred = [{"a": 179.0}] * 5
    truth = [{"a": -179.0}] * 5
    plain = score_features(pred, truth, angular=set())
    wrapped = score_features(pred, truth, angular={"a"})
    assert plain["a"]["bias"] == pytest.approx(358.0)
    assert wrapped["a"]["bias"] == pytest.approx(-2.0)


def test_feature_error_is_not_a_rescaling_of_keypoint_error():
    """The reason step 4 exists. Two features built from the SAME keypoints, with the
    same shared error, land in different places: a difference cancels it, a ratio with
    a small denominator amplifies it."""
    shift = 5.0
    a, b = 100.0, 110.0
    diff_pred = [{"d": (a + shift) - (b + shift)}]
    diff_truth = [{"d": a - b}]
    ratio_pred = [{"r": (a + shift) / 2.0}]
    ratio_truth = [{"r": a / 2.0}]
    assert score_features(diff_pred * 4, diff_truth * 4,
                          set())["d"]["max_abs"] == pytest.approx(0.0)
    assert score_features(ratio_pred * 4, ratio_truth * 4,
                          set())["r"]["max_abs"] == pytest.approx(2.5)


def test_features_with_too_few_paired_observations_are_dropped():
    pred = [{"x": 1.0}, {"x": float("nan")}]
    truth = [{"x": 1.0}, {"x": 1.0}]
    assert score_features(pred, truth, set()) == {}


# ------------------------------------------------------------------ comparison

def test_comparison_flags_features_worse_here_than_on_dogflw():
    local = {"mouth_open_jaw": {"error_sd": 0.080, "bias": 0.0, "n": 20},
             "ear_height": {"error_sd": 0.036, "bias": 0.0, "n": 20}}
    ref = {"mouth_open_jaw": {"error_sd": 0.0201},
           "ear_height": {"error_sd": 0.0362}}
    rows = compare_to_dogflw(local, ref)
    assert rows[0]["feature"] == "mouth_open_jaw"
    assert rows[0]["ratio"] == pytest.approx(3.98, abs=0.01)
    assert rows[1]["ratio"] == pytest.approx(0.99, abs=0.02)


def test_comparison_handles_a_feature_dogflw_never_measured():
    rows = compare_to_dogflw({"weight_shift": {"error_sd": 0.1, "bias": 0.0, "n": 9}},
                             {})
    assert rows[0]["dogflw_error_sd"] is None and rows[0]["ratio"] is None


# ------------------------------------------------------------------ session

def test_labels_round_trip(tmp_path):
    payload = {"frames": {"f0": {"nose": [1.0, 2.0]}}, "video": "clip.mp4"}
    save_labels(tmp_path / "gt.json", payload)
    assert load_labels(tmp_path / "gt.json") == payload


def test_loading_a_missing_file_is_empty_not_an_error(tmp_path):
    assert load_labels(tmp_path / "nope.json") == {}


def test_progress_lets_a_session_be_resumed():
    payload = {"frames": {"f0": {"a": [0, 0], "b": [1, 1]}, "f1": {"a": [0, 0]}}}
    p = labelling_progress(payload, wanted=[0, 1, 2, 3], per_frame=2)
    assert p["frames_labelled"] == 2
    assert p["frames_complete"] == 1
    assert p["points_placed"] == 3
    assert p["fraction"] == 0.5
