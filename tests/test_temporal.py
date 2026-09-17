"""Temporal smoothing and RANSAC rejection.

The tests that matter are the ones that pin down what RANSAC must NOT do: it must not
flag a fast but genuine movement, and it must not touch a point it has too little
evidence to judge. Over-rejection is the failure mode that would quietly delete real
motion from the emotion features, and it looks identical to success in a rendered video.
"""
from __future__ import annotations

import numpy as np
import pytest

from temporal import ransac_filter, rolling_median, summarise, window_frames


# ------------------------------------------------------------------ window sizing

@pytest.mark.parametrize("ms,fps,expect", [
    (100, 30, 3),      # the old hard-coded k=3, recovered at 30 fps
    (100, 60, 7),      # ...and doubled at 60, which is the whole point
    (100, 24, 3),
    (230, 30, 7),
    (0, 30, 1),        # disabled
])
def test_window_frames_scales_with_fps(ms: float, fps: float, expect: int) -> None:
    assert window_frames(ms, fps) == expect


def test_window_frames_always_odd() -> None:
    """An even window has no centre frame, so it would bias half a frame forward."""
    for fps in (12, 24, 25, 30, 50, 60, 120):
        for ms in range(10, 500, 7):
            assert window_frames(ms, fps) % 2 == 1


def test_window_frames_survives_a_broken_fps() -> None:
    assert window_frames(100, 0) == 1
    assert window_frames(100, -1) == 1


# ------------------------------------------------------------------ smoothing

def _seq(xy: np.ndarray, conf: float = 1.0) -> list[np.ndarray]:
    """(T, K, 2) -> the list-of-(K,3) layout the pipeline passes around."""
    t, k, _ = xy.shape
    return [np.concatenate([xy[i], np.full((k, 1), conf)], axis=1) for i in range(t)]


def test_rolling_median_k1_is_identity() -> None:
    seq = _seq(np.random.default_rng(0).normal(size=(10, 3, 2)))
    assert rolling_median(seq, 1) is seq


def test_rolling_median_passes_none_through() -> None:
    seq = _seq(np.zeros((5, 2, 2)))
    seq[2] = None
    out = rolling_median(seq, 3)
    assert out[2] is None
    assert all(o is not None for i, o in enumerate(out) if i != 2)


def test_rolling_median_removes_a_spike() -> None:
    xy = np.zeros((7, 1, 2))
    xy[3] = 100.0
    out = rolling_median(_seq(xy), 5)
    assert out[3][0, 0] == pytest.approx(0.0)


# ------------------------------------------------------------------ RANSAC

def _linear(t: int = 15, k: int = 2, vx: float = 2.0) -> np.ndarray:
    xy = np.zeros((t, k, 2))
    xy[:, :, 0] = (np.arange(t) * vx)[:, None]
    xy[:, :, 1] = 50.0
    return xy


def test_clean_linear_motion_has_no_outliers() -> None:
    """Constant velocity is exactly the model, so nothing may be rejected."""
    _, mask = ransac_filter(_seq(_linear()), window=7, thresh=1.0)
    assert not mask.any()


def test_rejects_a_single_jumped_landmark() -> None:
    xy = _linear()
    xy[7, 0] += 40.0                     # landmark 0 snaps away on frame 7 only
    out, mask = ransac_filter(_seq(xy), window=7, thresh=2.0)
    assert mask[7, 0]
    assert mask.sum() == 1, "only the corrupted point may be flagged"
    # repaired back onto the trajectory it left
    assert out[7][0, 0] == pytest.approx(14.0, abs=0.5)


def test_does_not_flag_genuine_fast_motion() -> None:
    """A dog's head snapping round is fast but still smooth - it must survive.

    This is the over-rejection failure mode. The trajectory below moves 20 px/frame,
    ten times the baseline, and every point is consistent with its neighbours.
    """
    _, mask = ransac_filter(_seq(_linear(vx=20.0)), window=7, thresh=1.0)
    assert not mask.any()


def test_leaves_the_point_alone_without_enough_consensus() -> None:
    """Too few voting frames means no opinion, not a rejection."""
    xy = _linear(t=9)
    seq = _seq(xy)
    for i in (0, 1, 2, 5, 6, 7, 8):      # only frames 3,4 remain valid
        seq[i] = None
    _, mask = ransac_filter(seq, window=7, thresh=1.0)
    assert not mask.any()


def test_low_confidence_points_do_not_vote_but_are_judged() -> None:
    xy = _linear()
    seq = _seq(xy)
    seq[7][0, 0] += 40.0
    seq[7][0, 2] = 0.01                  # wrong AND unconfident
    _, mask = ransac_filter(seq, window=7, thresh=2.0, conf_min=0.35)
    assert mask[7, 0], "a low-confidence point must still be checkable"


def test_a_confident_outlier_cannot_win_the_consensus() -> None:
    """Two adjacent bad frames must not out-vote five good ones."""
    xy = _linear(t=15)
    xy[7:9, 0] += 60.0
    _, mask = ransac_filter(_seq(xy), window=9, thresh=2.0)
    assert mask[7, 0] and mask[8, 0]


def test_drop_repair_zeroes_the_score_and_keeps_coordinates() -> None:
    xy = _linear()
    xy[7, 0] += 40.0
    out, mask = ransac_filter(_seq(xy), window=7, thresh=2.0, repair="drop")
    assert out[7][0, 2] == 0.0
    assert out[7][0, 0] == pytest.approx(xy[7, 0, 0]), "drop must not move the point"


def test_scale_makes_the_threshold_relative() -> None:
    """The same absolute error is an outlier on a small face and noise on a large one.

    This is the confound §10 removed from the accuracy numbers; the temporal stage must
    not smuggle it back in by judging every dog with one pixel budget.
    """
    xy = _linear()
    xy[7, 0] += 20.0
    small = np.full(len(xy), 100.0)
    large = np.full(len(xy), 2000.0)
    _, m_small = ransac_filter(_seq(xy), window=7, thresh=0.05, scale=small)
    _, m_large = ransac_filter(_seq(xy), window=7, thresh=0.05, scale=large)
    assert m_small[7, 0]
    assert not m_large[7, 0]


def test_missing_frames_pass_through_as_none() -> None:
    seq = _seq(_linear())
    seq[4] = None
    out, mask = ransac_filter(seq, window=7, thresh=2.0)
    assert out[4] is None
    assert not mask[4].any()


def _curved(t: int = 15, curv: float = 2.0) -> np.ndarray:
    xy = np.zeros((t, 1, 2))
    xy[:, 0, 0] = curv * np.arange(t) ** 2
    xy[:, 0, 1] = 50.0
    return xy


@pytest.mark.parametrize("curv", [0.5, 1.0, 2.0, 5.0, 10.0, 25.0])
@pytest.mark.parametrize("thresh", [1.0, 2.0, 4.0, 8.0, 16.0])
@pytest.mark.parametrize("degree", [1, 2])
def test_smooth_acceleration_is_never_an_outlier(curv: float, thresh: float,
                                                 degree: int) -> None:
    """A clean accelerating trajectory must survive at every threshold.

    Regression test for a tie-break defect. Ties on inlier count were resolved by
    enumeration order, so at particular threshold/curvature combinations a model that
    merely EQUALLED the honest one could win and evict a perfectly good point - which
    rejected 12 of 15 frames of a clean trajectory. Ties now go to the model that fits
    the frame under test, so only strictly better-supported evidence can condemn it.
    """
    _, mask = ransac_filter(_seq(_curved(curv=curv)), window=7, thresh=thresh,
                            degree=degree)
    assert not mask.any()


def test_degree_two_localises_a_spike_on_a_curve_better() -> None:
    """Under strong curvature degree 1 blames the neighbours; degree 2 gets it right.

    This is the honest limit of the constant-velocity model: a line through the spike
    and one neighbour explains a sharply curving window about as well as the truth, so
    the wrong frames get flagged. It stays the default anyway because it is the safer
    of the two when motion is mild, which is most of the time - but it is why
    --ransac-degree 2 exists.
    """
    xy = _curved(curv=5.0)
    xy[7, 0, 0] += 40.0
    _, m1 = ransac_filter(_seq(xy), window=7, thresh=2.0, degree=1)
    _, m2 = ransac_filter(_seq(xy), window=9, thresh=2.0, degree=2)
    assert not m1[7, 0] and m1.any(), "degree 1 mislocalises under strong curvature"
    assert list(np.flatnonzero(m2[:, 0])) == [7]


def test_is_deterministic() -> None:
    """Subsets are enumerated, not sampled, so two runs cannot disagree."""
    rng = np.random.default_rng(3)
    xy = _linear(t=20, k=5) + rng.normal(scale=0.3, size=(20, 5, 2))
    xy[9, 2] += 50.0
    a, ma = ransac_filter(_seq(xy), window=7, thresh=2.0)
    b, mb = ransac_filter(_seq(xy), window=7, thresh=2.0)
    assert np.array_equal(ma, mb)
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_rejects_an_impossible_window() -> None:
    with pytest.raises(ValueError):
        ransac_filter(_seq(_linear()), window=2, thresh=1.0, degree=1)
    with pytest.raises(ValueError):
        ransac_filter(_seq(_linear()), window=7, thresh=1.0, repair="nonsense")


def test_summarise_reports_nothing_when_clean() -> None:
    assert "no outliers" in summarise(np.zeros((10, 4), dtype=bool))


def test_summarise_names_the_worst_landmark() -> None:
    mask = np.zeros((10, 3), dtype=bool)
    mask[:, 1] = True
    assert "ear_left_tip" in summarise(mask, ["a", "ear_left_tip", "c"])
