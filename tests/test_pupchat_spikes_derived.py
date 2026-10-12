"""Spike flagging, stillness runs, and weight shift.

The distinction these tests defend hardest: a spike (jump and snap back) is a tracking
failure, a step (jump and stay) is a movement. Flagging the second as unreliable would
throw away exactly the events the pipeline exists to detect.
"""
from __future__ import annotations

import numpy as np
import pytest

from pupchat.derived import (NO_RELIABILITY, paw_visibility, stillness_intervals,
                             weight_shift, weight_shift_entry)
from pupchat.reliability import Reliability
from pupchat.spikes import find_spikes, scan, summarise

FPS = 10.0


def rel(name, scale, within=5.0):
    return Reliability(name=name, error_sd=scale, noise_sd=scale, within_clip=within,
                       across_dogs=5.0, source="measured against ground truth",
                       ref_coverage=0.9)


# =============================================================== spikes

def test_a_jump_that_snaps_back_is_flagged():
    x = np.full(40, 10.0)
    x[20] = 40.0
    found = find_spikes(x, scale=1.0, fps=FPS)
    assert [s["frame"] for s in found] == [20]
    assert found[0]["width_frames"] == 1
    assert found[0]["deviation"] == pytest.approx(30.0)


def test_a_step_change_is_not_flagged():
    """Jump and STAY. This is a real movement and must survive untouched."""
    x = np.concatenate([np.full(20, 10.0), np.full(20, 40.0)])
    assert find_spikes(x, scale=1.0, fps=FPS) == []


def test_a_sustained_excursion_is_not_a_spike():
    """Out for longer than max_width means the dog moved and came back, which is a
    movement with a duration, not a tracking glitch."""
    x = np.full(40, 10.0)
    x[20:30] = 40.0
    assert find_spikes(x, scale=1.0, fps=FPS, max_width=3) == []


def test_a_two_frame_spike_is_still_caught():
    x = np.full(40, 10.0)
    x[20:22] = 40.0
    found = find_spikes(x, scale=1.0, fps=FPS)
    assert len(found) == 1 and found[0]["width_frames"] == 2


def test_threshold_scales_with_the_features_own_error():
    """The same jump is a spike for a precise feature and noise for a sloppy one."""
    x = np.full(40, 10.0)
    x[20] = 16.0
    assert find_spikes(x, scale=1.0, fps=FPS, sigmas=4.0)
    assert find_spikes(x, scale=5.0, fps=FPS, sigmas=4.0) == []


def test_noise_below_the_threshold_is_not_flagged():
    rng = np.random.default_rng(0)
    x = 10.0 + rng.normal(0, 1.0, 200)
    assert find_spikes(x, scale=1.0, fps=FPS, sigmas=4.0) == []


def test_angular_features_wrap():
    """An angle crossing 180 looks like a 360-unit jump; wrapping makes it a small
    real move rather than a false spike."""
    x = np.array([179.0] * 10 + [-179.0] + [179.0] * 10)
    assert find_spikes(x, scale=1.0, fps=FPS, angular=True) == []
    assert find_spikes(x, scale=1.0, fps=FPS, angular=False)


def test_scan_skips_features_with_no_measured_error():
    """Without a scale there is no honest threshold, so no verdict is offered."""
    frames = [{"known": 10.0, "unknown": 10.0} for _ in range(40)]
    frames[20] = {"known": 40.0, "unknown": 40.0}
    out = scan(frames, {"known": rel("known", 1.0)}, FPS)
    assert "known" in out and "unknown" not in out


def test_summary_lists_flagged_frames_for_the_evidence_json():
    frames = [{"a": 10.0, "b": 10.0} for _ in range(40)]
    frames[20] = {"a": 40.0, "b": 40.0}
    s = summarise(scan(frames, {"a": rel("a", 1.0), "b": rel("b", 1.0)}, FPS))
    assert s["flagged_frames"] == [20]
    assert s["n_flagged_frames"] == 1
    assert s["per_feature"] == {"a": 1, "b": 1}


# =============================================================== stillness

def test_a_still_run_is_reported():
    frames = [{"x": 1.0} for _ in range(40)]
    out = stillness_intervals(frames, {"x": rel("x", 0.5)}, FPS, min_seconds=2.0)
    assert any(e["scope"] == "all features" for e in out)
    assert out[0]["duration_s"] >= 2.0


def test_movement_above_the_noise_floor_breaks_the_run():
    frames = [{"x": 1.0} for _ in range(40)]
    for i in range(18, 22):
        frames[i]["x"] = 50.0
    out = [e for e in stillness_intervals(frames, {"x": rel("x", 0.5)}, FPS)
           if e["scope"] == "all features"]
    assert all(e["duration_s"] < 4.0 for e in out)


def test_movement_below_the_noise_floor_does_not_break_it():
    """Wobble inside the measurement error is not movement, by definition."""
    rng = np.random.default_rng(1)
    frames = [{"x": 1.0 + rng.normal(0, 0.05)} for _ in range(40)]
    out = stillness_intervals(frames, {"x": rel("x", 1.0)}, FPS, min_seconds=2.0)
    assert any(e["scope"] == "all features" for e in out)


def test_runs_shorter_than_the_minimum_are_not_reported():
    frames = [{"x": float(i % 2) * 50} for i in range(40)]
    assert stillness_intervals(frames, {"x": rel("x", 0.5)}, FPS,
                               min_seconds=2.0) == []


def test_one_restless_feature_does_not_hide_the_others():
    """Whole-dog stillness needs every feature still at once, so a single jittery one
    vetoes it. Per-feature runs must still be reported, or an empty list reads as
    'never stopped moving' when it means 'one measurement never settled'."""
    frames = [{"calm": 1.0, "jumpy": float(i % 2) * 50} for i in range(40)]
    table = {"calm": rel("calm", 0.5), "jumpy": rel("jumpy", 0.5)}
    out = stillness_intervals(frames, table, FPS, min_seconds=2.0)
    assert not [e for e in out if e["scope"] == "all features"]
    assert [e for e in out if e.get("feature") == "calm"]


def test_stillness_is_marked_unvalidated():
    frames = [{"x": 1.0} for _ in range(40)]
    out = stillness_intervals(frames, {"x": rel("x", 0.5)}, FPS)
    assert all(e["source"] == NO_RELIABILITY for e in out)


# =============================================================== weight shift

SA = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
      "back_left_thai", "back_right_thai"]


def body_with(hip_x, *, paws_visible=True, n=20):
    b = np.zeros((n, len(SA), 3))
    pos = {"front_left_paw": (100.0, 200.0), "front_right_paw": (100.0, 210.0),
           "back_left_paw": (0.0, 200.0), "back_right_paw": (0.0, 210.0),
           "back_left_thai": (hip_x, 150.0), "back_right_thai": (hip_x, 160.0)}
    for i, name in enumerate(SA):
        b[:, i, :2] = pos[name]
        b[:, i, 2] = 0.9
    if not paws_visible:
        for name in ("front_left_paw", "front_right_paw",
                     "back_left_paw", "back_right_paw"):
            b[:, SA.index(name), 2] = 0.05
    return b


def test_weight_over_the_rear_paws_reads_zero():
    v = weight_shift(body_with(0.0), SA, 0.35)
    assert np.nanmedian(v) == pytest.approx(0.0, abs=0.01)


def test_weight_over_the_front_paws_reads_one():
    v = weight_shift(body_with(100.0), SA, 0.35)
    assert np.nanmedian(v) == pytest.approx(1.0, abs=0.01)


def test_weight_shift_is_unavailable_when_paws_are_not_visible():
    e = weight_shift_entry(body_with(50.0, paws_visible=False), SA, 0.35, FPS)
    assert e["available"] is False
    assert "paws were not visible" in e["reason"]
    assert e["paw_visibility"]["any_front"] == 0.0


def test_weight_shift_is_available_when_paws_are_visible():
    e = weight_shift_entry(body_with(50.0), SA, 0.35, FPS)
    assert e["available"] is True
    assert e["level"] == pytest.approx(0.5, abs=0.01)


def test_weight_shift_is_marked_unvalidated_even_when_available():
    """It has no measured error_sd, so it must not sit alongside the validated 20 as
    though it were on the same footing."""
    e = weight_shift_entry(body_with(50.0), SA, 0.35, FPS)
    assert e["source"] == NO_RELIABILITY
    assert "unvalidated" in e["note"]


def test_paw_visibility_is_reported_either_way():
    e = weight_shift_entry(body_with(50.0), SA, 0.35, FPS)
    assert set(e["paw_visibility"]) >= {"any_front", "any_back", "front_left_paw"}
