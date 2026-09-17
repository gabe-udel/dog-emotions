"""Clip-level aggregation.

Two of these pin bugs a synthetic smoke test caught, both of which produced plausible
numbers rather than errors:

* `test_angular_smoothing_does_not_average_across_the_wrap` - the rolling median was
  linear. At the edges of a series the window has EVEN length, so `np.median` averages
  the two middle values, and averaging +179 with -179.5 gives ~0. Two such frames in
  ninety pushed a perfectly steady head_tilt's circular spread from 0.8 to 17.3 degrees.
* `test_oscillation_is_not_attenuated_by_smoothing` - rhythm was measured on the
  smoothed series. Smoothing is a low-pass filter and a wag is high-frequency content,
  so the measurement was quietly destroying what it measured.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aggregate import (aggregate, circ_sd, derivative, motion_energy, oscillation,
                       smooth, window_frames)

FPS = 30.0


def series(n: int = 90, fps: float = FPS, **fns) -> list[dict[str, float]]:
    t = np.arange(n) / fps
    return [{k: float(fn(t[i], i)) for k, fn in fns.items()} for i in range(n)]


# ------------------------------------------------------------------ window sizing

@pytest.mark.parametrize("ms,fps,expect", [(100, 30, 3), (100, 60, 7), (100, 24, 3),
                                           (0, 30, 1)])
def test_window_frames_scales_with_fps(ms: float, fps: float, expect: int) -> None:
    assert window_frames(ms, fps) == expect


def test_window_frames_is_always_odd() -> None:
    for fps in (12, 24, 25, 30, 50, 60):
        for ms in range(10, 400, 11):
            assert window_frames(ms, fps) % 2 == 1


# ------------------------------------------------------------------ regressions

def test_angular_smoothing_does_not_average_across_the_wrap() -> None:
    rows = series(head_tilt=lambda t, i: 179.0 if i % 2 else -179.5)
    out = aggregate(rows, FPS)
    assert out["head_tilt__spread"] < 3.0, (
        f"steady-on-the-wrap series reported {out['head_tilt__spread']} deg of spread")
    assert abs(abs(out["head_tilt__level"]) - 180.0) < 3.0


def test_linear_smoothing_still_rejects_a_spike() -> None:
    """The angular path must not have cost the median's robustness on linear series."""
    x = np.zeros(21)
    x[10] = 100.0
    assert smooth(x, 5)[10] == pytest.approx(0.0)


def test_oscillation_is_not_attenuated_by_smoothing() -> None:
    """Must use a frequency the smoothing window actually eats, or it proves nothing.

    At 3 Hz / 30 fps the 100 ms window costs 0% of the amplitude, so a test there passes
    whether or not the rhythm is read from the raw series - which is how the first
    version of this test slipped through. At 5 Hz / 60 fps the window costs 26.8%, well
    outside the tolerance below, so this fails if oscillation ever reads the smoothed
    series again. 5 Hz is inside the real band: dog wags run ~3-5 Hz.
    """
    fps, freq, amp, n = 60.0, 5.0, 0.20, 180
    t = np.arange(n) / fps
    rows = [{"tail_lateral": float(amp * np.sin(2 * np.pi * freq * t[i]))}
            for i in range(n)]
    out = aggregate(rows, fps, noise_floor={"tail_lateral": 0.01})
    truth = float(np.median(np.abs(amp * np.sin(2 * np.pi * freq * t))))
    assert out["wag_amp"] == pytest.approx(truth, rel=0.10)


# ------------------------------------------------------------------ oscillation

def test_wag_rate_recovers_a_known_frequency() -> None:
    rows = series(tail_lateral=lambda t, i: 0.2 * np.sin(2 * np.pi * 3.0 * t))
    out = aggregate(rows, FPS, noise_floor={"tail_lateral": 0.01})
    assert out["wag_rate_hz"] == pytest.approx(3.0, abs=0.35)


def test_noise_floor_suppresses_a_rate_from_jitter() -> None:
    """Without the gate, jitter around a motionless tail manufactures a wag."""
    rng = np.random.default_rng(0)
    tiny = rng.normal(0.0, 0.001, 90)
    rate, amp = oscillation(tiny, FPS, noise_floor=0.01)
    assert math.isnan(rate)
    assert amp < 0.01


def test_missing_noise_floor_fails_closed() -> None:
    """An absent floor must mean NO ANSWER, not an ungated one.

    Defaulting the floor to 0.0 disabled the gate silently. On a real clip that gave a
    9.97 Hz "wag" and a 13.10 Hz "mouth rhythm" - both pure jitter, since counting the
    zero crossings of noise always yields a confident, meaningless frequency.
    """
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 0.001, 90)
    rate, _ = oscillation(jitter, FPS, noise_floor=None)
    assert math.isnan(rate)

    rows = [{"tail_lateral": float(v)} for v in jitter]
    out = aggregate(rows, FPS)                      # no noise_floor supplied
    assert math.isnan(out["wag_rate_hz"])


# ------------------------------------------------------------------ derivatives

def test_derivative_is_per_second_not_per_frame() -> None:
    """The same physical ramp must give the same rate at any frame rate."""
    for fps in (25.0, 30.0, 60.0):
        n = 60
        x = np.arange(n) * (2.0 / fps)          # 2 units per second
        d = derivative(x, fps, angular=False)
        assert np.nanmedian(d) == pytest.approx(2.0, rel=1e-6)


def test_angular_derivative_wraps() -> None:
    """A step from 179 to -179 is 2 degrees of motion, not 358."""
    x = np.array([179.0, 180.0, -179.0, -178.0, -177.0])
    d = derivative(x, FPS, angular=True)
    assert np.nanmax(np.abs(d)) < 100.0


# ------------------------------------------------------------------ bookkeeping

def test_coverage_counts_missing_frames() -> None:
    rows = series(ear_height=lambda t, i: np.nan if i % 3 == 0 else 0.5)
    out = aggregate(rows, FPS)
    assert out["ear_height__coverage"] == pytest.approx(2 / 3, abs=0.02)


def test_bookkeeping_columns_are_not_summarised() -> None:
    """face_px records whether to trust a frame; it is not a signal about the dog."""
    out = aggregate(series(face_px=lambda t, i: 300.0), FPS)
    assert "face_px__coverage" in out
    assert "face_px__level" not in out


def test_event_percentiles_only_where_the_signal_is_an_event() -> None:
    rows = series(tail_tuck=lambda t, i: 0.9 if i < 70 else 0.3,
                  mouth_open_jaw=lambda t, i: 0.1,
                  head_pitch=lambda t, i: 10.0)
    out = aggregate(rows, FPS)
    assert out["tail_tuck__p10"] == pytest.approx(0.3, abs=0.05)   # the tuck itself
    assert "mouth_open_jaw__p90" in out                            # widest gape
    assert "head_pitch__p10" not in out and "head_pitch__p90" not in out


def test_empty_input_is_empty_output() -> None:
    assert aggregate([], FPS) == {}


# ------------------------------------------------------------------ motion energy

def test_motion_energy_ignores_whole_body_translation() -> None:
    """A panning camera, or a dog simply walking across frame, moves every point
    together - that is not the dog moving against itself."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 30.0, (39, 2))
    n = 8
    bodies = np.zeros((n, 39, 3))
    for i in range(n):
        bodies[i, :, :2] = base + np.array([12.0 * i, -7.0 * i])   # pure translation
        bodies[i, :, 2] = 0.9
    faces = np.full((n, 38, 3), np.nan)
    e = motion_energy(bodies, faces, np.full(n, 200.0), np.full(n, 80.0), FPS)
    assert np.nanmax(np.abs(e[1:])) < 1e-6


def test_motion_energy_sees_real_articulation() -> None:
    rng = np.random.default_rng(2)
    n = 8
    bodies = np.zeros((n, 39, 3))
    bodies[:, :, 2] = 0.9
    base = rng.normal(0.0, 30.0, (39, 2))
    for i in range(n):
        bodies[i, :, :2] = base + rng.normal(0.0, 4.0, (39, 2))
    faces = np.full((n, 38, 3), np.nan)
    e = motion_energy(bodies, faces, np.full(n, 200.0), np.full(n, 80.0), FPS)
    assert np.nanmedian(e[1:]) > 0.0


def test_circ_sd_reads_a_steady_wrapped_series_as_steady() -> None:
    assert circ_sd(np.array([179.2, -179.7, 178.5, -179.9])) < 2.0
