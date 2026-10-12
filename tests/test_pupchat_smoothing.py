"""One Euro over keypoint tracks, and the guard against it becoming a flattener.

The requirement was two-sided: cut jitter, do not eat fast motion. Those pull against
each other through beta, so the tests pin both ends. The numbers in the retention tests
are the measured ones from the parameter sweep, not aspirations.
"""
from __future__ import annotations

import numpy as np
import pytest

from pupchat.config import SmoothingConfig
from pupchat.smoothing import (OneEuroConfig, filter_keypoints, filter_series,
                               jitter, retention)

FPS = 12.5
T = np.arange(0, 4, 1 / FPS)


def noisy_still(n=60, amp=3.0, seed=0):
    """A stationary keypoint set with per-frame noise: the still-dog case."""
    rng = np.random.default_rng(seed)
    base = np.array([[100.0, 100.0], [200.0, 120.0], [150.0, 180.0], [120.0, 90.0]])
    track = np.zeros((n, len(base), 3))
    track[:, :, :2] = base[None] + rng.normal(0, amp, (n, len(base), 2))
    track[:, :, 2] = 0.9
    return track


# ------------------------------------------------------------------ jitter

def test_filter_reduces_jitter_on_a_still_track():
    raw = noisy_still()
    before = jitter(raw)
    after = jitter(filter_keypoints(raw, FPS, OneEuroConfig(0.5, 5.0), pcut=0.35))
    assert after < before


def test_jitter_ignores_whole_frame_translation():
    """A dog walking across frame is not jitter. The metric subtracts the mean
    displacement, so a pure pan must read zero."""
    n = 30
    track = np.zeros((n, 5, 3))
    track[:, :, 2] = 0.9
    base = np.array([[0.0, 0], [10, 0], [20, 0], [30, 0], [40, 0]])
    for i in range(n):
        track[i, :, :2] = base + np.array([7.0 * i, 0.0])
    assert jitter(track) == pytest.approx(0.0, abs=1e-9)


def test_more_smoothing_means_less_jitter():
    raw = noisy_still()
    heavy = jitter(filter_keypoints(raw, FPS, OneEuroConfig(0.5, 0.7), pcut=0.35))
    light = jitter(filter_keypoints(raw, FPS, OneEuroConfig(0.5, 20.0), pcut=0.35))
    assert heavy < light


# ------------------------------------------------------------------ fast motion

@pytest.mark.parametrize("hz,floor", [(1.0, 0.95), (3.0, 0.90), (5.0, 0.85)])
def test_default_settings_do_not_flatten_fast_motion(hz, floor):
    """The shipped default must keep the band a wag lives in. Measured at beta=5.0:
    1 Hz 99%, 3 Hz 92%, 5 Hz 87%."""
    cfg = SmoothingConfig()
    sig = np.sin(2 * np.pi * hz * T)
    euro = OneEuroConfig(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    assert retention(sig, filter_series(sig, FPS, euro)) >= floor


def test_the_papers_default_beta_would_have_flattened_it():
    """beta=0.7 keeps less of a 3 Hz wag than the rolling median already in the
    pipeline, which is why the default here is 5.0 and not the usual 0.7."""
    sig = np.sin(2 * np.pi * 3.0 * T)
    assert retention(sig, filter_series(sig, FPS, OneEuroConfig(0.5, 0.7))) < 0.70


def test_a_step_change_survives_intact():
    """A sudden head drop is the signal, not noise."""
    step = np.where(T < 2, 0.0, 1.0)
    cfg = SmoothingConfig()
    out = filter_series(step, FPS, OneEuroConfig(cfg.min_cutoff, cfg.beta))
    assert retention(step, out) >= 0.95


def test_higher_beta_keeps_more_fast_motion():
    sig = np.sin(2 * np.pi * 5.0 * T)
    low = retention(sig, filter_series(sig, FPS, OneEuroConfig(0.5, 1.0)))
    high = retention(sig, filter_series(sig, FPS, OneEuroConfig(0.5, 20.0)))
    assert high > low


# ------------------------------------------------------------------ mechanics

def test_scores_are_never_modified():
    """A smoothed position is still only as trustworthy as the detection behind it."""
    raw = noisy_still()
    raw[:, 1, 2] = 0.12
    out = filter_keypoints(raw, FPS, OneEuroConfig(), pcut=0.35)
    assert np.array_equal(out[:, :, 2], raw[:, :, 2])


def test_low_confidence_points_are_treated_as_gaps_not_chased():
    """A position the pipeline already distrusts must not drag the track toward it."""
    n = 40
    track = np.zeros((n, 1, 3))
    track[:, 0, 0] = 100.0
    track[:, 0, 2] = 0.9
    track[20, 0, 0] = 10_000.0          # a wild detection...
    track[20, 0, 2] = 0.05              # ...that the cut already rejects
    out = filter_keypoints(track, FPS, OneEuroConfig(0.5, 5.0), pcut=0.35)
    assert abs(out[25, 0, 0] - 100.0) < 1.0


def test_nans_pass_through_without_poisoning_the_filter():
    x = np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    out = filter_series(x, FPS)
    assert np.isnan(out[2])
    assert np.isfinite(out[3]) and abs(out[-1] - 1.0) < 0.5


def test_disabled_config_is_a_passthrough():
    raw = noisy_still()
    out = filter_keypoints(raw, FPS, OneEuroConfig(enabled=False))
    assert np.array_equal(out, raw)


def test_retention_of_an_unchanged_signal_is_one():
    sig = np.sin(2 * np.pi * 2.0 * T)
    assert retention(sig, sig) == pytest.approx(1.0)
