"""Comparisons: "higher than most", "lower than usual".

The reason these are computed in Python rather than asked of the model is that they are
claims about measurement, and a model has no way to check one. So they get the same
treatment as the rest of the evidence layer: say nothing rather than say something
unearned.

Two refusals carry the weight. A difference from the dog's own baseline is only called
real once it clears twice the feature's measured error, because below that it is tracking
noise being reported as a fact about the dog. And a feature with no reference gets no
comparison at all - not a reassuring "typical".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pupchat.config import BANNED_WORDS
from pupchat.reference import (OWN_SIGMAS, band, baseline_levels, describe,
                               load_reference, phrase, vs_own)

PCT = {"n": 4333, "p10": 10.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "p90": 90.0}


# ------------------------------------------------------------------ against other dogs

@pytest.mark.parametrize("value,expected", [
    (5.0, "well below most"),
    (15.0, "lower than most"),
    (50.0, "typical"),
    (80.0, "higher than most"),
    (95.0, "well above most"),
])
def test_a_value_is_placed_among_the_reference_dogs(value, expected):
    assert band(value, PCT) == expected


@pytest.mark.parametrize("value,expected", [
    (10.0, "lower than most"),     # p10 itself is not "well below"
    (25.0, "typical"),            # p25 opens the typical band
    (75.0, "typical"),            # p75 closes it
    (90.0, "higher than most"),    # p90 itself is not "well above"
])
def test_the_band_edges_are_where_they_claim_to_be(value, expected):
    assert band(value, PCT) == expected


def test_no_reference_means_no_comparison():
    """Not "typical". A reassuring word for a feature nothing was compared against is a
    claim the pipeline cannot support."""
    assert band(50.0, {}) is None
    assert band(50.0, {"n": 10, "p10": 1.0}) is None       # incomplete


def test_a_missing_measurement_gets_no_band():
    assert band(None, PCT) is None
    assert band(float("nan"), PCT) is None


# ------------------------------------------------------------------ against itself

def test_a_small_change_from_the_baseline_is_called_the_same():
    """Below twice the measured error, a difference is tracking noise. Reporting it
    would turn a wobble into a statement about the dog."""
    assert vs_own(10.0, 10.0, error_sd=1.0) == "about the same as usual"
    assert vs_own(11.9, 10.0, error_sd=1.0) == "about the same as usual"


def test_a_real_change_from_the_baseline_is_reported():
    assert vs_own(12.1, 10.0, error_sd=1.0) == "higher than usual"
    assert vs_own(7.9, 10.0, error_sd=1.0) == "lower than usual"


def test_the_threshold_is_two_measured_errors():
    """Pinned, because loosening it is the easy way to make the feature look better and
    the honest way to make it wrong."""
    assert OWN_SIGMAS == 2.0
    just_under = 10.0 + OWN_SIGMAS * 1.0 - 0.001
    just_over = 10.0 + OWN_SIGMAS * 1.0 + 0.001
    assert vs_own(just_under, 10.0, 1.0) == "about the same as usual"
    assert vs_own(just_over, 10.0, 1.0) == "higher than usual"


def test_no_baseline_means_no_comparison():
    """Most dogs will have none. The line simply says less."""
    assert vs_own(10.0, None, error_sd=1.0) is None


def test_an_unknown_error_means_no_comparison():
    """Without a measured error there is no way to tell a change from noise, so the
    comparison cannot be made honestly at all."""
    assert vs_own(12.0, 10.0, error_sd=None) is None
    assert vs_own(12.0, 10.0, error_sd=0.0) is None


# ------------------------------------------------------------------ what it may say

def test_every_phrase_is_free_of_emotion_words():
    """These describe position relative to a reference, never a feeling. The evidence
    layer bans emotion words precisely so the model is not handed a conclusion dressed
    as a measurement, and these new strings go in the same place."""
    said = set()
    for v in (0.0, 15.0, 50.0, 80.0, 1000.0):
        said.add(band(v, PCT))
    for v in (10.0, 20.0, 0.0):
        said.add(vs_own(v, 10.0, 1.0))
    said.discard(None)
    assert said, "no phrases were produced"

    lowered = " ".join(said).lower()
    for word in BANNED_WORDS:
        assert word not in lowered.split(), f"{word!r} leaked into a comparison"


def test_the_phrase_says_how_many_dogs_it_compared_with():
    """"Higher than most dogs" from photographs is a different claim from one made on
    video, and the count is what lets a reader judge it."""
    said = phrase("ear_closeness_iod", describe(
        "ear_closeness_iod", 95.0, reference={"ear_closeness_iod": PCT},
        dog_baseline=None, error_sd=1.0))
    assert "4,333" in said and "photographed dogs" in said


def test_the_phrase_names_the_dog_for_its_own_baseline():
    said = phrase("x", {"vs_own": "lower than usual"}, dog_name="Biscuit")
    assert "Biscuit's usual" in said


def test_a_feature_with_neither_reference_says_nothing_at_all():
    assert describe("nope", 1.0, reference={}, dog_baseline=None, error_sd=1.0) == {}
    assert phrase("nope", {}) == ""


# ------------------------------------------------------------------ the built file

def test_the_reference_is_face_only_and_body_features_never_get_a_band():
    """The one that would catch someone later "fixing" the gap with uncalibrated
    numbers. Every body feature divides by body length, which needs the withers, and
    StanfordExtra leaves that slot empty in all 2,545 overlapping images."""
    ref = load_reference()
    if not ref:
        pytest.skip("reference.json has not been built on this machine")
    body = {"head_pitch", "head_tilt", "head_height", "tail_height", "tail_tuck",
            "tail_lateral", "tail_len", "posture", "body_L", "motion_energy"}
    assert not (set(ref) & body), f"body features in the reference: {set(ref) & body}"


def test_the_reference_holds_no_bookkeeping_columns():
    """n_paws and tail_conf are coverage, never signal. They survive a scale-free test
    because they are constant here, and a percentile over a column of zeros looks
    exactly like one that means something."""
    ref = load_reference()
    if not ref:
        pytest.skip("reference.json has not been built on this machine")
    for name in ("n_paws", "tail_conf", "head_tilt_used_ground", "face_px", "body_L"):
        assert name not in ref, name


def test_a_missing_reference_file_is_not_an_error(tmp_path):
    """Normal until the builder is run once."""
    assert load_reference(tmp_path / "nope.json") == {}


def test_a_corrupt_reference_file_is_not_an_error(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_reference(path) == {}


# ------------------------------------------------------------------ a dog's own normal

def test_a_dogs_usual_is_the_median_of_its_baselines():
    """The median, not the mean: one badly filmed baseline should not drag the
    reference for every future clip."""
    rows = [{"ear_height": 0.30}, {"ear_height": 0.32}, {"ear_height": 9.0}]
    assert baseline_levels(rows)["ear_height"] == 0.32


def test_missing_values_do_not_poison_a_dogs_usual():
    rows = [{"ear_height": 0.30}, {"ear_height": float("nan")}, {"ear_height": 0.34}]
    assert baseline_levels(rows)["ear_height"] in (0.30, 0.34)
    assert not math.isnan(baseline_levels(rows)["ear_height"])


def test_no_baselines_is_an_empty_usual():
    assert baseline_levels([]) == {}


# ------------------------------------------------------------------ angles

def test_an_angle_never_gets_a_population_band():
    """Found by comparing the distributions, not by reasoning.

    On a real clip ear_angle_L_basevec runs -115..+92 at runtime (median -96) where
    DogFLW's ground truth runs +29..+172 (median +144), and the right ear is offset by
    the same amount. Both ears consistently: a difference of convention between the two
    measurements, not between dogs. A band built across that gap would have read "well
    below most dogs" for a perfectly ordinary pair of ears.
    """
    pct = {"n": 4333, "p10": 29.0, "p25": 60.0, "p50": 144.0, "p75": 160.0, "p90": 172.0}
    for name in ("ear_angle_L_basevec", "ear_angle_R_basevec", "head_tilt",
                 "head_pitch", "tail_height"):
        out = describe(name, -96.0, reference={name: pct}, dog_baseline=None,
                       error_sd=13.0)
        assert "vs_dogs" not in out, f"{name} was given a population band"


def test_the_built_reference_contains_no_angles():
    ref = load_reference()
    if not ref:
        pytest.skip("reference.json has not been built on this machine")
    from pupchat._src import ANGULAR_FEATURES
    assert not (set(ref) & set(ANGULAR_FEATURES))


def test_an_angle_is_still_compared_with_its_own_baseline():
    """Unaffected by the convention problem: both sides come from this pipeline, in the
    same convention."""
    assert vs_own(-70.0, -96.0, 5.0, name="ear_angle_L_basevec") == "higher than usual"


def test_the_difference_from_a_baseline_wraps():
    """179 and -179 are two degrees apart. Plain subtraction says 358, which would
    report a dog standing perfectly still as having changed enormously."""
    assert vs_own(179.0, -179.0, 13.0, name="head_tilt") == "about the same as usual"
    assert vs_own(179.0, -179.0, 13.0, name="not_an_angle") != "about the same as usual"
