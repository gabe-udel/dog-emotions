"""The evidence layer's promises, checked against fixtures rather than a real video.

What these tests defend, in order of how badly it would hurt to get wrong:

1. A feature that cannot resolve motion never gets a timestamp. Inventing a moment is
   the worst thing this layer could do, because it reads as precision.
2. A feature that was barely visible is reported as unavailable WITH a reason, rather
   than quietly omitted - an omission reads as normality.
3. The wag rate is reported as unavailable whenever the oscillation gate refuses, so
   its absence is never mistaken for "the dog did not wag".
4. No emotion word reaches the text. The model does the interpreting; handing it a
   conclusion dressed as a measurement would make the loop circular.

No test here touches the network, the API key, or torch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pupchat.config import BANNED_WORDS, EvidenceConfig
from pupchat.evidence import (build_evidence, contains_banned, detect_events,
                              evidence_to_text)
from pupchat.reliability import (CLIP, EVENT, UNUSABLE, Reliability,
                                 load_reliability, tiers)

FPS = 10.0
N = 60


def rel(name, *, within, across=5.0, err=1.0, verified=True):
    return Reliability(name=name, error_sd=err if verified else None,
                       noise_sd=err, within_clip=within, across_dogs=across,
                       source=("measured against ground truth" if verified
                               else "reliability estimated, not yet verified"),
                       ref_coverage=0.9)


@pytest.fixture
def table():
    return {
        # resolves motion -> may carry timestamps
        "head_pitch": rel("head_pitch", within=17.0, err=5.0, verified=False),
        # separates dogs but not moments -> clip-level only
        "ear_height": rel("ear_height", within=0.33, across=49.9, err=0.036),
        # below its own error -> unusable
        "tail_lateral": rel("tail_lateral", within=0.0, across=0.0, err=0.097,
                            verified=False),
        # fine in principle, but barely visible in this clip
        "mouth_open_jaw": rel("mouth_open_jaw", within=0.0, across=3.76, err=0.02),
    }


def make_clip(**over):
    clip = {"fps": FPS, "n_frames": N}
    for name, (cov, level, spread) in {
        "head_pitch": (0.95, -5.0, 4.0),
        "ear_height": (0.90, -0.18, 0.01),
        "tail_lateral": (0.80, 0.63, 0.03),
        "mouth_open_jaw": (0.10, 0.20, 0.00),
    }.items():
        clip[f"{name}__coverage"] = cov
        clip[f"{name}__level"] = level
        clip[f"{name}__spread"] = spread
    clip.update(over)
    return clip


def make_frames(spike_at=None):
    """Flat series, optionally with a large excursion in head_pitch."""
    rows = []
    for i in range(N):
        hp = -5.0
        if spike_at is not None and spike_at <= i < spike_at + 10:
            hp = 40.0
        rows.append({"head_pitch": hp, "ear_height": -0.18,
                     "tail_lateral": 0.63, "mouth_open_jaw": 0.20})
    return rows


# --------------------------------------------------------------- the tier rule

def test_event_tier_is_derived_and_matches_the_measured_files():
    """The six timestamp-eligible features come out of the measurements, not a list."""
    t = load_reliability()
    assert set(tiers(t)[EVENT]) == {"head_pitch", "tail_height", "head_tilt",
                                    "head_height", "tail_tuck", "view_yaw"}


def test_ground_truth_beats_the_noise_model_where_both_exist():
    """The two files disagree on exactly these two, and ground truth has to win:
    view_yaw is 1.61 by ground truth and 0.94 by Monte Carlo, ear_height the reverse."""
    t = load_reliability()
    assert t["view_yaw"].tier == EVENT
    assert t["ear_height"].tier == CLIP


def test_bookkeeping_columns_are_never_promoted():
    """body_L has a within-clip SNR of 2.51 and would otherwise be reported as an
    event, i.e. as a dog's body length changing mid-clip."""
    t = load_reliability()
    assert t["body_L"].within_clip >= 1.0
    for group in tiers(t).values():
        assert "body_L" not in group


# --------------------------------------------------------------- no false timestamps

def test_low_snr_feature_never_produces_a_timestamp(table):
    ev = build_evidence(make_frames(spike_at=20), make_clip(), table)
    timed = {e["feature"] for e in ev["events"]}
    assert "ear_height" not in timed
    entry = next(e for e in ev["clip_level"] if e["feature"] == "ear_height")
    assert "events" not in entry


def test_a_real_excursion_on_a_high_snr_feature_does_produce_one(table):
    ev = build_evidence(make_frames(spike_at=20), make_clip(), table)
    hp = [e for e in ev["events"] if e["feature"] == "head_pitch"]
    assert hp, "a 45-degree swing against a 5-degree error should register"
    assert 1.5 <= hp[0]["t_start"] <= 2.5


def test_a_flat_series_produces_no_events(table):
    ev = build_evidence(make_frames(spike_at=None), make_clip(), table)
    assert ev["events"] == []


def test_excursion_smaller_than_the_feature_error_is_ignored():
    """The threshold is the feature's OWN measured error, so a 3-degree wobble on a
    5-degree error is not an event."""
    r = rel("head_pitch", within=17.0, err=5.0, verified=False)
    s = np.full(N, -5.0)
    s[20:30] = -2.0
    assert detect_events(s, r, FPS, EvidenceConfig()) == []


def test_text_never_times_a_clip_level_feature(table):
    text = evidence_to_text(build_evidence(make_frames(20), make_clip(), table))
    clip_section = text.split("Across the clip")[-1].split("Not measurable")[0]
    assert "s," not in clip_section.replace("frames,", "")


# --------------------------------------------------------------- coverage

def test_low_coverage_feature_is_unavailable_with_a_reason(table):
    ev = build_evidence(make_frames(), make_clip(), table)
    un = {u["feature"]: u for u in ev["unavailable"]}
    assert "mouth_open_jaw" in un
    assert "10%" in un["mouth_open_jaw"]["reason"]
    assert "face" in un["mouth_open_jaw"]["reason"]
    assert all(e["feature"] != "mouth_open_jaw" for e in ev["clip_level"])


def test_coverage_threshold_is_configurable(table):
    cfg = EvidenceConfig(min_coverage=0.05)
    ev = build_evidence(make_frames(), make_clip(), table, cfg)
    assert any(e["feature"] == "mouth_open_jaw" for e in ev["clip_level"])


def test_below_noise_floor_feature_says_so(table):
    ev = build_evidence(make_frames(), make_clip(), table)
    un = {u["feature"]: u for u in ev["unavailable"]}
    assert "tail_lateral" in un
    assert "noise floor" in un["tail_lateral"]["reason"]


# --------------------------------------------------------------- the wag gate

def test_wag_rate_is_unavailable_when_the_gate_refuses(table):
    ev = build_evidence(make_frames(), make_clip(wag_rate_hz=float("nan"),
                                                 wag_amp=0.044), table)
    un = {u["feature"]: u for u in ev["unavailable"]}
    assert "wag_rate_hz" in un
    assert "0.044" in un["wag_rate_hz"]["reason"]


def test_wag_rate_is_unavailable_when_the_key_is_absent_entirely(table):
    ev = build_evidence(make_frames(), make_clip(), table)
    assert any(u["feature"] == "wag_rate_hz" for u in ev["unavailable"])


def test_wag_rate_is_not_listed_when_the_gate_did_answer(table):
    ev = build_evidence(make_frames(), make_clip(wag_rate_hz=3.4, wag_amp=0.22), table)
    assert all(u["feature"] != "wag_rate_hz" for u in ev["unavailable"])


# --------------------------------------------------------------- the words

def test_evidence_text_contains_no_emotion_words(table):
    text = evidence_to_text(build_evidence(make_frames(20), make_clip(), table))
    assert contains_banned(text) == []


def test_the_word_check_actually_catches_things():
    assert "anxious" in contains_banned("the dog looks anxious here")
    assert "relaxed" in contains_banned("ears relaxed and low")
    assert contains_banned("ear tips 0.18 face-widths below eye level") == []


def test_banned_list_covers_interpretation_not_only_emotion():
    for w in ("relaxed", "alert", "submissive", "confident", "playful"):
        assert w in BANNED_WORDS


# --------------------------------------------------------------- the text itself

def test_text_stays_under_the_word_budget(table):
    cfg = EvidenceConfig(max_words=300)
    text = evidence_to_text(build_evidence(make_frames(20), make_clip(), table), cfg)
    assert len(text.split()) <= cfg.max_words


def test_text_states_both_reliability_labels(table):
    text = evidence_to_text(build_evidence(make_frames(20), make_clip(), table))
    assert "measured against ground truth" in text
    assert "reliability estimated, not yet verified" in text


def test_evidence_reports_clip_duration(table):
    ev = build_evidence(make_frames(), make_clip(), table)
    assert ev["clip"]["duration_s"] == pytest.approx(N / FPS)


def test_unknown_feature_does_not_crash_the_builder(table):
    clip = make_clip()
    clip["something_new__coverage"] = 0.9
    clip["something_new__level"] = 1.0
    ev = build_evidence(make_frames(), clip, table)
    un = {u["feature"]: u for u in ev["unavailable"]}
    assert "something_new" in un
    assert "no reliability measurement" in un["something_new"]["reason"]


# --------------------------------------------------------------- the trim bug
#
# The first version of evidence_to_text trimmed whole bullets off the END of the text
# when it ran over budget. "Not measurable" is the second-to-last section, so on a real
# clip it dropped 8 of 10 unavailable entries - including the wag rate, which must
# always be stated. A reader cannot tell a trimmed entry from one that never existed,
# which makes that the most dangerous possible truncation order.

def many_features_table():
    t = {}
    for i in range(14):
        t[f"ear_angle_{i}"] = rel(f"ear_angle_{i}", within=0.0, across=20.0, err=1.0)
    for i in range(10):
        t[f"tail_part_{i}"] = rel(f"tail_part_{i}", within=0.0, across=0.0, err=1.0,
                                  verified=False)
    return t


def many_features_clip():
    clip = {"fps": FPS, "n_frames": N}
    for i in range(14):
        clip[f"ear_angle_{i}__coverage"] = 0.9
        clip[f"ear_angle_{i}__level"] = 12.0 + i
    for i in range(10):
        clip[f"tail_part_{i}__coverage"] = 0.8
        clip[f"tail_part_{i}__level"] = 1.0
    return clip


def test_unavailable_entries_always_reach_the_text_even_over_budget():
    """Every unavailable feature must be named, however tight the budget."""
    t, clip = many_features_table(), many_features_clip()
    ev = build_evidence([{} for _ in range(N)], clip, t)
    text = evidence_to_text(ev, EvidenceConfig(max_words=300))
    assert len(ev["unavailable"]) >= 10
    for u in ev["unavailable"]:
        assert u["label"] in text, f"{u['label']} was dropped from the text"


def test_wag_rate_survives_a_tight_budget():
    t, clip = many_features_table(), many_features_clip()
    ev = build_evidence([{} for _ in range(N)], clip, t)
    text = evidence_to_text(ev, EvidenceConfig(max_words=200))
    assert "tail wag rate" in text


def test_reliability_note_survives_a_tight_budget():
    t, clip = many_features_table(), many_features_clip()
    ev = build_evidence([{} for _ in range(N)], clip, t)
    text = evidence_to_text(ev, EvidenceConfig(max_words=180))
    assert "measured against ground truth" in text
    assert "reliability estimated, not yet verified" in text


def test_dropped_clip_level_lines_are_announced_not_silent():
    t, clip = many_features_table(), many_features_clip()
    ev = build_evidence([{} for _ in range(N)], clip, t)
    text = evidence_to_text(ev, EvidenceConfig(max_words=200))
    assert "omitted for length" in text


def test_clip_level_lines_are_dropped_weakest_first():
    """When the budget binds, the measurement that separates dogs best is the one
    that stays."""
    t = {"strong": rel("strong", within=0.0, across=50.0, err=0.01),
         "weak": rel("weak", within=0.0, across=1.5, err=0.01)}
    clip = {"fps": FPS, "n_frames": N,
            "strong__coverage": 0.9, "strong__level": 1.0,
            "weak__coverage": 0.9, "weak__level": 2.0,
            "wag_rate_hz": 3.0}
    ev = build_evidence([{} for _ in range(N)], clip, t)
    text = evidence_to_text(ev, EvidenceConfig(max_words=95))
    # Match the bullet, not the bare word: "weak" is a substring of "weaker evidence"
    # in the reliability note, which is always present.
    assert "- strong:" in text
    assert "- weak:" not in text
    assert "omitted for length" in text


def test_event_delta_uses_the_feature_decimals():
    """view_yaw runs around 0.25 with an error of 0.0096, so a real excursion printed
    at one decimal place reads as 'rose by 0.0' - a reported event with no magnitude."""
    t = {"view_yaw": rel("view_yaw", within=1.6, across=20.0, err=0.0096)}
    s = np.full(N, 0.25)
    s[20:35] = 0.31
    frames = [{"view_yaw": float(v)} for v in s]
    clip = {"fps": FPS, "n_frames": N, "view_yaw__coverage": 1.0,
            "view_yaw__level": 0.25, "wag_rate_hz": 3.0}
    text = evidence_to_text(build_evidence(frames, clip, t))
    assert "by 0.0 " not in text and "by 0.0." not in text
