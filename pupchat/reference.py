"""Turning a measurement into a comparison.

"His ears sat 0.31 face-widths above eye level" is a number with nothing to judge it
against, so the model can only describe it vaguely. Two references fix that, and both are
computed here in Python rather than by the model, so they are exact and testable.

**Against other dogs** - percentiles from the DogFLW hand-annotated landmarks. FACE ONLY,
and that is not an oversight. Every body feature divides by body length, which runs
withers to tail base, and StanfordExtra leaves the withers slot empty in all 2,545
overlapping images (src/check_body_accuracy.py:21-23). There is no calibrated population
range for tail, posture or head height, and inventing one would be exactly the kind of
fabricated measurement this project refuses elsewhere.

**Against this dog's usual** - from baseline clips the owner saved. This is the better
comparison for body features anyway: same dog, same camera, so the body-length error is
common-mode and largely cancels, where an absolute band would inherit it whole.

Every phrase describes position relative to a reference. None of them describes a feeling:
a test asserts every string this module can emit is clear of `config.BANNED_WORDS`, because
handing the model a conclusion dressed as a measurement is what makes the loop circular.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ._src import ANGULAR_FEATURES, wrap
from .config import OUT_DIR

REFERENCE_FILE = OUT_DIR / "reference.json"

OWN_SIGMAS = 2.0
"""How far from a dog's own baseline counts as different.

Two times the feature's measured error. Inside that, a tracking wobble would read as
"his tail is lower today", which is a claim about the dog made out of noise.
"""

NO_POPULATION_BAND = set(ANGULAR_FEATURES)
"""Angles get no "compared with other dogs" band, and this is not caution.

Measured, on a real clip against the built reference: ear_angle_L_basevec runs -115..+92
at runtime (median -96) where DogFLW's ground truth runs +29..+172 (median +144), and
ear_angle_R_basevec is offset by the same amount. Both ears, consistently - that is a
difference of convention between the two, not a difference between dogs, and nothing
about this pipeline would have made it visible except comparing the distributions.

A percentile band on a wrapped quantity is unsound in principle anyway: 359 and 1 are
two degrees apart and the arithmetic says 358. CLAUDE.md s10 records the same lesson from
head_tilt, whose apparent 145 degree spread was 1.23 degrees once wrap was handled.

Comparing a dog with ITS OWN baseline is unaffected: both sides come from this pipeline
in the same convention, and `vs_own` wraps the difference.
"""


def load_reference(path: Path = REFERENCE_FILE) -> dict:
    """Percentiles per feature, or an empty dict if the file was never built.

    Missing is a normal state, not an error: the comparison is simply left out, which is
    the correct behaviour when there is nothing to compare against.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("features", {}) if isinstance(data, dict) else {}


def band(value: float, pct: dict) -> str | None:
    """Where a value sits among the reference dogs, in words.

    The cuts are the 10th/25th/75th/90th percentiles, so "typical" covers the middle
    half. Returns None when the reference is missing or the value is not a number - the
    field is then omitted rather than filled with a reassuring "typical".
    """
    if value is None or not isinstance(value, (int, float)) or math.isnan(value):
        return None
    if not pct or any(k not in pct for k in ("p10", "p25", "p75", "p90")):
        return None
    if value < pct["p10"]:
        return "well below most"
    if value < pct["p25"]:
        return "lower than most"
    if value <= pct["p75"]:
        return "typical"
    if value <= pct["p90"]:
        return "higher than most"
    return "well above most"


def vs_own(value: float, baseline: float | None, error_sd: float | None,
           sigmas: float = OWN_SIGMAS, name: str = "") -> str | None:
    """How this clip compares with the dog's own saved baseline.

    Refuses to call a difference real unless it clears `sigmas` times the feature's own
    measured error. Without that, normal tracking noise becomes a statement about the
    dog, and the owner has no way to tell the two apart.
    """
    if value is None or baseline is None:
        return None
    if not isinstance(value, (int, float)) or math.isnan(value):
        return None
    if not isinstance(baseline, (int, float)) or math.isnan(baseline):
        return None
    if not error_sd or error_sd <= 0 or math.isnan(error_sd):
        return None
    diff = value - baseline
    if name in ANGULAR_FEATURES:
        # 359 and 1 are two degrees apart; plain subtraction says 358.
        diff = wrap(diff)
    if abs(diff) < sigmas * error_sd:
        return "about the same as usual"
    return "higher than usual" if diff > 0 else "lower than usual"


def describe(name: str, value: float, *, reference: dict, dog_baseline: dict | None,
             error_sd: float | None) -> dict:
    """Both comparisons for one feature, each omitted when it cannot be made.

    The population half carries `n` with it, so the evidence can say "compared with
    1,214 photographed dogs" rather than implying a measurement from video.
    """
    out: dict = {}
    pct = None if name in NO_POPULATION_BAND else (reference or {}).get(name)
    word = band(value, pct or {})
    if word:
        out["vs_dogs"] = word
        out["vs_dogs_n"] = pct.get("n")

    own = vs_own(value, (dog_baseline or {}).get(name), error_sd, name=name)
    if own:
        out["vs_own"] = own
    return out


def phrase(name: str, comparison: dict, dog_name: str = "") -> str:
    """The comparison as a clause to append to an evidence line."""
    bits = []
    if comparison.get("vs_dogs"):
        n = comparison.get("vs_dogs_n")
        bits.append(f"{comparison['vs_dogs']}"
                    + (f" of {n:,} photographed dogs" if n else " dogs"))
    if comparison.get("vs_own"):
        who = f"{dog_name}'s" if dog_name else "their"
        bits.append(f"{comparison['vs_own']}".replace(
            "usual", f"{who} usual", 1))
    return "; ".join(bits)


def baseline_levels(clip_rows: list[dict]) -> dict[str, float]:
    """One dog's usual, from the clips they saved as baselines.

    The median across clips, not the mean: a baseline the owner filmed badly should not
    drag the reference, and with two or three clips the median is the more forgiving of
    the two.
    """
    out: dict[str, float] = {}
    if not clip_rows:
        return out
    names = {k for row in clip_rows for k, v in row.items()
             if isinstance(v, (int, float)) and not math.isnan(v)}
    for name in sorted(names):
        values = sorted(row[name] for row in clip_rows
                        if isinstance(row.get(name), (int, float))
                        and not math.isnan(row[name]))
        if values:
            out[name] = values[len(values) // 2]
    return out
