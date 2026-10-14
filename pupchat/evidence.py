"""Turn measurements into a description a language model can reason over.

Three rules this module exists to enforce, all of them about not overclaiming:

**No emotion words.** The evidence says where the ears were, never what the dog felt.
Interpretation is the model's job, and handing it a conclusion dressed as a measurement
would make the whole pipeline circular. `config.BANNED_WORDS` is the list; a test checks
real output against it.

**No timestamp a feature cannot support.** A feature whose within-clip SNR is below 1
moved less, inside the clip, than its own measurement error. Giving it a time would be
inventing a moment. Those features describe the clip as a whole or not at all.

**Nothing silently missing.** A feature that could not be measured is listed with a
reason, so the model sees the hole rather than assuming the absence means normality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._src import ANGULAR_FEATURES, NOT_FEATURES, smooth, window_frames, wrap
from .config import BANNED_WORDS, EvidenceConfig
from .reference import describe, phrase
from .reliability import CLIP, EVENT, UNUSABLE, Reliability


@dataclass(frozen=True)
class Phrase:
    """How one feature is said in English.

    `low`/`high` are used only where zero means something physical - ear height is
    measured from eye level, head pitch from the body axis - so "below" is a fact about
    the geometry rather than a judgement about the dog. Features with no meaningful
    zero (a length, a ratio) carry no direction words and are reported as a number.
    """

    label: str
    unit: str = ""
    low: str = ""
    high: str = ""
    decimals: int = 2

    def value(self, v: float) -> str:
        return f"{v:.{self.decimals}f}{self.unit}"

    def direction(self, v: float) -> str:
        if not (self.low and self.high):
            return ""
        return self.high if v > 0 else self.low


PHRASES: dict[str, Phrase] = {
    # --- event tier: a meaningful zero in every case -------------------------------
    "head_pitch":    Phrase("nose angle relative to the body axis", "°",
                            "down", "up", 0),
    "head_tilt":     Phrase("head roll relative to level", "°",
                            "rolled one way", "rolled the other", 0),
    "head_height":   Phrase("nose height above the ground line", " body-lengths"),
    "tail_height":   Phrase("tail angle away from the spine line", "°",
                            "below the spine line", "above the spine line", 0),
    "tail_tuck":     Phrase("tail-tip distance from the hips", " body-lengths"),
    "view_yaw":      Phrase("how square the face is to the camera", ""),
    # --- clip tier -----------------------------------------------------------------
    "ear_height":    Phrase("ear tips relative to eye level", " face-widths",
                            "below eye level", "above eye level"),
    "ear_angle_L_eyeframe": Phrase("left ear tip angle from the skull axis", "°",
                                   decimals=0),
    "ear_angle_R_eyeframe": Phrase("right ear tip angle from the skull axis", "°",
                                   decimals=0),
    "ear_angle_L_basevec": Phrase("left ear base-to-tip angle", "°", decimals=0),
    "ear_angle_R_basevec": Phrase("right ear base-to-tip angle", "°", decimals=0),
    "ear_len_L":     Phrase("left ear length", " face-widths"),
    "ear_len_R":     Phrase("right ear length", " face-widths"),
    "ear_closeness_iod": Phrase("gap between the ear tips, in eye-widths", ""),
    "mouth_open_jaw": Phrase("jaw opening", " face-widths"),
    "view_dir":      Phrase("head turn away from the camera", "",
                            "turned one way", "turned the other"),
    # Below the noise floor on every clip measured so far, so these appear only in the
    # unavailable list - but they still need a readable name there.
    "tail_lateral":  Phrase("tail swing to the side", " body-lengths"),
    "tail_len":      Phrase("visible tail length", " body-lengths"),
    "posture":       Phrase("front-to-rear height difference", " body-lengths"),
    "motion_energy": Phrase("overall movement", " body-lengths per second"),
}

# Which part of the dog a feature needs, so an unavailable one can say what was missing
# rather than only that it failed.
_MISSING_REASON = (
    ("tail_", "the tail was not tracked for enough of the clip"),
    ("ear_", "the face was not clearly visible for enough of the clip"),
    ("mouth_", "the face was not clearly visible for enough of the clip"),
    ("view_", "the face was not clearly visible for enough of the clip"),
    ("head_", "the body was not fully visible for enough of the clip"),
    ("posture", "the body was not fully visible for enough of the clip"),
)


def _reason_for(name: str) -> str:
    for prefix, reason in _MISSING_REASON:
        if name.startswith(prefix):
            return reason
    return "it could not be measured for enough of the clip"


def _phrase(name: str) -> Phrase:
    return PHRASES.get(name, Phrase(name.replace("_", " ")))


# ---------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------

def detect_events(series: np.ndarray, rel: Reliability, fps: float,
                  cfg: EvidenceConfig) -> list[dict]:
    """Excursions from the clip's own baseline, thresholded by the feature's OWN error.

    The threshold is `event_sigmas` x the measured error_sd, not a round number chosen
    to look reasonable. A feature with a large error needs a large excursion before
    anything is reported, which is the correct behaviour and falls out of the
    measurement rather than being imposed on it.
    """
    scale = rel.scale
    if scale is None or not np.isfinite(series).any():
        return []
    angular = rel.name in ANGULAR_FEATURES
    k = window_frames(100.0, fps)
    sm = smooth(np.asarray(series, dtype=float), k, angular)

    good = np.isfinite(sm)
    if good.sum() < 5:
        return []
    base = float(np.median(sm[good]))
    dev = wrap(sm - base) if angular else sm - base

    thr = cfg.event_sigmas * scale
    hot = good & (np.abs(dev) > thr)
    min_len = max(1, int(round(cfg.event_min_ms * fps / 1000.0)))

    events: list[dict] = []
    i = 0
    n = len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            seg = dev[i:j + 1]
            peak = int(np.nanargmax(np.abs(seg)))
            events.append({
                "feature": rel.name,
                "t_start": round(i / fps, 2),
                "t_end": round((j + 1) / fps, 2),
                "t_peak": round((i + peak) / fps, 2),
                "delta": round(float(seg[peak]), 3),
                "baseline": round(base, 3),
                "threshold": round(thr, 3),
                "source": rel.source,
            })
        i = j + 1

    events.sort(key=lambda e: -abs(e["delta"]))
    return events[:cfg.max_events_per_feature]


# ---------------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------------

def build_evidence(frames: list[dict[str, float]], clip: dict[str, float],
                   reliability: dict[str, Reliability],
                   cfg: EvidenceConfig | None = None,
                   reference: dict | None = None,
                   dog_baseline: dict | None = None) -> dict:
    """Measurements -> a structured, tier-aware description.

    `frames` is the per-frame rows; `clip` is what aggregate() produced from them. Both
    are needed: a timestamp cannot come from the clip row, and a coverage figure should
    not be recomputed when aggregate already measured it.
    """
    cfg = cfg or EvidenceConfig()
    fps = float(clip.get("fps") or 0.0) or 1.0
    n_frames = int(clip.get("n_frames") or len(frames))

    # The live feature set is whatever aggregate() reported coverage for. Deriving it
    # from the data means the six deleted features cannot reappear here just because
    # their rows survive in the measurement files.
    live = sorted({k[:-len("__coverage")] for k in clip if k.endswith("__coverage")}
                  - set(NOT_FEATURES))

    events: list[dict] = []
    clip_level: list[dict] = []
    unavailable: list[dict] = []

    for name in live:
        rel = reliability.get(name)
        cov = clip.get(f"{name}__coverage")
        cov = float(cov) if cov is not None else 0.0
        p = _phrase(name)

        if rel is None:
            unavailable.append({"feature": name, "label": p.label,
                                "reason": "no reliability measurement exists for it"})
            continue
        if cov < cfg.min_coverage:
            unavailable.append({
                "feature": name, "label": p.label, "coverage": round(cov, 3),
                "reason": f"{_reason_for(name)} (visible {cov * 100:.0f}% of frames)"})
            continue
        if rel.tier == UNUSABLE:
            unavailable.append({
                "feature": name, "label": p.label, "coverage": round(cov, 3),
                "reason": "below the noise floor - it moved less than its own "
                          "measurement error"})
            continue

        level = clip.get(f"{name}__level")
        if level is None or not math.isfinite(float(level)):
            unavailable.append({
                "feature": name, "label": p.label, "coverage": round(cov, 3),
                "reason": "too few usable frames to summarise"})
            continue
        level = float(level)
        spread = clip.get(f"{name}__spread")

        entry = {
            "feature": name, "label": p.label, "level": round(level, 4),
            "spread": round(float(spread), 4) if spread is not None else None,
            "coverage": round(cov, 3), "source": rel.source,
            "value_text": p.value(level), "direction": p.direction(level),
            "across_dogs": round(rel.across_dogs, 2),
        }

        # A number with nothing to judge it against can only be described vaguely. Both
        # comparisons are computed, and each is simply absent when it cannot be made -
        # never filled in with a reassuring "typical".
        comparison = describe(name, level, reference=reference,
                              dog_baseline=dog_baseline,
                              error_sd=(rel.error_sd or rel.noise_sd))
        if comparison:
            entry["comparison"] = comparison

        if rel.tier == EVENT:
            series = np.array([r.get(name, np.nan) for r in frames], dtype=float)
            found = detect_events(series, rel, fps, cfg)
            entry["events"] = found
            events.extend(found)
            clip_level.append(entry)
        elif rel.tier == CLIP:
            clip_level.append(entry)

    # The oscillation gate is a separate fact from coverage: the tail can be perfectly
    # visible and still produce no wag rate, because the amplitude did not clear the
    # feature's measured error. Reported every time it refuses, so its absence is never
    # read as "the dog did not wag".
    wag = clip.get("wag_rate_hz")
    if wag is None or not math.isfinite(float(wag)):
        amp = clip.get("wag_amp")
        detail = (f" measured swing {float(amp):.3f} body-lengths," if amp is not None
                  and math.isfinite(float(amp)) else "")
        unavailable.append({
            "feature": "wag_rate_hz", "label": "tail wag rate",
            "reason": f"the oscillation gate refused it -{detail} below the level this "
                      f"pipeline can separate from measurement error"})

    events.sort(key=lambda e: e["t_start"])

    return {
        "clip": {
            "n_frames": n_frames,
            "fps": round(fps, 2),
            "duration_s": round(n_frames / fps, 2) if fps else None,
        },
        "events": events,
        "clip_level": clip_level,
        "unavailable": unavailable,
        "reliability_note": {
            "face": "measured against ground truth on 574 annotated images",
            "body": "reliability estimated from a noise model, not yet verified "
                    "against ground truth",
        },
    }


# ---------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------

def _group_unavailable(unavailable: list[dict]) -> list[str]:
    """One line per reason, listing the features that share it.

    Compression, not omission. Ten separate bullets will not fit a 300-word budget, and
    the first version of this function trimmed them off the end - which silently removed
    the entire record of what could NOT be measured, including the wag rate. The
    unavailable list is the honesty mechanism; it is the last thing that may be cut, not
    the first. Grouping buys the space instead.
    """
    by_reason: dict[str, list[str]] = {}
    for u in unavailable:
        by_reason.setdefault(u["reason"], []).append(u["label"])
    return [f"- {', '.join(labels)}: {reason}."
            for reason, labels in by_reason.items()]


def evidence_to_text(evidence: dict, cfg: EvidenceConfig | None = None,
                     dog_name: str = "") -> str:
    """Plain English, under ~300 words, no emotion words, no unearned timestamps.

    Section order is also the priority order. If the budget binds, clip-level
    descriptions are dropped lowest-value first (weakest across-dog separation goes
    first). Events, the unavailable list and the reliability note are never dropped -
    each of those is a statement about what the pipeline does or does not know, and a
    reader cannot tell a trimmed one from an absent one.
    """
    cfg = cfg or EvidenceConfig()
    c = evidence["clip"]

    head = [f"Automated body-language measurements from {c['n_frames']} frames "
            f"({c['duration_s']}s at {c['fps']} fps). These describe geometry only."]

    ev = evidence["events"]
    moments: list[str] = []
    if ev:
        moments.append("\nMoments (only these features resolve change within a clip):")
        for e in ev:
            p = _phrase(e["feature"])
            d = p.direction(e["delta"]) or ("rose" if e["delta"] > 0 else "fell")
            moments.append(
                f"- {p.label}: {d} by {abs(e['delta']):.{p.decimals}f}{p.unit} from its "
                f"clip baseline at {e['t_peak']}s ({e['t_start']}-{e['t_end']}s).")
    else:
        moments.append("\nMoments: none. No feature able to resolve change within a "
                       "clip moved beyond its own measurement error.")

    # A line that carries a comparison goes first, then strongest separation.
    #
    # The budget drops clip-level lines from the end, and without this it dropped
    # exactly the informative ones: on a real clip the two features with a population
    # band ranked 7th and 10th of 11, so both were trimmed while bare numbers survived.
    # The prompt forbids repeating a raw number, so a line with nothing to compare it
    # against is close to unusable - it should be the first to go, not the last.
    steady = sorted((e for e in evidence["clip_level"] if not e.get("events")),
                    key=lambda e: (0 if e.get("comparison") else 1,
                                   -(e.get("across_dogs") or 0.0)))
    body: list[str] = []
    for e in steady:
        bits = f"- {e['label']}: {e['value_text']}"
        if e["direction"]:
            bits += f" ({e['direction']})"
        # The comparison is the point of the line: a number on its own can only be
        # described vaguely, and the prompt forbids repeating the number itself.
        said = phrase(e["feature"], e.get("comparison") or {}, dog_name)
        if said:
            bits += f" - {said}"
        bits += f", visible {e['coverage'] * 100:.0f}% of frames."
        body.append(bits)

    tail = []
    if evidence["unavailable"]:
        tail.append("\nNot measurable in this clip:")
        tail.extend(_group_unavailable(evidence["unavailable"]))
    tail.append(
        "\nReliability: face measurements are measured against ground truth on 574 "
        "annotated images. Body measurements (head, tail, posture) have their "
        "reliability estimated, not yet verified - treat those as weaker evidence.")

    section_head = "\nAcross the clip (no timing available for these):"
    omitted_note = "- ({n} further clip-level measurements omitted for length; " \
                   "all are in the JSON.)"

    fixed = head + moments + tail
    budget = cfg.max_words - len(" ".join(fixed).split())
    if body:
        # Charge for the section header and for the worst case of the omission note
        # BEFORE filling, or the last line fits on paper and overflows in practice.
        budget -= len(section_head.split())
        budget -= len(omitted_note.format(n=len(body)).split())

    kept: list[str] = []
    for line in body:
        n = len(line.split())
        if n > budget:
            break
        kept.append(line)
        budget -= n

    out = head + moments
    if kept:
        out += [section_head] + kept
    if len(kept) < len(body):
        out.append(omitted_note.format(n=len(body) - len(kept)))
    return "\n".join(out + tail)


def contains_banned(text: str, banned: tuple[str, ...] = BANNED_WORDS) -> list[str]:
    """Which forbidden words appear. Whole-word matching, so 'contentment' is caught
    but 'content-type' would not trip on a substring of an unrelated word."""
    import re
    low = text.lower()
    return [w for w in banned if re.search(rf"\b{re.escape(w)}\b", low)]
