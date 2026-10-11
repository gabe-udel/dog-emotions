"""Which features may carry a timestamp, which may only describe a whole clip.

Driven entirely by outputs/feature_error.json and outputs/feature_snr.json. No feature
is named in this module - change the measurements and the tiers change with them.

The two files disagree about within-clip SNR, and the disagreement is not noise:

    view_yaw     ground truth 1.61    Monte Carlo 0.94
    ear_height   ground truth 0.33    Monte Carlo 1.42

Face features have real ground truth (a feature computed from the prediction and from
the DogFLW annotation, differenced). Body features have only a Monte-Carlo estimate,
which perturbs each landmark independently and so destroys the common-mode cancellation
that difference features rely on - an upper bound on noise, not a measurement.

So ground truth wins wherever it exists, exactly as aggregate.load_noise_floors already
resolves the same conflict. That rule, applied to the current files, promotes view_yaw
and demotes ear_height, and yields the event tier
{head_pitch, tail_height, head_tilt, head_height, tail_tuck, view_yaw}.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ._src import NOT_FEATURES
from .config import ERROR_JSON, SNR_JSON

GROUND_TRUTH = "measured against ground truth"
ESTIMATED = "reliability estimated, not yet verified"

EVENT = "event"
"""Resolves motion inside one clip, so a timestamp is meaningful."""
CLIP = "clip"
"""Separates dogs but not moments; describe the whole clip, never a time."""
UNUSABLE = "unusable"
"""Indistinguishable from its own measurement error."""


@dataclass(frozen=True)
class Reliability:
    name: str
    error_sd: float | None
    """In the feature's own units. None for body features, which have no ground truth."""
    noise_sd: float | None
    """Monte-Carlo upper bound, used where error_sd is absent."""
    within_clip: float
    across_dogs: float
    source: str
    ref_coverage: float | None
    """Coverage measured over the reference clips - context only. The clip actually
    being analysed reports its own."""

    @property
    def tier(self) -> str:
        if self.within_clip >= 1.0:
            return EVENT
        if self.across_dogs >= 1.0:
            return CLIP
        return UNUSABLE

    @property
    def scale(self) -> float | None:
        """The spread to judge an excursion against: the real error where it exists,
        otherwise the Monte-Carlo bound."""
        for v in (self.error_sd, self.noise_sd):
            if v is not None and math.isfinite(v) and v > 0:
                return float(v)
        return None

    @property
    def verified(self) -> bool:
        return self.error_sd is not None


def _finite(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_reliability(error_json: Path = ERROR_JSON,
                     snr_json: Path = SNR_JSON) -> dict[str, Reliability]:
    """Merge the two measurement files into one table, ground truth winning."""
    gt: dict[str, dict] = {}
    mc: dict[str, dict] = {}
    if Path(error_json).exists():
        gt = json.loads(Path(error_json).read_text()).get("features", {})
    if Path(snr_json).exists():
        mc = json.loads(Path(snr_json).read_text()).get("features", {})

    out: dict[str, Reliability] = {}
    for name in sorted(set(gt) | set(mc)):
        g, m = gt.get(name, {}), mc.get(name, {})
        err = _finite(g.get("error_sd"))
        noise = _finite(m.get("noise_sd"))

        # Ground truth wins where it exists; fall back to the Monte-Carlo figure.
        within = _finite(g.get("snr_within_clip"))
        if within is None:
            within = _finite(m.get("snr"))
        across = _finite(g.get("snr_across_dogs"))

        out[name] = Reliability(
            name=name,
            error_sd=err,
            noise_sd=noise,
            within_clip=within if within is not None else 0.0,
            across_dogs=across if across is not None else 0.0,
            source=GROUND_TRUTH if err is not None else ESTIMATED,
            ref_coverage=_finite(m.get("coverage")),
        )
    return out


def tiers(table: dict[str, Reliability],
          live: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Feature names grouped by tier.

    `live` restricts the result to features the pipeline actually produces. Without it
    the answer still includes the six features that were measured and then deleted
    (eyelid_open, eye_width, the two lip_raise variants, mouth_open_lips and
    ear_closeness_basesep) - their rows survive in the measurement files as the record
    of WHY they were dropped, and that record is worth keeping.

    Bookkeeping columns are always excluded: `body_L` has a within-clip SNR of 2.51 and
    would otherwise be promoted to the event tier and reported as if a dog's body length
    changing mid-clip were a finding.
    """
    keep = set(live) if live is not None else None
    out: dict[str, list[str]] = {EVENT: [], CLIP: [], UNUSABLE: []}
    for name, r in sorted(table.items()):
        if name in NOT_FEATURES or (keep is not None and name not in keep):
            continue
        out[r.tier].append(name)
    return out
