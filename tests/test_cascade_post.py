"""The quarantined corrections must actually run when the config enables them.

This exists because of a real bug: `ear_correct` and `shape_refine` were applied only
inside run_video.py, so `evaluate_face.py --shape-refine` accepted the flag and silently
did nothing. The ablation came back byte-identical to the run without it, which looks
exactly like "the correction has no effect" rather than "the correction never ran".

A silently-ignored flag is worse than a broken one: it produces a confident, wrong
measurement. These tests assert the wiring, not the correction quality.

They construct a Cascade without __init__ so no checkpoint or DeepLabCut model is
needed - the thing under test is whether config reaches behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from cascade import Cascade
from faceconfig import CascadeConfig, PostConfig


class _Spy:
    """Stands in for EarCorrector / Refiner and records that it was called."""

    def __init__(self, tag: float) -> None:
        self.calls = 0
        self.tag = tag

    def apply(self, kpts: np.ndarray) -> int:
        self.calls += 1
        kpts[0, 0] += self.tag
        return 1


def _bare(cfg: CascadeConfig, ear=None, refiner=None) -> Cascade:
    c = object.__new__(Cascade)
    c.cfg = cfg
    c.ear_corrector = ear
    c.refiner = refiner
    return c


def test_apply_post_runs_both_when_enabled() -> None:
    ear, refiner = _Spy(1.0), _Spy(10.0)
    c = _bare(CascadeConfig(post=PostConfig(ear_correct=True, shape_refine=True)),
              ear, refiner)
    out = c.apply_post(np.zeros((46, 3)))
    assert (ear.calls, refiner.calls) == (1, 1)
    assert out[0, 0] == pytest.approx(11.0)


def test_both_corrections_are_on_by_default() -> None:
    """They were quarantined OFF after the rebuild and turned back on only by
    measurement on the validation split - ear 0.0700 -> 0.0554, head 0.0443 -> 0.0277,
    all-46 NME 0.0346 -> 0.0294. This pins the decision so a later edit that silently
    flips a default has to change a test that says why."""
    cfg = CascadeConfig()
    assert cfg.post.ear_correct is True
    assert cfg.post.shape_refine is True


def test_apply_post_is_a_no_op_when_nothing_is_loaded() -> None:
    """The safety property behind the flags: with no correctors, predictions pass
    through untouched rather than being silently altered."""
    c = _bare(CascadeConfig(post=PostConfig(ear_correct=False, shape_refine=False)))
    before = np.arange(46 * 3, dtype=float).reshape(46, 3)
    out = c.apply_post(before.copy())
    assert np.array_equal(out, before)


@pytest.mark.parametrize("ear_on,shape_on", [(True, False), (False, True)])
def test_each_flag_acts_independently(ear_on: bool, shape_on: bool) -> None:
    ear, refiner = _Spy(1.0), _Spy(10.0)
    c = _bare(CascadeConfig(post=PostConfig(ear_correct=ear_on,
                                            shape_refine=shape_on)),
              ear if ear_on else None, refiner if shape_on else None)
    c.apply_post(np.zeros((46, 3)))
    assert ear.calls == int(ear_on)
    assert refiner.calls == int(shape_on)


def test_ear_correction_runs_before_the_shape_model() -> None:
    """Order matters: the shape model derives head_top from the reliable landmarks, so
    ear correction must see untouched inputs rather than derived ones."""
    order: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def apply(self, kpts: np.ndarray) -> int:
            order.append(self.name)
            return 1

    c = _bare(CascadeConfig(post=PostConfig(ear_correct=True, shape_refine=True)),
              Recorder("ear"), Recorder("shape"))
    c.apply_post(np.zeros((46, 3)))
    assert order == ["ear", "shape"]


def test_stage2_is_the_single_place_post_processing_happens() -> None:
    """run_video.py used to apply the corrections itself, which is how the two paths
    drifted. Only cascade.py may call apply_post."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for p in src.glob("*.py"):
        if p.name == "cascade.py":
            continue
        text = p.read_text(encoding="utf-8")
        if "EarCorrector(" in text or "Refiner(" in text:
            # postfit.py legitimately fits them; it does not apply them per frame
            if p.name not in ("postfit.py", "ear_correct.py", "shape_refine.py"):
                offenders.append(p.name)
    assert not offenders, f"post-processing is applied outside cascade.py in {offenders}"
