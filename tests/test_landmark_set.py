"""The active landmark set, and the indexing hazard it introduces.

While all 46 DogFLW landmarks were predicted, a DogFLW index and a model channel were
the same number, and modules indexed prediction arrays with DogFLW indices freely. With
the eight sliding ear-contour points dropped they are no longer the same, and a stale
`kpts[14]` now silently reads a different landmark rather than failing.

These tests pin the mapping and check that the modules which consume prediction arrays
were converted to channel space.
"""
from __future__ import annotations

import ear_correct
import shape_refine
from keypoint_scheme import (ACTIVE, ACTIVE_NAMES, CH_OF, DOGFLW_NAMES,
                             DROP_EAR_CONTOUR, REGION_OF)

N = len(ACTIVE)


def test_active_set_is_38_of_46() -> None:
    assert N == 38
    assert len(DROP_EAR_CONTOUR) == 8
    assert set(ACTIVE) | set(DROP_EAR_CONTOUR) == set(range(46))
    assert not set(ACTIVE) & set(DROP_EAR_CONTOUR)


def test_active_is_sorted_and_unique() -> None:
    """Channel order must be deterministic; a set literal would not be."""
    assert ACTIVE == sorted(ACTIVE)
    assert len(set(ACTIVE)) == N


def test_ch_of_round_trips() -> None:
    for k, d in enumerate(ACTIVE):
        assert CH_OF[d] == k
        assert ACTIVE[CH_OF[d]] == d


def test_dropped_landmarks_have_no_channel() -> None:
    """The whole hazard in one assertion: a dropped DogFLW index must raise, not
    silently resolve to some other landmark's channel."""
    for d in DROP_EAR_CONTOUR:
        assert d not in CH_OF


def test_only_ear_contour_points_were_dropped() -> None:
    assert all(REGION_OF[d] == "ear" for d in DROP_EAR_CONTOUR)
    assert all("mid" in DOGFLW_NAMES[d] or "low" in DOGFLW_NAMES[d]
               for d in DROP_EAR_CONTOUR)


def test_the_anatomical_ear_points_survive() -> None:
    """Ear orientation is a base->tip vector; those must remain predictable."""
    kept = [n for n in ACTIVE_NAMES if n.startswith("ear")]
    assert kept == ["ear_right_inner_base", "ear_left_inner_base",
                    "ear_right_tip", "ear_left_tip",
                    "ear_right_outer_base", "ear_left_outer_base"]
    for side in ("right", "left"):
        assert f"ear_{side}_inner_base" in kept and f"ear_{side}_tip" in kept


def test_every_non_ear_region_is_untouched() -> None:
    for region, n in (("eye", 8), ("nose", 7), ("mouth", 11), ("muzzle", 4),
                      ("head", 2)):
        assert sum(1 for d in ACTIVE if REGION_OF[d] == region) == n


def test_post_correction_indices_are_channels_not_dogflw_indices() -> None:
    """Both modules index prediction arrays, so their constants must be channels and
    must fit inside the array. A DogFLW index of 45 would be out of range for 38
    channels; one of 14 would quietly point at the wrong landmark."""
    for name, idx in (("ear_correct.EAR", ear_correct.EAR),
                      ("ear_correct.REL", ear_correct.REL),
                      ("shape_refine.SOURCE", shape_refine.SOURCE),
                      ("shape_refine.TARGET", shape_refine.TARGET)):
        assert idx, f"{name} is empty"
        assert max(idx) < N, f"{name} indexes past {N} channels: {max(idx)}"
        assert min(idx) >= 0


def test_shape_model_targets_the_head_top_channels() -> None:
    assert shape_refine.TARGET == [CH_OF[14], CH_OF[15]]
    assert shape_refine.TARGET_NAMES == ["head_top_right", "head_top_left"]


def test_shape_model_sources_exclude_ear_and_head() -> None:
    for k in shape_refine.SOURCE:
        assert REGION_OF[ACTIVE[k]] not in ("ear", "head")


def test_ear_and_reliable_sets_are_disjoint() -> None:
    assert not set(ear_correct.EAR) & set(ear_correct.REL)
    assert len(ear_correct.EAR) == 6


def test_renderer_contours_are_in_channel_space() -> None:
    """draw.FACE_CONTOURS is written in DogFLW indices; the Renderer must translate."""
    from draw import Renderer
    r = Renderer(["nose"] * 39)
    for seg in r.face_contours:
        assert max(seg) < N, f"contour segment indexes past {N} channels: {seg}"
