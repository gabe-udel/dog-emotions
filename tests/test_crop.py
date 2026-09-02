"""Coordinate round-trip tests.

This is the surface where a cascade fails silently: a half-pixel error in the transform
looks exactly like a slightly worse model.

Two layers are tested separately on purpose:

* `CropTransform` is pure arithmetic and must be an EXACT inverse (1e-9). Those tests
  import no DeepLabCut.
* `crop_face` delegates pixel work to DeepLabCut's `top_down_crop`, which rounds box
  corners to integers and pads. Those tests assert agreement in IMAGE pixels, where the
  rounding lives, rather than in crop pixels, where a tiny box magnifies it.

Assumption stated explicitly: boxes are axis-aligned squares. The cascade never produces
a rotated box, so rotation is out of scope; `test_square_box_is_isotropic` pins that down
so a future move to non-square boxes fails here rather than in production.
"""
from __future__ import annotations

import numpy as np
import pytest

from crop import CropTransform, crop_face, jitter_box, transform_for
from facebox import FaceBox
from faceconfig import CropConfig

CFG = CropConfig(size=256)

BOXES = [
    pytest.param(FaceBox(100.0, 100.0, 80.0, "anchors", 11), id="interior"),
    pytest.param(FaceBox(100.5, 60.25, 33.75, "anchors", 9), id="fractional"),
    pytest.param(FaceBox(5.0, 5.0, 60.0, "anchors", 5), id="off-top-left"),
    pytest.param(FaceBox(315.0, 235.0, 90.0, "anchors", 4), id="off-bottom-right"),
    pytest.param(FaceBox(0.0, 120.0, 50.0, "anchors", 3), id="off-left-edge"),
    pytest.param(FaceBox(160.0, 0.0, 44.0, "anchors", 3), id="off-top-edge"),
    pytest.param(FaceBox(160.0, 120.0, 900.0, "fallback", 0), id="larger-than-image"),
    pytest.param(FaceBox(160.0, 120.0, 12.0, "anchors", 3), id="tiny"),
]

TRANSFORMS = [
    pytest.param(CropTransform(0.0, 0.0, 1.0, 1.0, 256), id="identity"),
    pytest.param(CropTransform(-13.5, 220.25, 0.3125, 0.3125, 256), id="upscale"),
    pytest.param(CropTransform(640.0, -12.0, 7.5, 7.5, 256), id="downscale"),
    pytest.param(CropTransform(1.25, -0.75, 0.001, 0.001, 256), id="extreme-upscale"),
]


# ---------------------------------------------------------------- pure transform math

@pytest.mark.parametrize("t", TRANSFORMS)
def test_round_trip_is_exact(t: CropTransform) -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-500, 900, size=(500, 2))
    back = t.to_image(t.to_crop(pts))
    assert np.allclose(back, pts, rtol=1e-12, atol=1e-9), np.abs(back - pts).max()


@pytest.mark.parametrize("t", TRANSFORMS)
def test_round_trip_the_other_way(t: CropTransform) -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 256, size=(500, 2))
    back = t.to_crop(t.to_image(pts))
    assert np.allclose(back, pts, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("t", TRANSFORMS)
def test_score_column_passes_through_untouched(t: CropTransform) -> None:
    """A transform that mangles scores would look like a confidence bug elsewhere."""
    pts = np.array([[10.0, 20.0, 0.9], [-5.0, 300.0, 0.1]])
    out = t.to_image(t.to_crop(pts))
    assert np.array_equal(out[:, 2], pts[:, 2])
    assert np.allclose(out[:, :2], pts[:, :2], atol=1e-9)


def test_to_crop_does_not_mutate_its_input() -> None:
    t = CropTransform(5.0, 5.0, 2.0, 2.0, 256)
    pts = np.array([[10.0, 20.0]])
    before = pts.copy()
    t.to_crop(pts)
    assert np.array_equal(pts, before)


def test_integer_input_is_not_truncated() -> None:
    """An int array must not silently floor the result."""
    t = CropTransform(0.0, 0.0, 2.0, 2.0, 256)
    out = t.to_crop(np.array([[1, 3]]))
    assert out.dtype.kind == "f"
    assert np.allclose(out, [[0.5, 1.5]])


def test_inside_mask_flags_landmarks_outside_the_crop() -> None:
    t = CropTransform(50.0, 50.0, 100.0 / 256.0, 100.0 / 256.0, 256)
    pts = np.array([[100.0, 100.0],     # centre
                    [55.0, 100.0],      # inside
                    [40.0, 100.0],      # left of the box
                    [100.0, 1000.0]])   # far below
    assert t.inside(pts).tolist() == [True, True, False, False]


def test_isotropy_flag() -> None:
    assert CropTransform(0.0, 0.0, 2.0, 2.0, 256).isotropic
    assert not CropTransform(0.0, 0.0, 2.0, 2.5, 256).isotropic


# ---------------------------------------------------------------- against DeepLabCut

@pytest.mark.parametrize("box", BOXES)
def test_crop_shape_and_dtype(box: FaceBox) -> None:
    img = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    crop, t = crop_face(img, box, CFG)
    assert crop.shape == (CFG.size, CFG.size, 3)
    assert crop.dtype == img.dtype
    assert isinstance(t, CropTransform)


@pytest.mark.parametrize("box", BOXES)
def test_square_box_is_isotropic(box: FaceBox) -> None:
    """A square box must not produce anisotropic scale, or to_image is box-dependent."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, t = crop_face(img, box, CFG)
    assert t.isotropic, (t.scale_x, t.scale_y)


@pytest.mark.parametrize("box", BOXES)
def test_box_corners_land_on_crop_corners(box: FaceBox) -> None:
    """The point of not clipping: the box fills the crop. DLC rounds the corners to
    integers, so allow one image pixel - but no more."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, t = crop_face(img, box, CFG)
    x1, y1, x2, y2 = box.xyxy
    got_tl = t.to_image(np.array([0.0, 0.0]))
    got_br = t.to_image(np.array([float(CFG.size), float(CFG.size)]))
    assert np.abs(got_tl - [x1, y1]).max() <= 1.0
    assert np.abs(got_br - [x2, y2]).max() <= 1.0


@pytest.mark.parametrize("box", BOXES)
def test_transform_for_matches_crop_face(box: FaceBox) -> None:
    """Training annotations use transform_for; inference uses crop_face. If these ever
    diverge, every training target is offset from every prediction."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, a = crop_face(img, box, CFG)
    b = transform_for(box, (240, 320), CFG)
    assert a == b


def test_out_of_image_area_is_padded_not_clipped() -> None:
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    crop, _ = crop_face(img, FaceBox(-400.0, -400.0, 50.0, "anchors", 3), CFG)
    assert crop.shape == (CFG.size, CFG.size, 3)
    assert (crop == 0).all()


def test_crop_recovers_a_known_patch() -> None:
    """End-to-end: a box drawn around a bright square lands it centred in the crop, and
    the transform agrees about where it came from."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[90:110, 90:110] = 255
    crop, t = crop_face(img, FaceBox(100.0, 100.0, 40.0, "anchors", 11), CFG)
    ys, xs = np.where(crop[:, :, 0] > 128)
    assert abs(xs.mean() - CFG.size / 2) < 3.0
    assert abs(ys.mean() - CFG.size / 2) < 3.0
    back = t.to_image(np.array([xs.mean(), ys.mean()]))
    assert np.allclose(back, [100.0, 100.0], atol=1.5)


def test_landmarks_survive_the_round_trip_through_a_real_crop() -> None:
    """The whole cascade contract in one test: image landmarks -> crop -> back."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    box = FaceBox(180.0, 140.0, 96.0, "anchors", 11)
    _, t = crop_face(img, box, CFG)
    lm = np.array([[150.0, 110.0], [210.0, 170.0], [180.0, 140.0]])
    assert np.allclose(t.to_image(t.to_crop(lm)), lm, atol=1e-9)


def test_zero_side_box_raises() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="side must be positive"):
        crop_face(img, FaceBox(1.0, 1.0, 0.0, "anchors", 3), CFG)


def test_grayscale_image_raises() -> None:
    with pytest.raises(ValueError, match="HxWxC"):
        crop_face(np.zeros((10, 10)), FaceBox(1.0, 1.0, 4.0, "anchors", 3), CFG)


# ---------------------------------------------------------------- augmentation

def test_jitter_stays_in_family_and_is_bounded() -> None:
    rng = np.random.default_rng(7)
    box = FaceBox(100.0, 100.0, 80.0, "anchors", 11)
    for _ in range(500):
        j = jitter_box(box, rng, scale_frac=0.20, translate_frac=0.10)
        assert 0.8 * 80.0 <= j.side <= 1.25 * 80.0
        assert abs(j.cx - 100.0) <= 0.10 * 80.0 + 1e-9
        assert abs(j.cy - 100.0) <= 0.10 * 80.0 + 1e-9
        assert j.source == box.source and j.n_anchors == box.n_anchors


def test_jitter_is_deterministic_for_a_seed() -> None:
    box = FaceBox(100.0, 100.0, 80.0, "anchors", 11)
    a = jitter_box(box, np.random.default_rng(3), 0.2, 0.1)
    b = jitter_box(box, np.random.default_rng(3), 0.2, 0.1)
    assert a == b
