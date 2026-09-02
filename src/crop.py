"""Crop a face box to a fixed square and map coordinates in and out of it.

This is the silent-failure surface of a two-stage cascade: a half-pixel disagreement
between the training crop and the inference crop is invisible in the rendered video and
shows up only as a small, stable NME penalty that looks like model error.

So pixel extraction is NOT reimplemented here. Training goes through DeepLabCut's
dataloader, which calls `deeplabcut...data.image.top_down_crop`, and this module calls
that same function - parity by construction rather than by careful matching. An earlier
draft of this file did its own exact-float `warpAffine`; DLC rounds the box corners to
integers and pads asymmetrically, so the two agreed only to within half a pixel, which
is precisely the bug this module exists to prevent.

What IS defined here is `CropTransform`: DLC returns `(offset, scale)` and documents
`x_crop = (x - offset_x) / scale_x`, but nothing in DLC packages that as an invertible
object or tests the round trip. It is pure arithmetic with no DeepLabCut import, so
`tests/test_crop.py` exercises it without loading the framework.

The box is never clipped to the image. Clipping would change its aspect and make the
scale anisotropic; DLC pads the out-of-image region instead, which keeps scale_x and
scale_y equal for a square box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facebox import FaceBox
from faceconfig import CropConfig


@dataclass(frozen=True)
class CropTransform:
    """Maps between image coordinates and crop coordinates.

    Follows DeepLabCut's own convention, so the numbers can be taken straight from
    `top_down_crop` without reinterpretation:

        crop  = (image - offset) / scale
        image = crop * scale + offset

    `scale` is image pixels per crop pixel, so it is > 1 when the box is larger than
    the crop (a DogFLW close-up) and < 1 when it is smaller (a distant dog on video).
    """

    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float
    size: int

    def to_crop(self, xy: np.ndarray) -> np.ndarray:
        """Image coords -> crop coords. Accepts (..., 2) or (..., 3); score passes through."""
        xy = np.asarray(xy, dtype=float)
        out = xy.astype(float, copy=True)
        out[..., 0] = (xy[..., 0] - self.offset_x) / self.scale_x
        out[..., 1] = (xy[..., 1] - self.offset_y) / self.scale_y
        return out

    def to_image(self, xy: np.ndarray) -> np.ndarray:
        """Crop coords -> image coords. Exact inverse of `to_crop`."""
        xy = np.asarray(xy, dtype=float)
        out = xy.astype(float, copy=True)
        out[..., 0] = xy[..., 0] * self.scale_x + self.offset_x
        out[..., 1] = xy[..., 1] * self.scale_y + self.offset_y
        return out

    def inside(self, xy: np.ndarray) -> np.ndarray:
        """Which image-coordinate points land inside the crop.

        Used when building training targets, so the model is never asked to place a
        landmark the crop does not cover. At the default pad of 1.8 this excludes at
        least one landmark in roughly 2.3% of DogFLW images.
        """
        c = self.to_crop(xy)
        return ((c[..., 0] >= 0) & (c[..., 0] <= self.size - 1)
                & (c[..., 1] >= 0) & (c[..., 1] <= self.size - 1))

    @property
    def isotropic(self) -> bool:
        """True when x and y share a scale, as they must for a square box."""
        return bool(np.isclose(self.scale_x, self.scale_y, rtol=1e-9, atol=0.0))


def to_dlc_bbox(box: FaceBox) -> np.ndarray:
    """Snap a FaceBox to the integer grid DeepLabCut is going to round it to anyway.

    `top_down_crop` computes the corners as `int(round(cx +/- w/2))`. For a fractional
    box that rounding is not symmetric: a 33.75-px square at cx=100.5, cy=60.25 becomes
    84..117 by 43..77, i.e. 33 x 34, and the crop scale comes out anisotropic - 0.1289
    against 0.1328, a 3% distortion that grows as the box shrinks. A face model trained
    on subtly stretched crops learns a subtly stretched face.

    Snapping the corner and the side to integers first makes DLC's rounding the identity
    and guarantees w == h == side. The box moves by at most half a pixel, identically in
    the training and inference paths, because both go through this function.
    """
    x1, y1, _, _ = box.xyxy
    side = max(1.0, float(round(box.side)))
    return np.array([float(round(x1)), float(round(y1)), side, side], dtype=float)


def crop_face(image: np.ndarray, box: FaceBox,
              cfg: CropConfig) -> tuple[np.ndarray, CropTransform]:
    """Extract `box` from `image` as a cfg.size x cfg.size crop.

    Delegates to DeepLabCut's `top_down_crop` so the result is byte-identical to what
    the training dataloader produces for the same box. Returns the crop and the
    transform that produced it.
    """
    from deeplabcut.pose_estimation_pytorch.data.image import top_down_crop

    if image.ndim != 3:
        raise ValueError(f"image must be HxWxC; got {image.shape}")
    if box.side <= 0:
        raise ValueError(f"box side must be positive; got {box.side}")

    crop, offset, scale = top_down_crop(
        image, to_dlc_bbox(box),
        output_size=(cfg.size, cfg.size), margin=0,
    )
    return crop, CropTransform(offset_x=float(offset[0]), offset_y=float(offset[1]),
                               scale_x=float(scale[0]), scale_y=float(scale[1]),
                               size=cfg.size)


def transform_for(box: FaceBox, image_shape: tuple[int, int],
                  cfg: CropConfig) -> CropTransform:
    """The transform a box induces, without doing pixel work.

    Needed when building training annotations, where the landmarks must be masked
    against the crop but the image itself is not being loaded. Runs DLC's geometry on a
    zero-cost stub so the arithmetic - including its integer rounding and padding - is
    the same code path as the real crop.
    """
    h, w = image_shape
    stub = np.zeros((int(h), int(w), 1), dtype=np.uint8)
    _, t = crop_face(stub, box, cfg)
    return t


def jitter_box(box: FaceBox, rng: np.random.Generator, scale_frac: float,
               translate_frac: float) -> FaceBox:
    """Randomly perturb a box, for training-time augmentation.

    The cascade's first stage is noisier on video than on DogFLW stills, so the face
    model has to tolerate a box that is somewhat too big, too small or off-centre.
    Training only on clean derived boxes produces a model that is brittle exactly where
    it is deployed. Scale is sampled multiplicatively so shrink and grow are symmetric.
    """
    s = float(np.exp(rng.uniform(-1.0, 1.0) * np.log1p(scale_frac)))
    dx, dy = rng.uniform(-translate_frac, translate_frac, size=2) * box.side
    return FaceBox(cx=box.cx + float(dx), cy=box.cy + float(dy),
                   side=box.side * s, source=box.source, n_anchors=box.n_anchors)
