"""Every tunable in the face cascade, named and documented in one place.

The old pipeline scattered magic numbers across modules: PSEUDO_THRESH in make_coco,
pos_dist_thresh in train_dogface, 0.55/0.45/0.35 in run_video. Anything that changes
model behaviour now lives here as a documented field, so a config can be serialised
alongside a checkpoint and a run reproduced from it.

Nothing here reads a file or touches global state. Construct a dataclass, pass it down.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


# --------------------------------------------------------------------------------------
# Stage 1: which SuperAnimal keypoints anchor the face box.
# --------------------------------------------------------------------------------------
# The 4 antler keypoints are excluded on purpose: on a dog the pretrained model puts
# antler_base on top of earbase, so including them adds noise without adding extent.
# The remaining 11 head keypoints bound the region DogFLW annotates.
FACE_ANCHORS: tuple[str, ...] = (
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "left_eye", "right_earbase", "left_earbase",
    "right_earend", "left_earend",
)


@dataclass(frozen=True)
class FaceBoxConfig:
    """How a face box is derived from SuperAnimal's body-pose output.

    These values are shared by the training data builder and by video inference. They
    are the definition of the crop the face model sees, so changing one invalidates a
    trained checkpoint. `pad` and `anchor_conf` were both measured, not chosen:

    Over all 4,335 DogFLW images, the box side needed to contain every one of the 46
    ground-truth landmarks, expressed as a multiple of the anchor hull's side:

        anchor_conf   p50     p90     p95     pad 1.6x   pad 1.8x   pad 2.0x
        0.1           1.202   1.536   1.662   93.1%      97.7%      98.9%
        0.3           1.223   1.609   1.800   89.6%      95.0%      97.0%
        0.5           1.270   1.814   2.221   82.0%      89.7%      92.6%

    Two things follow, and the second is counterintuitive:

    * pad 1.8 contains all 46 landmarks in 97.7% of images while still leaving the face
      filling ~67% of the box (about 43 of 64 heatmap cells at stride 4). 2.0 buys 1.2
      more points of containment for a smaller face; 1.6 loses 4.6 points of it.
    * a LOW anchor_conf is better. Raising it drops marginal ear tips, which shrinks the
      hull, which means MORE padding is needed - p95 goes 1.66 -> 1.80 -> 2.22 as the
      threshold rises. Filter anchors loosely and let the pad absorb the error.
    """

    pad: float = 1.8
    """Box side as a multiple of the anchor hull's long side. See table above."""

    anchor_conf: float = 0.1
    """Minimum SuperAnimal score for a keypoint to anchor the box. Deliberately low."""

    min_anchors: int = 3
    """Below this many visible anchors the derivation is refused and the fallback runs.
    At anchor_conf=0.1 only 3 of 4,335 DogFLW images fall back."""

    min_side_px: float = 16.0
    """Reject a degenerate hull. A box smaller than this cannot carry 46 landmarks."""

    fallback: str = "dog_box_upper"
    """What to do when fewer than `min_anchors` are visible.

    'dog_box_upper' - take the upper `fallback_frac` of the detector's dog box, squared
                      off. A dog's head is at the top of its bounding box often enough
                      to be worth attempting, and it degrades to a bad crop rather than
                      to no prediction.
    'skip'          - emit no face prediction for this frame. Honest, but leaves gaps.
    """

    fallback_frac: float = 0.55
    """For 'dog_box_upper': fraction of the dog box's height treated as head region."""

    square: bool = True
    """Force aspect 1:1 so the resize to a square crop introduces no anisotropic scale.
    Set False only if the face model is retrained on non-square crops."""


@dataclass(frozen=True)
class CropConfig:
    """Geometry of the crop handed to the face model."""

    size: int = 256
    """Output crop is size x size. 256 matches SuperAnimal's own top-down crop, so the
    HRNet-W32 backbone operates at the scale its pretraining optimised, and the stride-4
    heatmap is 64x64."""

    border_value: int = 0
    """Fill for the region of a box that falls outside the image. A box is NOT clipped
    to image bounds - clipping would change its aspect and break the square invariant -
    so out-of-image area is padded instead."""


@dataclass(frozen=True)
class TrainConfig:
    """Face-model training. Not used at inference; recorded next to the checkpoint."""

    n_keypoints: int = 46
    """All 46 DogFLW landmarks as independent channels. The old 9-way merge with
    SuperAnimal channels existed only to avoid duplicate outputs in a shared head and
    has no meaning in a dedicated face model."""

    pos_dist_thresh: int = 8
    """Heatmap target width. SuperAnimal ships 17 for body pose. On the old whole-dog
    crop 17 was kept for consistency with the pretrained head and sharpening to 8 made
    NME worse - but that was measured when the face spanned ~36 heatmap cells. On a
    face-filling crop the landmarks are ~1.8x further apart in grid units, so 8 is the
    principled default here. It is a config field precisely because that reasoning is
    untested on the new architecture: sweep it on val."""

    crop: int = 256
    batch_size: int = 8
    epochs: int = 4
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    unfreeze: str = "stage4"
    """Backbone submodules left trainable. 'none' = head only, 'all' = everything."""

    box_jitter_scale: float = 0.20
    """Training-time box augmentation, +/- this fraction of the box side. Without it the
    model overfits to boxes derived from clean stills and degrades on video, where the
    stage-1 pose is noisier. This is the mechanism that makes the cascade robust to its
    own first stage."""

    box_jitter_translate: float = 0.10
    """Training-time box centre jitter, +/- this fraction of the box side."""

    seed: int = 42


@dataclass(frozen=True)
class PostConfig:
    """Post-hoc corrections, refitted against the cascade and measured on validation.

    Both started OFF after the rebuild, because each learns a correction to one specific
    model's residuals and neither was assumed to transfer. Both were then refitted on the
    train split and scored on the 574-image validation split, which is what turned them
    back on. Ablation, all-46 NME / PCK@5%:

        neither          0.0346 / 80.6%      ear 0.0700   head 0.0443
        + shape_refine   0.0339 / 81.7%      ear 0.0700   head 0.0277
        + both           0.0294 / 84.4%      ear 0.0554   head 0.0277

    The two are independent - one fixes ears, the other the skull top - and together
    they are worth 15% of NME on top of the architecture change.
    """

    ear_correct: bool = True
    """Per-ear-type systematic bias subtraction, refitted on 3,274 train faces
    (pointy 969 / half_floppy 630 / floppy 1,675). Ear NME 0.0700 -> 0.0554, -20.9%.
    Ear-type multimodality is a property of the labels, not the crop, which is why this
    survived the architecture change intact."""

    shape_refine: bool = True
    """Derive head_top_left/right from face geometry. This was expected to be obsolete -
    those points failed on the old model partly because they sat at the edge of a
    low-resolution crop, and the face-filling crop did fix most of it (0.1240 -> 0.0443
    with no correction at all). But the residual is still systematic: the shape model
    takes it to 0.0277, a further -37.5%, and PCK@5% from 65.9% to 91.2%."""

    subpixel: bool = True
    """Parabolic peak fitting when decoding heatmaps. Not a 'fix' like the other two -
    it is the correct way to read a heatmap, and argmax-only decoding is a defect. On
    by default and implemented as an explicit function in decode.py, not a patch."""

    min_confidence: float = 0.35
    """Display threshold for rendering. On the old model confidence ran OPPOSITE to
    accuracy across crop scales; re-check that relationship before tuning this."""


@dataclass(frozen=True)
class CascadeConfig:
    """The whole pipeline, and what gets written next to a checkpoint."""

    facebox: FaceBoxConfig = field(default_factory=FaceBoxConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    post: PostConfig = field(default_factory=PostConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
