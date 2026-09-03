"""Training data plumbing for the face model: box jitter, via subclassing not patching.

DeepLabCut has no random box augmentation - `top_down_crop_margin` is a fixed margin,
not a jitter - so the face model would otherwise only ever see boxes derived from clean
DogFLW stills. In deployment its boxes come from SuperAnimal running on video frames,
where the first stage is noisier: a slightly-too-large box here, an off-centre one
there. A cascade trained only on its own best case is brittle exactly where it is used.

`PoseDataset._get_raw_item_crop` is the natural seam - it returns the single annotation
whose `bbox` the cropper is about to use - so this subclasses it rather than reaching
into DeepLabCut at import time. Jitter applies in train mode only; val and test must
stay deterministic or their numbers are not comparable between runs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.cocoloader import COCOLoader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDataset
from deeplabcut.pose_estimation_pytorch.task import Task

from crop import to_dlc_bbox
from facebox import FaceBox


class JitterBoxDataset(PoseDataset):
    """PoseDataset that perturbs the crop box on every training read.

    The jitter is resampled per __getitem__, so an image is seen under a different crop
    each epoch. Boxes are re-snapped through `to_dlc_bbox` afterwards, keeping the same
    integer-grid guarantee the rest of the pipeline relies on.
    """

    def __init__(self, *args: Any, jitter_scale: float = 0.0,
                 jitter_translate: float = 0.0, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.jitter_scale = float(jitter_scale)
        self.jitter_translate = float(jitter_translate)
        self._rng = np.random.default_rng(seed)

    @property
    def jitter_active(self) -> bool:
        return (self.mode == "train"
                and (self.jitter_scale > 0.0 or self.jitter_translate > 0.0))

    def _get_raw_item_crop(self, index: int) -> tuple[str, list[dict], int]:
        name, anns, img_id = super()._get_raw_item_crop(index)
        if not self.jitter_active or not anns:
            return name, anns, img_id

        ann = dict(anns[0])                      # never mutate the shared annotation
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            return name, anns, img_id

        side = max(w, h)
        s = float(np.exp(self._rng.uniform(-1.0, 1.0) * np.log1p(self.jitter_scale)))
        dx, dy = self._rng.uniform(-self.jitter_translate,
                                   self.jitter_translate, size=2) * side
        box = FaceBox(cx=x + w / 2.0 + float(dx), cy=y + h / 2.0 + float(dy),
                      side=side * s, source="jitter", n_anchors=0)
        ann["bbox"] = to_dlc_bbox(box)
        return name, [ann] + list(anns[1:]), img_id


class FaceLoader(COCOLoader):
    """COCOLoader that hands back a JitterBoxDataset.

    Constructed exactly like COCOLoader plus the two jitter fractions, so
    `apis.training.train` needs no changes to use it.
    """

    def __init__(self, *args: Any, jitter_scale: float = 0.0,
                 jitter_translate: float = 0.0, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.jitter_scale = jitter_scale
        self.jitter_translate = jitter_translate
        self.seed = seed

    def create_dataset(self, transform=None, mode: str = "train",
                       task: Task = Task.BOTTOM_UP) -> PoseDataset:
        parameters = self.get_dataset_parameters()
        data = self.load_data(mode)
        # filter_annotations is a @staticmethod on Loader - no self.
        data["annotations"] = Loader.filter_annotations(data["annotations"], task)
        return JitterBoxDataset(
            images=data["images"],
            annotations=data["annotations"],
            transform=transform,
            mode=mode,
            task=task,
            parameters=parameters,
            ctd_config=None,
            jitter_scale=self.jitter_scale,
            jitter_translate=self.jitter_translate,
            seed=self.seed,
        )
