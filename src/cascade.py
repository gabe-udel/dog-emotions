"""Two-stage inference: stock SuperAnimal for the body, a dedicated model for the face.

    frame
      -> SuperAnimal detector (Faster R-CNN)        -> whole-dog box
      -> SuperAnimal-Quadruped pose, STOCK/FROZEN   -> 39 body keypoints
      -> derive_face_box() from the head anchors    -> face box
      -> crop + resize to 256x256                   -> face crop
      -> DogFLW face model                          -> 46 landmarks
      -> CropTransform.to_image                     -> image coordinates

Stage 1 is the released SuperAnimal-Quadruped checkpoint, unmodified. There is no head
surgery, no fine-tuning and therefore nothing to forget - the drift metric the old
architecture needed does not apply, because the body weights are the published ones.

The face box comes from `facebox.derive_face_box`, the same function and the same
config the training data was built with. `tests/test_facebox.py` asserts that identity.

Decoding is explicit: this module runs the face model's forward pass itself and calls
`decode.decode_heatmaps`, rather than going through DeepLabCut's predictor. That is
what lets sub-pixel decoding be a tested function instead of an import-time patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import superanimal as sa
from crop import CropTransform, crop_face
from decode import decode_heatmaps
from facebox import FaceBox, anchor_indices, derive_face_box, failure_rate
from faceconfig import CascadeConfig

# DeepLabCut's SuperAnimal configs set colormode RGB and normalize_images true, which is
# albumentations' ImageNet normalisation. The face model inherits that config, so its
# inference preprocessing has to match or every activation is shifted.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FrameResult:
    """What the cascade knows about one frame."""

    dog_box_xywh: np.ndarray | None
    body: np.ndarray | None
    """(39, 3) SuperAnimal keypoints in image coordinates, or None if no dog found."""
    face_box: FaceBox | None
    face: np.ndarray | None
    """(46, 3) DogFLW landmarks in image coordinates, or None if no face box."""

    @property
    def has_face(self) -> bool:
        return self.face is not None


def preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 crop -> normalised NCHW float tensor, matching DLC's inference config."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def load_face_model(config_path: str | Path, snapshot: str | Path,
                    device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Build the face model from its pytorch_config.yaml and load a snapshot."""
    from deeplabcut.core.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    cfg = read_config_as_dict(str(config_path))
    model = PoseModel.build(cfg["model"], weight_init=None, pretrained_backbone=False)
    snap = torch.load(str(snapshot), map_location=device, weights_only=False)
    state = snap["model"] if isinstance(snap, dict) and "model" in snap else snap
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


class Cascade:
    """Stage 1 + stage 2, sharing one face-box definition."""

    def __init__(self, face_config: str | Path, face_snapshot: str | Path,
                 cfg: CascadeConfig | None = None, device: str = "cpu",
                 with_detector: bool = True) -> None:
        self.cfg = cfg or CascadeConfig()
        self.device = device
        self.sa_bodyparts = sa.sa_bodyparts()
        self.anchor_idx = anchor_indices(self.sa_bodyparts)
        self._sa_cfg, (self.pose_runner, self.detector_runner) = sa.build_runners(
            device=device, max_individuals=1, with_detector=with_detector)
        self.face_model, self.face_cfg = load_face_model(face_config, face_snapshot,
                                                         device)
        self.n_face = self.cfg.train.n_keypoints

    # ---------------------------------------------------------------- stage 1

    def stage1(self, image_paths: list[str]) -> tuple[list[np.ndarray | None],
                                                      list[np.ndarray | None]]:
        """Detector + stock SuperAnimal pose. Returns (dog boxes xywh, 39-kpt poses)."""
        preds = self.detector_runner.inference(images=image_paths)
        boxes: list[np.ndarray | None] = []
        for p in preds:
            b = p.get("bboxes")
            boxes.append(None if b is None or len(b) == 0 else np.asarray(b[0], float))

        items, keep = [], []
        for path, box in zip(image_paths, boxes):
            if box is not None:
                items.append((str(path), {"bboxes": np.array([box])}))
                keep.append(True)
            else:
                keep.append(False)

        poses: list[np.ndarray | None] = [None] * len(image_paths)
        if items:
            out = self.pose_runner.inference(items)
            it = iter(out)
            for i, k in enumerate(keep):
                if k:
                    poses[i] = np.asarray(next(it)["bodyparts"])[0]
        return boxes, poses

    # ---------------------------------------------------------------- stage 2

    def face_box_for(self, pose: np.ndarray | None,
                     dog_box: np.ndarray | None) -> FaceBox | None:
        """The one place inference decides where the face is. Same call as training."""
        if pose is None:
            return None
        return derive_face_box(pose, self.anchor_idx, self.cfg.facebox, dog_box)

    def stage2(self, image_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        """Face crop -> 46 landmarks in IMAGE coordinates."""
        crop, transform = crop_face(image_bgr, box, self.cfg.crop)
        with torch.no_grad():
            out = self.face_model(preprocess(crop).to(self.device))
        heat = out["bodypart"]["heatmap"].cpu().numpy()
        stride = self.cfg.crop.size / heat.shape[-2 if heat.shape[1] == self.n_face
                                                 else 1]
        pts = decode_heatmaps(heat, stride=stride,
                              subpixel=self.cfg.post.subpixel,
                              n_keypoints=self.n_face)[0]
        return _to_image(pts, transform)

    # ---------------------------------------------------------------- both

    def run_image(self, image_path: str, image_bgr: np.ndarray | None = None
                  ) -> FrameResult:
        boxes, poses = self.stage1([image_path])
        img = cv2.imread(image_path) if image_bgr is None else image_bgr
        return self._assemble(img, boxes[0], poses[0])

    def _assemble(self, image_bgr: np.ndarray, dog_box: np.ndarray | None,
                  pose: np.ndarray | None) -> FrameResult:
        fb = self.face_box_for(pose, dog_box)
        face = None
        if fb is not None and image_bgr is not None:
            face = self.stage2(image_bgr, fb)
        return FrameResult(dog_box_xywh=dog_box, body=pose, face_box=fb, face=face)

    def report(self, results: list[FrameResult]) -> dict[str, float | int]:
        """Box-derivation health over a run. Reported alongside every evaluation."""
        r = failure_rate([f.face_box for f in results])
        r["no_dog"] = sum(f.body is None for f in results)
        return r


def _to_image(points_crop: np.ndarray, transform: CropTransform) -> np.ndarray:
    """(J, 3) crop coords -> image coords, scores untouched."""
    out = np.asarray(points_crop, dtype=float).copy()
    out[:, :2] = transform.to_image(out[:, :2])
    return out
