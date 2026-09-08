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


def _find_key(node: object, key: str) -> object | None:
    """First value for `key` anywhere in a nested config. Structure-agnostic on purpose:
    DeepLabCut has moved this field between releases."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for v in node:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def checkpoint_n_keypoints(face_cfg: dict, expected: int) -> int:
    """How many landmarks this CHECKPOINT emits, checked against what this code expects.

    Two different facts that were being conflated. `TrainConfig.n_keypoints` describes
    the landmark set this source tree predicts (`keypoint_scheme.ACTIVE`); the
    checkpoint's own pytorch_config.yaml describes the set it was trained for. Taking
    the first as the second fails silently and expensively: a 46-channel checkpoint read
    as 38 divides the crop by the wrong number to get the heatmap stride, and every
    landmark then carries a neighbour's identity. The output still looks like a face, so
    nothing about the rendered video says it is wrong.

    Refusing is the only safe answer - the two sets are not a resize apart, they are
    different channels in a different order.
    """
    found = _find_key(face_cfg, "num_heatmaps")
    if found is None or int(found) == expected:
        return expected
    raise ValueError(
        f"this checkpoint predicts {found} landmarks, but this source tree predicts "
        f"{expected} (keypoint_scheme.ACTIVE). Decoding one as the other is silently "
        f"wrong, so it is refused.\n"
        f"  - to run a 38-landmark checkpoint: pick one from face_project/face38/\n"
        f"  - to run the 46-landmark face1 model: check out the source it was trained "
        f"with,\n      git worktree add ../dog-emotion-46kp 8437a6f^")


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
        self.n_face = checkpoint_n_keypoints(self.face_cfg,
                                             self.cfg.train.n_keypoints)
        self.ear_corrector, self.refiner = self._load_post()

    def _load_post(self):
        """Load the quarantined corrections, only when the config enables them.

        These live here rather than in each caller so that evaluation and video
        rendering apply exactly the same post-processing. They previously lived only in
        run_video.py, which meant `evaluate_face.py --shape-refine` accepted the flag
        and silently did nothing - producing an ablation that looked like "no effect"
        when the correction had never run at all.
        """
        ear = refiner = None
        if self.cfg.post.ear_correct:
            from ear_correct import EarCorrector
            ear = EarCorrector()
        if self.cfg.post.shape_refine:
            from shape_refine import Refiner
            refiner = Refiner()
        return ear, refiner

    def apply_post(self, face: np.ndarray) -> np.ndarray:
        """Apply whichever corrections the config enabled, in place. Order matters:
        ear correction reads the reliable landmarks, and the shape model overwrites
        head_top from them, so ears are corrected against untouched inputs first."""
        if self.ear_corrector is not None:
            self.ear_corrector.apply(face)
        if self.refiner is not None:
            self.refiner.apply(face)
        return face

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
        return self.apply_post(_to_image(pts, transform))

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
