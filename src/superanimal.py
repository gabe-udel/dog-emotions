"""Helpers to run the pretrained SuperAnimal-Quadruped model (DLC 3.0.1, PyTorch engine)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

SUPER_ANIMAL = "superanimal_quadruped"
POSE_MODEL = "hrnet_w32"                      # model registered for superanimal_quadruped in DLC 3.0.1
# mobilenet, not resnet50_fpn_v2: CLAUDE.md §3 benchmarked it at 0.31 s/img with 12/12
# recall vs 3.64 s/img and 9/12, and every caller (build_dogboxes, run_video, evaluate)
# already overrode the old default to this. Keeping resnet50 here only had the effect of
# making snapshot_paths() download a ~170 MB detector nothing uses.
DETECTOR = "fasterrcnn_mobilenet_v3_large_fpn"


def sa_bodyparts() -> list[str]:
    from deeplabcut.utils import auxiliaryfunctions as af
    from deeplabcut.pose_estimation_pytorch.modelzoo import get_super_animal_project_config_path
    cfg = af.read_plainconfig(get_super_animal_project_config_path(super_animal=SUPER_ANIMAL))
    return list(cfg["bodyparts"])


def snapshot_paths() -> tuple[Path, Path]:
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
    return (get_super_animal_snapshot_path(SUPER_ANIMAL, POSE_MODEL, download=True),
            get_super_animal_snapshot_path(SUPER_ANIMAL, DETECTOR, download=True))


def build_runners(device: str = "cpu", max_individuals: int = 1, with_detector: bool = True):
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    pose_snap, det_snap = snapshot_paths()
    # NOTE: detector_name must always be passed. build_superanimal_inference_config
    # switches `method` to bottom-up when it is None, and a bottom-up runner ignores the
    # bounding boxes we hand it. We keep the top-down config and simply skip loading the
    # detector weights when we already have boxes.
    cfg = PoseConfig.build_for_superanimal_inference(
        SUPER_ANIMAL, model_name=POSE_MODEL, detector_name=DETECTOR,
        max_individuals=max_individuals, device=device)
    cfg_d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    return cfg_d, get_inference_runners(
        cfg_d, snapshot_path=pose_snap, max_individuals=max_individuals,
        num_bodyparts=len(sa_bodyparts()), num_unique_bodyparts=0,
        device=device, detector_path=det_snap if with_detector else None)


def pose_on_boxes(pose_runner, items: list[tuple[str, np.ndarray]]) -> list[np.ndarray]:
    """items: (image_path, bboxes_xywh (n,4)) -> list of (n, 39, 3) arrays."""
    inputs = [(str(p), {"bboxes": np.asarray(b, dtype=float)}) for p, b in items]
    return [np.asarray(r["bodyparts"]) for r in pose_runner.inference(inputs)]
