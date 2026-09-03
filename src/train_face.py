"""Train the DogFLW face model on face-filling crops.

Replaces train_dogface.py. What is gone, and why it is gone:

* No head surgery. The old model grew SuperAnimal's 39-channel head to 76 and had to
  copy the pretrained channels verbatim to avoid forgetting. This model has its own
  46-channel head and shares nothing with the body model, so there is nothing to
  preserve and no drift metric to watch.
* No memory replay. Every one of the 46 channels has DogFLW ground truth.
* No pseudo-labels, no visibility -1 for low-confidence body channels. The only masked
  landmarks are the few a crop does not cover.

What is kept from the old trainer, because it was right: parameter groups so the
pretrained backbone moves slowly while the fresh head learns fast, selective backbone
freezing so backward stops early on CPU, file logging (DeepLabCut reports progress
through `logging` and is otherwise silent), and sleep suppression on Windows.

Transfer learning: the backbone is initialised from the released SuperAnimal-Quadruped
checkpoint - quadruped anatomy at the same 256 crop scale is a far better starting point
than ImageNet, and on a CPU-only box it is the difference between fine-tuning and
training from scratch. The head is initialised fresh; there is no donor mapping, because
there is no correspondence to exploit once the face model is independent.

    python src/train_face.py --run-name face1 --epochs 4
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from faceconfig import CascadeConfig, TrainConfig
from keypoint_scheme import DOGFLW_NAMES

PROJECT = Path("face_project")
HEAD_W = "heads.bodypart.heatmap_head.deconv_layers.0.weight"
HEAD_B = "heads.bodypart.heatmap_head.deconv_layers.0.bias"


def default_workers() -> int:
    """Windows has no fork(): under `spawn` each worker re-imports this module, which is
    slow and makes the albumentations/OpenCV stack in workers a known hang risk."""
    return 0 if sys.platform == "win32" else 2


def keep_awake() -> None:
    """Stop Windows sleeping mid-run. A 3-epoch phase that should have taken 43 minutes
    once took 14 hours because the machine slept unattended. Process-scoped, so it
    lapses when training exits."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        if ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001):
            print("[power] sleep suppressed for the duration of this run", flush=True)
    except Exception as e:                       # never let this stop training
        print(f"[power] could not suppress sleep: {e}", flush=True)


def build_seed_snapshot(out: Path, n_keypoints: int, model_cfg: dict,
                        device: str = "cpu") -> Path:
    """SuperAnimal backbone + a fresh n_keypoints head, saved as a startable snapshot.

    DeepLabCut's trainer loads a snapshot with a STRICT `load_state_dict`, so the seed
    has to be a complete state dict for the model being built. Simply deleting the
    39-channel head tensors fails with "Missing key(s)" - the head must be present at
    the new width instead.

    So: build the 46-keypoint model to get a correctly-shaped, freshly-initialised head,
    and graft it onto the pretrained backbone. Any key whose shape disagrees is reported
    rather than silently replaced, because a silent mismatch would mean part of the
    backbone quietly failed to transfer.
    """
    import superanimal as sa
    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    src, _ = sa.snapshot_paths()
    snap = torch.load(src, map_location=device, weights_only=False)
    pretrained = dict(snap["model"])

    fresh = PoseModel.build(model_cfg, weight_init=None,
                            pretrained_backbone=False).state_dict()

    state, transferred, reinit = {}, 0, []
    for key, target in fresh.items():
        source = pretrained.get(key)
        if source is not None and source.shape == target.shape:
            state[key] = source.clone()
            transferred += target.numel()
        else:
            state[key] = target.clone()
            reinit.append((key, tuple(target.shape),
                           None if source is None else tuple(source.shape)))

    snap["model"] = state
    snap["metadata"] = {"epoch": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, out)

    total = sum(t.numel() for t in fresh.values())
    print(f"[seed] transferred {transferred:,}/{total:,} params "
          f"({100 * transferred / total:.2f}%) from SuperAnimal-Quadruped", flush=True)
    for key, want, got in reinit:
        print(f"[seed]   fresh: {key} {want}"
              + (f"  (pretrained had {got})" if got else "  (absent upstream)"),
              flush=True)
    expected = {HEAD_W, HEAD_B}
    unexpected = [k for k, _, _ in reinit if k not in expected]
    if unexpected:
        raise RuntimeError(
            "backbone weights failed to transfer, which would silently throw away the "
            f"pretraining this run depends on: {unexpected}")
    print(f"[seed] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def build_config(tc: TrainConfig, bodyparts: list[str], detector: str, workers: int,
                 eval_every: int, max_snapshots: int, save_epochs: int) -> dict:
    from deeplabcut.core.config import read_config_as_dict
    import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
        get_super_animal_model_config_path,
    )

    cfg = read_config_as_dict(get_super_animal_model_config_path("hrnet_w32"))
    cfg["method"] = "td"
    cfg["net_type"] = "hrnet_w32"
    cfg["device"] = "cpu"
    cfg["detector"] = read_config_as_dict(get_super_animal_model_config_path(detector))
    cfg["model"] = config_utils.replace_default_values(cfg["model"],
                                                       num_bodyparts=tc.n_keypoints)
    tg = cfg["model"]["heads"]["bodypart"]["target_generator"]
    if tg.get("pos_dist_thresh") != tc.pos_dist_thresh:
        print(f"[targets] pos_dist_thresh {tg.get('pos_dist_thresh')} -> "
              f"{tc.pos_dist_thresh}", flush=True)
    tg["pos_dist_thresh"] = tc.pos_dist_thresh

    cfg["metadata"] = {
        "project_path": str(PROJECT.resolve()),
        "pose_config_path": str((PROJECT / "train" / "pytorch_config.yaml").resolve()),
        "bodyparts": bodyparts, "unique_bodyparts": [], "individuals": ["animal0"],
        "with_identity": None,
    }
    box = {"width": tc.crop, "height": tc.crop, "margin": 0}
    cfg["data"]["train"]["top_down_crop"] = dict(box)
    cfg["data"]["inference"]["top_down_crop"] = dict(box)
    cfg["train_settings"].update(batch_size=tc.batch_size, epochs=tc.epochs,
                                 dataloader_workers=workers, display_iters=25,
                                 seed=tc.seed)
    cfg["runner"]["optimizer"] = {"type": "AdamW",
                                  "params": {"lr": tc.lr_backbone,
                                             "head_lr": tc.lr_head}}
    cfg["runner"]["scheduler"] = None
    # Score the held-out split every epoch. Train loss cannot distinguish learning from
    # memorising - on the old model it went flat while test NME was still falling.
    cfg["runner"]["eval_interval"] = eval_every if eval_every > 0 else 10_000
    cfg["runner"]["snapshots"] = {"max_snapshots": max_snapshots,
                                  "save_epochs": save_epochs,
                                  "save_optimizer_state": False}
    return cfg


def patch_optimizer(unfreeze: str) -> None:
    """Head/backbone parameter groups plus selective freezing.

    DeepLabCut's build_optimizer puts every parameter in one group at one LR. The fresh
    head must learn fast while the transferred backbone barely moves, and on CPU
    freezing everything below HRNet's last stage stops backward from reaching 8M
    parameters that are not being trained anyway.
    """
    import deeplabcut.pose_estimation_pytorch.runners.train as dlc_train

    def build_optimizer(model, optimizer_config):
        params = dict(optimizer_config["params"])
        head_lr = params.pop("head_lr", params["lr"])
        base_lr = params.pop("lr")
        keep = [] if unfreeze in ("none", "") else (
            None if unfreeze == "all" else [k.strip() for k in unfreeze.split(",")])
        for name, p in model.backbone.named_parameters():
            p.requires_grad_(keep is None or any(f".{k}." in f".{name}." for k in keep))

        head = [p for n, p in model.named_parameters() if n.startswith("heads.")]
        back = [p for n, p in model.named_parameters()
                if not n.startswith("heads.") and p.requires_grad]
        frozen = sum(p.numel() for n, p in model.named_parameters()
                     if not n.startswith("heads.") and not p.requires_grad)
        groups = [{"params": head, "lr": head_lr}]
        if back:
            groups.append({"params": back, "lr": base_lr})
        print(f"[optimizer] head {sum(p.numel() for p in head):,} @ lr={head_lr} | "
              f"backbone trainable {sum(p.numel() for p in back):,} @ lr={base_lr} | "
              f"frozen {frozen:,}  (unfreeze={unfreeze!r})", flush=True)
        return torch.optim.AdamW(groups, **params)

    dlc_train.build_optimizer = build_optimizer


def main() -> None:
    tc = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="face1")
    ap.add_argument("--epochs", type=int, default=tc.epochs)
    ap.add_argument("--batch-size", type=int, default=tc.batch_size)
    ap.add_argument("--crop", type=int, default=tc.crop)
    ap.add_argument("--lr-backbone", type=float, default=tc.lr_backbone)
    ap.add_argument("--lr-head", type=float, default=tc.lr_head)
    ap.add_argument("--unfreeze", default=tc.unfreeze,
                    help='"none" (head only), "all", or submodules e.g. "stage4"')
    ap.add_argument("--pos-dist-thresh", type=int, default=tc.pos_dist_thresh,
                    help="heatmap target width; 8 is the default for face-filling crops")
    ap.add_argument("--jitter-scale", type=float, default=tc.box_jitter_scale)
    ap.add_argument("--jitter-translate", type=float, default=tc.box_jitter_translate)
    ap.add_argument("--snapshot", default=None,
                    help="resume from this snapshot; default builds a SuperAnimal-"
                         "backbone seed")
    ap.add_argument("--detector", default="fasterrcnn_mobilenet_v3_large_fpn")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-snapshots", type=int, default=20)
    ap.add_argument("--save-epochs", type=int, default=1)
    a = ap.parse_args()

    tc = TrainConfig(
        n_keypoints=len(DOGFLW_NAMES), pos_dist_thresh=a.pos_dist_thresh, crop=a.crop,
        batch_size=a.batch_size, epochs=a.epochs, lr_backbone=a.lr_backbone,
        lr_head=a.lr_head, unfreeze=a.unfreeze, box_jitter_scale=a.jitter_scale,
        box_jitter_translate=a.jitter_translate,
    )
    keep_awake()

    from deeplabcut.pose_estimation_pytorch import utils
    from deeplabcut.pose_estimation_pytorch.apis.training import train
    from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
    from deeplabcut.pose_estimation_pytorch.task import Task

    from facedata import FaceLoader

    bodyparts = [DOGFLW_NAMES[i] for i in range(tc.n_keypoints)]
    cfg = build_config(tc, bodyparts, a.detector, a.workers, a.eval_every,
                       a.max_snapshots, a.save_epochs)

    model_folder = PROJECT / a.run_name
    model_folder.mkdir(parents=True, exist_ok=True)
    cfg["metadata"]["pose_config_path"] = str(
        (model_folder / "pytorch_config.yaml").resolve())

    snapshot = a.snapshot or str(build_seed_snapshot(
        PROJECT / "seed_superanimal_backbone.pt", tc.n_keypoints, cfg["model"]))

    loader = FaceLoader(project_root=PROJECT, model_config=cfg,
                        train_json_filename="train.json",
                        test_json_filename="val.json",
                        jitter_scale=tc.box_jitter_scale,
                        jitter_translate=tc.box_jitter_translate, seed=tc.seed)
    assert loader.model_folder == model_folder.resolve(), loader.model_folder
    loader.model_cfg.to_yaml(model_folder / "pytorch_config.yaml", overwrite=True)
    (model_folder / "cascade_config.json").write_text(
        json.dumps(CascadeConfig(train=tc).to_dict(), indent=2))
    setup_file_logging(model_folder / "log.txt")
    utils.fix_seeds(tc.seed)

    patch_optimizer(tc.unfreeze)
    print(f"[train] {tc.n_keypoints} face keypoints | crop {tc.crop} | "
          f"batch {tc.batch_size} | epochs {tc.epochs} | unfreeze={tc.unfreeze} | "
          f"sigma_thresh={tc.pos_dist_thresh} | "
          f"jitter scale +/-{tc.box_jitter_scale} translate +/-"
          f"{tc.box_jitter_translate} | workers={a.workers}", flush=True)
    print("[train] validation split is val.json - test.json is NOT touched here",
          flush=True)

    t = time.time()
    train(loader=loader, run_config=loader.model_cfg, task=Task.TOP_DOWN,
          device="cpu", snapshot_path=snapshot)
    print(f"[train] finished in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
