"""Fine-tune SuperAnimal-Quadruped with the added DogFLW face keypoints.

Uses DeepLabCut's own PyTorch training stack (COCOLoader + apis.training.train) on the
39+N-keypoint COCO dataset, resuming from the surgically extended snapshot.

Two things are customised, both deliberately:
  * build_optimizer is overridden to give parameter groups - the pretrained backbone is
    kept at a very low LR (or frozen) while the heatmap head, which contains the newly
    added output channels, trains at a higher one.
  * with --freeze-backbone the backbone requires_grad is switched off, so a warm-up phase
    costs a forward pass only.  On CPU that is the difference between hours and minutes.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path

import torch
sys.path.insert(0, "src")

PROJECT = Path("dlc_project")


def default_workers() -> int:
    """DataLoader workers.

    Windows has no fork(): each worker re-imports this module under `spawn`, which costs
    seconds per epoch and makes the albumentations/OpenCV stack in the workers a known
    hang risk. Default to in-process loading there; override with --workers.
    """
    return 0 if sys.platform == "win32" else 2


def keep_awake():
    """Stop Windows sleeping mid-run.

    A 3-epoch phase 2 that should have taken 43 minutes took 14 hours because the
    machine slept unattended. This is process-scoped - it lapses automatically when
    training exits, unlike editing the power plan.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            print("[power] sleep suppressed for the duration of this run", flush=True)
    except Exception as e:                      # never let this stop training
        print(f"[power] could not suppress sleep: {e}", flush=True)


def build_config(n_kpt: int, bodyparts: list[str], crop: int, batch_size: int,
                 epochs: int, lr_backbone: float, lr_head: float, save_epochs: int,
                 detector: str, workers: int, pos_dist_thresh: int,
                 eval_every: int = 0, max_snapshots: int = 2) -> dict:
    from deeplabcut.core.config import read_config_as_dict
    import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_model_config_path

    cfg = read_config_as_dict(get_super_animal_model_config_path("hrnet_w32"))
    cfg["method"] = "td"
    cfg["net_type"] = "hrnet_w32"
    cfg["device"] = "cpu"
    cfg["detector"] = read_config_as_dict(get_super_animal_model_config_path(detector))
    cfg["model"] = config_utils.replace_default_values(cfg["model"], num_bodyparts=n_kpt)
    # SuperAnimal ships pos_dist_thresh=17 (sigma ~11.3 px in the 256 crop, ~2.8 px on the
    # 64x64 heatmap). The face fills ~39 px of that grid and carries 46 landmarks, so at 17
    # neighbouring targets overlap almost entirely and the model cannot separate them.
    tg = cfg["model"]["heads"]["bodypart"]["target_generator"]
    if tg.get("pos_dist_thresh") != pos_dist_thresh:
        print(f"[targets] pos_dist_thresh {tg.get('pos_dist_thresh')} -> {pos_dist_thresh}",
              flush=True)
    tg["pos_dist_thresh"] = pos_dist_thresh
    cfg["metadata"] = {
        "project_path": str(PROJECT.resolve()),
        "pose_config_path": str((PROJECT / "train" / "pytorch_config.yaml").resolve()),
        "bodyparts": bodyparts, "unique_bodyparts": [], "individuals": ["animal0"],
        "with_identity": None,
    }
    # Top-down crops: 256x256 with margin 0, exactly what get_inference_runners defaults to
    # for the released SuperAnimal config, so training crops match inference crops and the
    # crops the memory-replay pseudo-labels were produced on.
    box = {"width": crop, "height": crop, "margin": 0}
    cfg["data"]["train"]["top_down_crop"] = dict(box)
    cfg["data"]["inference"]["top_down_crop"] = dict(box)
    cfg["train_settings"].update(batch_size=batch_size, epochs=epochs,
                                 dataloader_workers=workers, display_iters=25, seed=42)
    cfg["runner"]["optimizer"] = {"type": "AdamW",
                                  "params": {"lr": lr_backbone, "head_lr": lr_head}}
    cfg["runner"]["scheduler"] = None
    # eval_every=1 scores the held-out split after every epoch. Without it the only
    # signal is train loss, which cannot distinguish learning from memorising - so
    # there is no way to tell when more epochs stop helping.
    cfg["runner"]["eval_interval"] = eval_every if eval_every > 0 else 10_000
    cfg["runner"]["snapshots"] = {"max_snapshots": max_snapshots, "save_epochs": save_epochs,
                                  "save_optimizer_state": False}
    return cfg


def patch_optimizer(unfreeze: str):
    """Head/backbone parameter groups + selective backbone freezing.

    DeepLabCut's build_optimizer puts every parameter in one group at one LR. We need the
    added head channels to learn fast while the pretrained backbone barely moves, and on
    CPU we also need to stop backward early: freezing everything up to HRNet's last stage
    means the backward pass never reaches the 8M parameters below it.

    unfreeze: "none" (head only), "all", or a comma-separated list of backbone submodule
    prefixes to keep trainable, e.g. "stage4" or "stage4,transition3".
    """
    import deeplabcut.pose_estimation_pytorch.runners.train as dlc_train

    def build_optimizer(model, optimizer_config):
        params = dict(optimizer_config["params"])
        head_lr = params.pop("head_lr", params["lr"])
        base_lr = params.pop("lr")

        keep = [] if unfreeze in ("none", "") else (
            None if unfreeze == "all" else [k.strip() for k in unfreeze.split(",")])
        for name, p in model.backbone.named_parameters():
            trainable = keep is None or any(f".{k}." in f".{name}." for k in keep)
            p.requires_grad_(trainable)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--unfreeze", default="none",
                    help='"none" (head only), "all", or backbone submodules e.g. "stage4"')
    ap.add_argument("--snapshot", default="model_weights/superanimal_quadruped_hrnet_w32_dogface.pt")
    ap.add_argument("--run-name", default="phase1")
    ap.add_argument("--save-epochs", type=int, default=1)
    ap.add_argument("--detector", default="fasterrcnn_mobilenet_v3_large_fpn")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="DataLoader workers (default: 0 on Windows, 2 elsewhere)")
    ap.add_argument("--pos-dist-thresh", type=int, default=17,
                    help="heatmap target width; 17 is SuperAnimal's body-pose default, "
                         "lower is sharper and suits dense facial landmarks")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="score the held-out split every N epochs (1 = every epoch). "
                         "0 disables it, leaving train loss as the only signal")
    ap.add_argument("--max-snapshots", type=int, default=2,
                    help="how many epoch checkpoints to keep; raise it to keep every "
                         "epoch so the best one can be picked afterwards")
    args = ap.parse_args()
    keep_awake()

    from deeplabcut.pose_estimation_pytorch import COCOLoader, utils
    from deeplabcut.pose_estimation_pytorch.apis.training import train
    from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
    from deeplabcut.pose_estimation_pytorch.task import Task

    km = json.load(open("data/keypoint_map.json"))
    bodyparts = km["bodyparts"]
    cfg = build_config(len(bodyparts), bodyparts, args.crop, args.batch_size, args.epochs,
                       args.lr_backbone, args.lr_head, args.save_epochs, args.detector,
                       args.workers, args.pos_dist_thresh,
                       args.eval_every, args.max_snapshots)

    # Loader.model_folder is derived from metadata.pose_config_path, so the run folder
    # is chosen by pointing that at the config we are about to write.
    model_folder = PROJECT / args.run_name
    model_folder.mkdir(parents=True, exist_ok=True)
    cfg["metadata"]["pose_config_path"] = str((model_folder / "pytorch_config.yaml").resolve())

    loader = COCOLoader(project_root=PROJECT, model_config=cfg,
                        train_json_filename="train.json", test_json_filename="test.json")
    assert loader.model_folder == model_folder.resolve(), loader.model_folder
    loader.model_cfg.to_yaml(model_folder / "pytorch_config.yaml", overwrite=True)
    # DeepLabCut reports training progress through `logging`; without this the run is silent.
    setup_file_logging(model_folder / "log.txt")
    utils.fix_seeds(42)

    patch_optimizer(args.unfreeze)
    print(f"[train] {len(bodyparts)} keypoints | crop {args.crop} | batch {args.batch_size} "
          f"| epochs {args.epochs} | unfreeze={args.unfreeze} | workers={args.workers}",
          flush=True)
    t = time.time()
    train(loader=loader, run_config=loader.model_cfg, task=Task.TOP_DOWN, device="cpu",
          snapshot_path=args.snapshot)
    print(f"[train] finished in {(time.time()-t)/60:.1f} min")


if __name__ == "__main__":
    main()
