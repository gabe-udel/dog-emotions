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


def build_config(n_kpt: int, bodyparts: list[str], crop: int, batch_size: int,
                 epochs: int, lr_backbone: float, lr_head: float, save_epochs: int,
                 detector: str) -> dict:
    from deeplabcut.core.config import read_config_as_dict
    import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_model_config_path

    cfg = read_config_as_dict(get_super_animal_model_config_path("hrnet_w32"))
    cfg["method"] = "td"
    cfg["net_type"] = "hrnet_w32"
    cfg["device"] = "cpu"
    cfg["detector"] = read_config_as_dict(get_super_animal_model_config_path(detector))
    cfg["model"] = config_utils.replace_default_values(cfg["model"], num_bodyparts=n_kpt)
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
                                 dataloader_workers=2, display_iters=25, seed=42)
    cfg["runner"]["optimizer"] = {"type": "AdamW",
                                  "params": {"lr": lr_backbone, "head_lr": lr_head}}
    cfg["runner"]["scheduler"] = None
    cfg["runner"]["eval_interval"] = 10_000          # we evaluate separately, see evaluate.py
    cfg["runner"]["snapshots"] = {"max_snapshots": 2, "save_epochs": save_epochs,
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
    args = ap.parse_args()

    from deeplabcut.pose_estimation_pytorch import COCOLoader, utils
    from deeplabcut.pose_estimation_pytorch.apis.training import train
    from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
    from deeplabcut.pose_estimation_pytorch.task import Task

    km = json.load(open("data/keypoint_map.json"))
    bodyparts = km["bodyparts"]
    cfg = build_config(len(bodyparts), bodyparts, args.crop, args.batch_size, args.epochs,
                       args.lr_backbone, args.lr_head, args.save_epochs, args.detector)

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
          f"| epochs {args.epochs} | unfreeze={args.unfreeze}", flush=True)
    t = time.time()
    train(loader=loader, run_config=loader.model_cfg, task=Task.TOP_DOWN, device="cpu",
          snapshot_path=args.snapshot)
    print(f"[train] finished in {(time.time()-t)/60:.1f} min")


if __name__ == "__main__":
    main()
