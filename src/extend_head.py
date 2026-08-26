"""Grow the SuperAnimal-Quadruped heatmap head from 39 outputs to 39 + N.

The head is a single 1x1 ConvTranspose2d over the 32-channel HRNet feature map, so
extending it is exact: the 39 pretrained channels are copied verbatim (the model keeps
everything it already knows) and each added channel is warm-started from its nearest
SuperAnimal keypoint, plus a little noise so the copies are not degenerate.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "src")
import superanimal as sa

W = "heads.bodypart.heatmap_head.deconv_layers.0.weight"   # (in=32, out=39, 1, 1)
B = "heads.bodypart.heatmap_head.deconv_layers.0.bias"     # (39,)
OUT = Path("model_weights/superanimal_quadruped_hrnet_w32_dogface.pt")
NOISE = 1e-3


def main():
    km = json.load(open("data/keypoint_map.json"))
    new_idx = km["new_dogflw_indices"]
    donors = [int(km["init_donor_idx"][str(i)]) for i in new_idx]
    n_new = len(new_idx)

    src, _ = sa.snapshot_paths()
    snap = torch.load(src, map_location="cpu", weights_only=False)
    sd = snap["model"]
    w, b = sd[W], sd[B]
    assert w.shape[1] == 39 and b.shape[0] == 39, (w.shape, b.shape)

    g = torch.Generator().manual_seed(0)
    new_w = torch.empty(w.shape[0], 39 + n_new, *w.shape[2:], dtype=w.dtype)
    new_b = torch.empty(39 + n_new, dtype=b.dtype)
    new_w[:, :39], new_b[:39] = w, b
    for k, d in enumerate(donors):
        new_w[:, 39 + k] = w[:, d] + NOISE * torch.randn(w.shape[0], *w.shape[2:], generator=g)
        new_b[39 + k] = b[d]

    sd[W], sd[B] = new_w, new_b
    snap["metadata"] = {"epoch": 0}      # start the fine-tune at epoch 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, OUT)

    print(f"head {tuple(w.shape)} -> {tuple(new_w.shape)}   bias {tuple(b.shape)} -> {tuple(new_b.shape)}")
    print(f"39 pretrained channels copied verbatim; {n_new} added channels warm-started")
    print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
