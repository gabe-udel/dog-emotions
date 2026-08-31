"""Recover the sub-cell precision DeepLabCut's argmax decoding throws away.

The pose head emits a 64x64 heatmap per keypoint and
`HeatmapPredictor.get_pose_prediction` takes `torch.argmax`, then adds a learned offset
from the locref map.  This project's models were trained with `generate_locref: false`,
so that offset is always None and every keypoint snaps to a cell centre - 4 px in the
256 px crop.

That is fatal for dense facial landmarks.  Measured on video, the 46 face channels
occupied only **25 distinct pixel positions** out of 46 drawn: any two landmarks whose
peaks land in the same cell decode to byte-identical coordinates and render as one dot.

Fix: fit a parabola through the peak and its immediate neighbours along each axis and
take the vertex - the standard HRNet / DARK-pose post-processing step.  It is
inference-only, needs no retraining, and is skipped automatically when a model does
supply locref.

    import subpixel; subpixel.enable()      # before building any inference runner
"""
from __future__ import annotations

import torch

_PATCHED = False


def enable() -> bool:
    """Monkey-patch HeatmapPredictor. Idempotent; returns False if already active."""
    global _PATCHED
    if _PATCHED:
        return False
    from deeplabcut.pose_estimation_pytorch.models.predictors import single_predictor as sp

    original = sp.HeatmapPredictor.get_pose_prediction

    def get_pose_prediction(self, heatmap, locref, scale_factors):
        # A model with a real locref map already has sub-cell offsets; leave it alone.
        if locref is not None:
            return original(self, heatmap, locref, scale_factors)

        y, x = self.get_top_values(heatmap)              # (B, J) cell indices
        B, J = x.shape
        Hh, Ww = heatmap.shape[1], heatmap.shape[2]
        bi = torch.arange(B, device=heatmap.device).unsqueeze(1).expand(-1, J)
        ji = torch.arange(J, device=heatmap.device).unsqueeze(0).expand(B, -1)

        def at(yy, xx):
            return heatmap[bi, yy.clamp(0, Hh - 1), xx.clamp(0, Ww - 1), ji]

        centre = at(y, x)
        # Parabola vertex: d = 0.5 * (h[+1] - h[-1]) / (2*h[0] - h[+1] - h[-1]).
        # At a true maximum the denominator is positive; guard the degenerate flat case.
        def offset(hp, hm):
            den = 2.0 * centre - hp - hm
            den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
            return (0.5 * (hp - hm) / den).clamp(-0.5, 0.5)

        dx = offset(at(y, x + 1), at(y, x - 1))
        dy = offset(at(y + 1, x), at(y - 1, x))
        # A peak on the border has no neighbour on one side - keep the cell centre.
        edge = (x <= 0) | (x >= Ww - 1) | (y <= 0) | (y >= Hh - 1)
        zero = torch.zeros_like(dx)
        dx, dy = torch.where(edge, zero, dx), torch.where(edge, zero, dy)

        xf = (x.float() + dx).unsqueeze(1) * scale_factors[1] + 0.5 * scale_factors[1]
        yf = (y.float() + dy).unsqueeze(1) * scale_factors[0] + 0.5 * scale_factors[0]
        return torch.stack([xf, yf, centre.unsqueeze(1)], dim=-1)

    sp.HeatmapPredictor.get_pose_prediction = get_pose_prediction
    _PATCHED = True
    return True
