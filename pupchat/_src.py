"""Bridge to the existing pipeline in `src/`, which is a directory of modules rather
than a package.

Two groups, and the split matters for test speed:

* `NOT_FEATURES`, `ANGULAR_FEATURES`, `OSCILLATORY` come from `aggregate`/`features`,
  which import nothing heavier than numpy. Importing this module is cheap, so the
  evidence layer and its tests never load torch.
* The cascade itself is imported lazily inside `vision()`, because that one does pull
  DeepLabCut and takes seconds.

Importing these rather than restating them is deliberate: a second copy of
NOT_FEATURES that drifted would put a bookkeeping column into the evidence as if it
were a signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate import (NOT_FEATURES, OSCILLATORY, aggregate,  # noqa: E402
                       circ_mean, load_noise_floors, smooth, window_frames, wrap)
from features import ANGULAR_FEATURES  # noqa: E402

__all__ = ["NOT_FEATURES", "OSCILLATORY", "ANGULAR_FEATURES", "aggregate",
           "load_noise_floors", "smooth", "window_frames", "circ_mean", "wrap",
           "vision"]


def vision():
    """The heavy half, imported only when a video is actually being processed."""
    import superanimal as sa
    from cascade import Cascade, FrameResult, config_beside
    from faceconfig import CascadeConfig, FaceBoxConfig, PostConfig
    from keypoint_scheme import ACTIVE_NAMES
    from run_video import clip_features
    from temporal import ransac_filter, window_frames
    return {
        "sa": sa, "Cascade": Cascade, "FrameResult": FrameResult,
        "config_beside": config_beside, "CascadeConfig": CascadeConfig,
        "FaceBoxConfig": FaceBoxConfig, "PostConfig": PostConfig,
        "ACTIVE_NAMES": ACTIVE_NAMES, "clip_features": clip_features,
        "ransac_filter": ransac_filter, "window_frames": window_frames,
    }
