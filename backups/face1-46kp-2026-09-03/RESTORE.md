# Backup: face1, the 46-landmark cascade

Taken **2026-09-08**, of the model trained **2026-09-03**. This is the checkpoint every
number in `CLAUDE.md` §10 was measured on: **NME 0.0313, PCK@5% 84.1%** on the 479-image
held-out test split.

Verify with `MANIFEST.sha256` before trusting anything here.

## Why this backup exists

Not disk failure. The concrete risk is **in-place overwrite**: `src/postfit.py` writes
`data/shape_model.npz` and `data/ear_bias.pkl` to fixed paths. Refitting those for the
38-landmark model destroys the 46-landmark versions, and without them the checkpoint
does **not** reproduce 0.0313 — the two corrections are worth 15% of NME (`faceconfig.py`,
`PostConfig` docstring). The checkpoint alone is not the model. That is why the fitted
corrections, the config, and the split are in here next to it.

## Contents

| file | why it is needed |
|---|---|
| `face_project_face1/snapshot-best-008.pt` | the weights, 46 output channels |
| `face_project_face1/pytorch_config.yaml` | `Cascade` cannot build the model without it |
| `face_project_face1/cascade_config.json` | the crop/box settings the weights assume |
| `face_project_face1/learning_stats.csv`, `log.txt` | the training record |
| `data/shape_model.npz` | shape refine, **fitted for 46 channels** (target `[14, 15]`) |
| `data/ear_bias.pkl` | ear correction, **fitted for 46 channels** |
| `data/splits.json` | without it no reported number is reproducible |
| `data/face_coco_summary.json` | what the 46-channel dataset build produced |
| `outputs/face_eval_*.json` | the scored val ablation and the single test run |

## Restoring

```powershell
$b = "backups\face1-46kp-2026-09-03"
Copy-Item "$b\face_project_face1\*" "face_project\face1\" -Force
Copy-Item "$b\data\shape_model.npz","$b\data\ear_bias.pkl" "data\" -Force
```

Then check out the source that matches. **The current `main` cannot run this checkpoint**:
`keypoint_scheme.ACTIVE` is 38 long, `TrainConfig.n_keypoints` is 38, and
`shape_refine.Refiner` raises on a model fitted for a different landmark set — which is
the guard working, not a bug. The 46-landmark source is the commit before
`8437a6f` ("Drop the 8 sliding ear-contour landmarks"):

```powershell
git log --oneline -1 8437a6f^
git worktree add ..\dog-emotion-46kp 8437a6f^
```

Do not restore over a 38-channel `data/*.npz` without backing that one up first — the
overwrite runs in both directions.

## What this is not

A copy on the same physical disk as the original. It protects against overwrite and
mistake, not against drive loss. For that, the checkpoint needs to leave this machine —
it is 113 MB, over GitHub's 100 MB file limit and excluded by `.gitignore` (`*.pt`), so
the route is a **GitHub Release asset**, not a commit.
