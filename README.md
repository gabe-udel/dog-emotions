# SuperAnimal-Quadruped + DogFLW face keypoints

Predicts **39 body keypoints** and **46 dog-face landmarks** on video, as a two-stage
cascade: the released DeepLabCut **SuperAnimal-Quadruped** model handles the body and
locates the head, and a dedicated model trained on **DogFLW** (Dog Facial Landmarks in
the Wild) handles the face from a face-filling crop.

> ### Just want to run it?
> **[SETUP.md](SETUP.md)** — three steps, no terminal.
>
> **[CLAUDE.md](CLAUDE.md)** is the full handoff: architecture, results, what was tried
> and what failed.

## Architecture

```
frame
  -> SuperAnimal detector (Faster R-CNN)        -> whole-dog box
  -> SuperAnimal-Quadruped pose, STOCK/FROZEN   -> 39 body keypoints
  -> derive_face_box() from the head keypoints  -> face box
  -> crop + resize to 256x256
  -> DogFLW face model                          -> 46 landmarks
```

Stage 1 is the published checkpoint, run unmodified. There is no head surgery, no
fine-tuning of it, and therefore nothing to forget.

**The invariant:** the face box is produced by `src/facebox.py::derive_face_box`, with
the same config, in **both** training and inference. Training the face model on DogFLW's
shipped face boxes and inferring on derived ones would mean the model never sees its
deployment distribution — different tightness, centring and aspect.
`tests/test_facebox.py` asserts both call sites resolve to the same function object.

## Results

Test split, 479 held-out DogFLW images, scored once against the previous architecture:

| | cascade | old unified | change |
|---|---|---|---|
| NME mean | **0.0313** | 0.0438 | **−28.6%** |
| NME median | **0.0173** | 0.0319 | −45.7% |
| PCK@5% | **84.1%** | 71.0% | +18.5% |
| PCK@10% | **94.0%** | 92.5% | +1.6% |

Every region improved: eye −49.6%, nose −49.8%, muzzle −56.9%, mouth −26.1%,
head top −70.2%, ear −5.5%. Face box derived on 478 of 479 images. Per-region detail and
the validation ablation that set the defaults are in **CLAUDE.md §10**.

Ears remain the worst region (0.0596, PCK@5% 57.6%) and barely moved — as predicted, since
ear-type multimodality is a label property rather than a resolution one.

## Why a cascade

The previous architecture extended SuperAnimal's head from 39 to 76 channels and
predicted body and face together from one whole-dog crop. It shipped at NME 0.0438 and
is preserved on the branch `stable-pre-model-separation`.

Its limit was structural: **99.99% of that model was shared** between body and face — one
HRNet-W32 backbone (29,363,185 params) feeding a single 1x1 head (2,508 params), so every
keypoint was a 32-number readout of the same feature vector. Two consequences the cascade
addresses:

* **Scale.** The face spanned a median 36 of 64 heatmap cells on DogFLW close-ups but only
  ~16 on video, where the detector returns a whole standing dog. A within-image crop sweep
  measured NME 0.0411 / 0.0362 / 0.0328 at 25% / 45% / 55% face fill — monotonic. The
  cascade normalises that: every face arrives at ~40 cells regardless of how big it was
  in the frame.
* **Supervision.** The old training set was 60.5% ground truth, 22.9% pseudo-label and
  16.6% masked, because body channels had no DogFLW labels and were supervised by the
  pretrained model's own predictions. The face model's is **99.8% ground truth**.

An honest caveat, measured rather than projected: on **DogFLW** the resolution gain is
modest — median face span goes 35.9 → 39.7 heatmap cells, about +11%, because the default
padding of 1.8x spends most of the crop guaranteeing that flopped ears stay inside it. The
larger effects are the variance collapse (the old spread was 21–60 cells) and the jump in
supervision density. `--pad` is a config field precisely so it can be swept on validation.

## Layout

```
src/faceconfig.py       every tunable, named and documented
src/facebox.py          derive_face_box() - the shared invariant
src/crop.py             crop + the coordinate transform, exactly invertible
src/decode.py           heatmap -> coordinates, explicit and tested
src/splits.py           train / val / test with leakage assertions
src/build_face_coco.py  COCO dataset from DERIVED boxes
src/facedata.py         box-jitter augmentation (subclass, not monkeypatch)
src/train_face.py       trains the 46-keypoint face model
src/cascade.py          two-stage inference
src/evaluate_face.py    NME / PCK against the 76-channel baseline
src/postfit.py          refits the quarantined corrections
src/run_video.py        labelled MP4
src/video_app.py        point-and-click front end
tests/                  coordinate round trips, the box invariant, decoding, splits
```

Run it with `.\run_pipeline.ps1` (`-SkipTrain` stops before the long step).

## Data

| thing | source | notes |
|---|---|---|
| SuperAnimal-Quadruped pose | `mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/superanimal_quadruped_hrnet_w32.pt` | HRNet-w32, 113 MB. What DeepLabCut 3.0.1 / dlclibrary 0.0.12 resolve |
| SuperAnimal detector | `superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt` | 73 MB. Benchmarked 12x faster and better recall than resnet50-v2 on DogFLW |
| DogFLW | Kaggle `georgemartvel/dogflw` (CC BY-NC 4.0) | 4,335 images, 46 landmarks + a face bbox each |

DogFLW is built on **Stanford Dogs**: ids look like `n02085620_11477`, where the prefix is
a WordNet synset. Breed is therefore exactly inferable (120 breeds) and individual dog is
not — which is why the validation split is stratified by breed rather than disjoint by
individual. See `src/splits.py` for the reasoning and the residual near-duplicate risk.

## Environment

**CPU-only on both machines this has run on** — a Windows 11 / Ryzen box (the current one;
PyTorch's ROCm builds are Linux-only, so the iGPU is unusable) and an aarch64 Asahi Linux
box. `device="cpu"` throughout is deliberate.

DeepLabCut 3.0.1 pins `numpy<2` and is installed with `--no-deps` to dodge that pin; the
PyTorch engine runs fine against numpy 2. Full per-platform install steps are in
**CLAUDE.md §8**; to just run the app, use **[SETUP.md](SETUP.md)**.

## Licence

DogFLW is **CC BY-NC 4.0 — non-commercial**, and that restriction propagates to any model
trained on it.
