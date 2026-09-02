# SuperAnimal-Quadruped + DogFLW face keypoints

Takes the released **DeepLabCut SuperAnimal-Quadruped** pose model and re-trains it so it
predicts, in addition to its 39 body keypoints, the facial landmarks of the **DogFLW**
(Dog Facial Landmarks in the Wild) dataset — then runs the result on a dog-walking video.

> ### Just want to run it?
> **[SETUP.md](SETUP.md)** — three steps, no terminal. The trained checkpoint is 113 MB,
> over GitHub's file limit, so it comes from the
> [Releases page](https://github.com/gabe-udel/dog-emotions/releases) rather than the repo.
>
> **[CLAUDE.md](CLAUDE.md)** is the full handoff: results, what was tried, what failed,
> and §11 on the four inference-time fixes that most of the accuracy came from.

## What was downloaded

| thing | source | notes |
|---|---|---|
| SuperAnimal-Quadruped pose model | `mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/superanimal_quadruped_hrnet_w32.pt` | HRNet-w32, 118 MB. This is the pose model that DeepLabCut 3.0.1 / dlclibrary 0.0.12 resolve for `superanimal_quadruped` |
| SuperAnimal-Quadruped detector | `superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt` | 76 MB, top-down animal detector |
| DogFLW | Kaggle `georgemartvel/dogflw` (CC BY-NC 4.0) | 4,335 images (3,855 train / 480 test), 46 facial landmarks + a face bbox each |
| demo video | Mixkit clip 1476, "Dog walking with its owner in a park" | Mixkit Free License |

Both `hrnet_w48` and `rtmpose_m/x` checkpoints also exist in the HuggingFace repo (uploaded
2025-06-30) but are **not** in the model registry shipped with the current DeepLabCut
release, so `hrnet_w32` is the current SuperAnimal-Quadruped pose model and the one used here.

## The problem with the stock approach

DeepLabCut ships "memory replay" precisely for fine-tuning a SuperAnimal model without
forgetting — but it cannot *add* keypoints. `prepare_memory_replay` projects **every**
project bodypart into SuperAnimal's 39-keypoint space:

```python
for idx, bpt in enumerate(bodyparts):
    conversion_table[bpt] = sa_bodyparts[weight_init.conversion_array[idx]]
```

and `PoseModel.build` then loads the pretrained head with `head.load_state_dict(...)`, which
requires the head to have exactly `len(conversion_array)` outputs. Either path caps the model
at 39 keypoints. So the fine-tune here generalises memory replay instead of using it directly.

## Pipeline

```
src/extract_dogflw.py        Kaggle zip -> JPEG + one annotations.json (coords unchanged)
src/build_dogboxes.py        SuperAnimal detector -> whole-dog box; SuperAnimal pose -> 39 kpts
src/analyze_correspondence.py which DogFLW landmarks already exist in SuperAnimal (data-driven)
src/make_coco.py             COCO dataset, 39 + N channels, per-keypoint supervision flags
src/extend_head.py           grow the heatmap head 39 -> 39+N, copying pretrained channels
src/train_dogface.py         fine-tune with DeepLabCut's own trainer (COCOLoader + apis.train)
src/evaluate.py              NME / PCK on the DogFLW test split + forgetting check
src/run_video.py             detector -> pose -> labelled MP4 (side-by-side before/after)
src/make_figures.py          keypoint-scheme, supervision and qualitative-result figures
```

Run it all with `./run_pipeline.sh` (assumes `src/extract_dogflw.py` has already been run).

### Three decisions worth calling out

**Whole-dog crops, not face crops.** The model is top-down: at inference a detector supplies
a whole-animal box and the pose head sees that crop. Training crops must match, so every
DogFLW image is run through the SuperAnimal detector and the dog box — not the DogFLW face
box — becomes the training crop. Face landmarks are annotated in image coordinates, so they
transfer to any crop unchanged.

**Generalised memory replay.** Each training sample carries `39 + N` keypoint channels:
DogFLW ground truth supervises the face, the pretrained model's own predictions supervise
the body, and channels the pretrained model is not confident about are given COCO visibility
`-1`, which makes DeepLabCut's `HeatmapGaussianGenerator` zero their loss weights entirely —
so training neither invents a label nor pushes the channel to background.

**Warm-started new channels.** The head is a single 1×1 `ConvTranspose2d` over the 32-channel
HRNet feature map. The 39 pretrained channels are copied verbatim; each added channel is
initialised from the filter of its spatially nearest SuperAnimal keypoint, so it starts out
predicting a nearby point rather than noise.

## Environment notes

**CPU-only on both machines it has run on** — a Windows 11 / Ryzen box (the current one;
PyTorch's ROCm builds are Linux-only, so the iGPU is unusable) and the original aarch64
Asahi Linux box. `device="cpu"` throughout is deliberate, not a leftover.

DeepLabCut 3.0.1 pins `numpy<2` and is installed with `--no-deps` to dodge that pin; the
PyTorch engine runs fine against numpy 2. Full per-platform install steps, including the
Windows `timm` and dataloader-worker gotchas, are in **CLAUDE.md §8**. To just run the app,
use **[SETUP.md](SETUP.md)**.
