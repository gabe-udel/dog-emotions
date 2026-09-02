# CLAUDE.md — handoff notes

Project state as of **2026-09-02**. Training is **finished and validated**; the model in use
is `superanimal_quadruped_dogface_final.pt` (σ=17). Read §2 for what is and is not done, and
**§12 for the current measured accuracy** — that section is the stable reference point, cut
just before the planned face/body model separation.

---

## 1. The original request

Verbatim prompt this project was built from:

> download the deeplabcut superanimalquadraped model (most recent iteration). then,
> re-train the model to add addiitonal face keypoints from the docFLW dataset. you may
> use the entire dataset to re-train. then, downlaod a sample video of dog walking and
> run the new model (with teh face keypoints added) and save to demonstrate your work.

Two things were interpreted rather than stated, and a peer should sanity-check both:

- **"docFLW" was read as DogFLW** ("Dog Facial Landmarks in the Wild", Martvel et al.).
  Nothing called "docFLW" exists; DogFLW is a dog-face landmark dataset, which fits
  "add face keypoints" exactly. Confident, but it is an inference.
- **"most recent iteration"** was read as *the checkpoint the current DeepLabCut release
  actually resolves* for `superanimal_quadruped`, i.e. `hrnet_w32`. See §3.

---

## 2. Status: what is and is not done

### Built and verified
| item | evidence |
|---|---|
| SuperAnimal-Quadruped weights downloaded | `hrnet_w32` (118 MB) + 2 detectors, in `.venv/.../deeplabcut/modelzoo/checkpoints/` |
| DogFLW downloaded and extracted, all 4,335 images | `data/dogflw/`, `annotations.json` |
| SuperAnimal run over all 4,335 images (dog box + 39 kpts) | `data/sa_dogboxes.npz`, 42 min runtime |
| DogFLW↔SuperAnimal correspondence derived from data | `data/keypoint_map.json` |
| COCO training set, 76 keypoints | `dlc_project/annotations/{train,test}.json` |
| Head surgery 39→76 outputs | `model_weights/superanimal_quadruped_hrnet_w32_dogface.pt` |
| Head surgery is **bit-exact** for the original 39 | baseline eval: SuperAnimal drift `0.0000`, identical confidences |
| Pre-training baseline on the full test split | `outputs/evaluation_baseline.json` |
| Training loop runs end-to-end | 375 iterations completed before it was killed |
| Video pipeline runs end-to-end | smoke-tested on 8 frames; detector found the dog 8/8 |
| Eval script runs end-to-end | produced the baseline JSON over 479 test images |
| Figures 1 and 2 | `outputs/figures/` |

### Trained and validated — 2026-08-30
Training ran. Three checkpoints exist in `model_weights/` (gitignored, 112 MB each;
rebuild with `extend_head.py` + `train_dogface.py`). Full test split, 479 held-out images:

| checkpoint | added NME ↓ | PCK@5% | PCK@10% | SuperAnimal drift |
|---|---|---|---|---|
| `..._hrnet_w32_dogface.pt` — untrained warm start | 0.0754 | 39.4% | 73.3% | 0.0000 |
| **`..._dogface_final.pt` — σ=17, 4 epochs — the one in use** | **0.0583** | 57.0% | 83.1% | 0.0000 |
| `..._dogface_sigma8.pt` — σ=8, 7 epochs | 0.0630 | 54.4% | 79.7% | 0.0000 |

Drift `0.0000` throughout: memory replay held, the 39 body keypoints are untouched.

**σ=8 was an experiment that failed on accuracy.** Sharpening `pos_dist_thresh` 17→8 was
predicted to help dense facial landmarks and made NME worse. It *is* better calibrated
(bad/good confidence ratio 0.46 vs 0.75), but it expresses that by scoring its weak
channels so low they vanish from the display — 33.5 of 46 points drawn against σ=17's
46.0. Keep σ=17 as the default.

Its per-epoch curve also showed **train loss going flat while test NME was still
falling**. Train loss is not a stopping signal on this project; use `--eval-every 1`.

### Then: four inference-time fixes, no retraining (§11)
The rendered output was still poor, and every cause turned out to be fixable after the
CNN rather than inside it. See §11 — this is where most of the accuracy now comes from.

### Still not done
| item | state |
|---|---|
| Figure 3 (qualitative test predictions) | never run |
| Two-model split (face + body, dual inference) | **the next planned change.** Branch `stable-pre-model-separation` marks the state before it; §12 has the numbers it must beat |
| The DogFLW landmark manual | still behind a dead link; §6 naming remains inferred |

---

## 3. Which SuperAnimal checkpoint, and why

`mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped` on HuggingFace contains more pose
checkpoints than DeepLabCut ships support for. As of dlclibrary 0.0.12, the registry
(`dlcmodelzoo/modelzoo_urls_pytorch.yaml`) lists only `hrnet_w32`, `resnet_50`, `rtmpose_s`
for `superanimal_quadruped`. `hrnet_w48`, `rtmpose_m` and `rtmpose_x` exist in the repo
(uploaded 2025-06-30) but are **not registered**, so `hrnet_w32` is what a current
DeepLabCut install resolves and is what this project uses.

If "most recent iteration" was meant literally as *newest file in the repo*, that reading
points at `hrnet_w48` instead, and would need `superanimal.POSE_MODEL` changed plus a
manual download — `dlclibrary` will not resolve it. Worth confirming with whoever asked.

Detector: `fasterrcnn_mobilenet_v3_large_fpn`. Benchmarked against
`fasterrcnn_resnet50_fpn_v2` on 12 DogFLW images: mobilenet was **0.31 s/img and found the
dog in 12/12**; resnet50-v2 was **3.64 s/img and found only 9/12** (DogFLW images are
face close-ups, which the heavier detector handles worse). 12× faster *and* better recall
here — but that comparison was 12 images, so treat it as indicative, not settled.

---

## 4. Why the stock DeepLabCut path does not work

DeepLabCut ships "memory replay" for fine-tuning a SuperAnimal model without catastrophic
forgetting. **It cannot add keypoints.** Two independent blocks:

1. `pose_estimation_pytorch/modelzoo/memory_replay.py::prepare_memory_replay` maps
   *every* project bodypart into SuperAnimal's 39-keypoint space:
   ```python
   for idx, bpt in enumerate(bodyparts):
       conversion_table[bpt] = sa_bodyparts[weight_init.conversion_array[idx]]
   ```
2. `pose_estimation_pytorch/models/model.py::PoseModel.build` loads the pretrained head
   with `head.load_state_dict(...)` after indexing it by the conversion array, so the head
   must have exactly `len(conversion_array)` outputs.

So this project **generalises** memory replay rather than calling it. Everything else —
model construction, dataloading, augmentation, the training loop, inference runners — is
stock DeepLabCut.

---

## 5. The pipeline

```
src/extract_dogflw.py         Kaggle zip -> JPEG + one annotations.json (coords unchanged)
src/build_dogboxes.py         SuperAnimal detector -> whole-dog box; SuperAnimal pose -> 39 kpts
src/analyze_correspondence.py which DogFLW landmarks already exist in SuperAnimal
src/make_coco.py              COCO dataset, 76 channels, per-keypoint supervision flags
src/extend_head.py            grow the heatmap head 39 -> 76
src/train_dogface.py          fine-tune via DeepLabCut's COCOLoader + apis.training.train
src/evaluate.py               NME / PCK on the DogFLW test split + forgetting check
src/run_video.py              detector -> pose -> labelled MP4 (side-by-side before/after)
src/make_figures.py           figures
src/superanimal.py            thin wrappers for loading the pretrained model
src/keypoint_scheme.py        names for the 46 DogFLW landmarks
src/draw.py                   skeleton, palette, frame rendering
```

### Four design decisions a peer should understand

**(a) Whole-dog crops, not face crops.** The model is top-down: at inference a detector
supplies a whole-animal box and the pose head sees that crop. Training crops must match,
so every DogFLW image was run through the SuperAnimal detector and the *dog* box — not the
DogFLW face box — is the training crop. Result: detector box containing the annotated face
in 4,011 images, detector-only 318, grown-face-box fallback 6. DogFLW landmarks are in
image coordinates so they transfer to any crop unchanged. Face occupies a median 62% of
the crop's long side (158 px of 256, i.e. 39 px on the 64×64 heatmap grid).

**(b) Generalised memory replay.** Each sample carries 76 keypoint channels:
- DogFLW ground truth supervises the face → COCO visibility `2`
- the pretrained model's own predictions supervise the body → visibility `2`, gated at
  score ≥ `PSEUDO_THRESH = 0.4`
- anything below that gate → visibility `-1`

Visibility `-1` matters: `HeatmapGaussianGenerator` sets that channel's loss weights to
**0** (`heatmap_targets.py`, the `keypoint[-1] == -1` branch), so training neither invents
a label nor pushes the channel toward background. Final split: **60.5% ground truth,
22.9% pseudo-label, 16.6% masked** (199,378 / 75,363 / 54,719).

**(c) Data-derived correspondence.** Rather than trusting a published landmark figure, the
pretrained model was run over all 4,335 faces and each SuperAnimal keypoint matched to its
nearest DogFLW landmark. Merge requires mutual-nearest-neighbour AND median separation
< 0.05 of the face-bbox diagonal. **9 merged, 37 added → 76 outputs.**

```
nose            <-> 32 nose_bottom          0.014
upper_jaw       <-> 38 mouth_center         0.015
lower_jaw       <-> 42 chin_bottom          0.023
mouth_end_right <-> 39 mouth_corner_right   0.046
mouth_end_left  <-> 40 mouth_corner_left    0.043
right_eye       <-> 22 eye_right_lower_lid  0.020
left_eye        <-> 23 eye_left_lower_lid   0.018
right_earend    <->  6 ear_right_tip        0.023
left_earend     <->  7 ear_left_tip         0.022
```

The 4 antler keypoints are excluded from merging (`NO_MERGE` in
`analyze_correspondence.py`): on a dog the model puts `antler_base` on top of `earbase`, so
they win the mutual-NN test by accident, and attaching dog annotations to an "antler"
channel would be meaningless. They stay in the model, supervised by pseudo-labels.

**(d) Warm-started new channels.** The head is a single 1×1 `ConvTranspose2d` over the
32-channel HRNet map. The 39 pretrained channels are copied verbatim; each added channel is
initialised from the filter of its spatially nearest SuperAnimal keypoint (+1e-3 noise), so
it starts predicting a nearby point rather than noise.

**This warm start is also why the baseline is not zero** — see §7.

---

## 6. Landmark naming is inferred, not authoritative

DogFLW ships coordinates only. The landmark manual is in supplementary materials behind a
dead link shortener (`rb.gy/r6srv9` → 403). The names in `src/keypoint_scheme.py` were
derived from the data: each face normalised into its own bbox and averaged over all 4,335
images gives a near-perfectly bilaterally symmetric mean shape — 20 mirror pairs and 6
midline points — and the groups were read off that.

**Indices are exact; names are inference.** Two independent cross-checks passed: the
pretrained model's `right_eye` matched the point independently labelled
`eye_right_lower_lid`, and `right_earend` matched `ear_right_tip` — which also confirms
"right" = the animal's right = image-left, matching SuperAnimal's convention.

Anyone publishing off this should still get the real manual from the authors.

---

## 7. The trap in the numbers

`outputs/evaluation_baseline.json` — full 479-image test split, **untrained** extended model:

| | NME ↓ | PCK@5% | PCK@10% |
|---|---|---|---|
| 37 added face keypoints | 0.0754 | 39.4% | 73.3% |
| 9 kept SuperAnimal face keypoints | 0.0377 | | |
| SuperAnimal body-keypoint drift | 0.0000 | | |

**Do not read the 73.3% as a result.** The added channels are warm-start copies of their
donor keypoints, so this measures "how close is the nearest existing SuperAnimal keypoint
to this DogFLW landmark", not any learned ability. **0.0754 is the bar training must beat**,
and a fine-tuned model must be compared against this row, not against zero.

**Training cleared it** (0.0583, −22.7%), and the drift row stayed `0.0000`, proving head
surgery preserved the original 39 channels exactly.

### The second trap, which cost more
Mean NME across all 46 landmarks is a bad way to rank checkpoints here, for two reasons
that both bit:

1. **It weights a garbage ear point exactly like an eye corner.** The error was never
   spread across the face — per region, eye 0.0278 / nose 0.0301 / mouth 0.0344 against
   ear 0.0890 / head-top 0.1250. Training moved nose and mouth 30-47% and the ear/head
   group 6-8%.
2. **It ignores confidence entirely**, so it ranked the model that *hides* its failures
   below the one that displays them.

And an aggregate can hide the mechanism outright: ear error looked uniformly mediocre
until it was split by ear type, at which point it **inverted** — `ear_*_tip` scores 0.0485
on erect ears and 0.102 on floppy, while `ear_*_outer_base` is 0.130 erect against 0.072
floppy. Averaged together those cancel to a 1.11× ratio and look like noise. They are not
noise; see §11.

---

## 8. Running it

### Windows setup (this machine — 2026-08-26)

The project was ported from the original aarch64 Asahi box to **Windows 11, AMD Ryzen AI 5
340 (6 physical / 12 logical cores), 15.2 GB RAM, Radeon 840M integrated graphics**.
Still **CPU-only**: PyTorch's ROCm builds are Linux-only, so the iGPU is not usable and
`device="cpu"` throughout is correct, not a leftover.

Python **3.11** (not 3.13): the `--system-site-packages` + `numpy<2` dance in the next
section existed only because numpy<2 had no cp313 aarch64 wheel. On Windows/cp311 that
constraint is gone, but the venv still ends up on **numpy 2.4** because CPU torch pulls it,
so the `--no-deps` install is still the right shape:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install --no-deps deeplabcut==3.0.1
.venv\Scripts\python.exe -m pip install "dlclibrary>=0.0.12" matplotlib einops filterpy networkx `
    pydantic tqdm imageio-ffmpeg scikit-learn scikit-image statsmodels tables pycocotools `
    numba "albumentations<=1.4.3"
.venv\Scripts\python.exe -m pip install timm
```

`timm` is **explicit here**. On the original box it came from system site-packages; a
Windows venv has no system packages to inherit, and DeepLabCut's HRNet backbone needs it.

`imgaug` is declared by DeepLabCut but deliberately **not installed** — it is a
TensorFlow-era dependency, `import deeplabcut` succeeds without it, and the original
`requirements-venv.txt` omits it too. pip will print a conflict warning about it, plus
numpy/matplotlib/pandas/filelock. All five are expected under `--no-deps` and match the
reference env.

Verify: `.venv\Scripts\python.exe -c "import deeplabcut; print(deeplabcut.__version__)"`

**Windows-specific code changes** (all committed here):

| file | change | why |
|---|---|---|
| `src/extract_dogflw.py` | `/tmp/dogflw.zip` → `--zip`, defaulting to `data\dogflw.zip` and auto-finding a `*dogflw*.zip` in `~/Downloads` | no `/tmp` on Windows |
| `src/run_video.py` | `/tmp/dlc_frames` → `tempfile.gettempdir()` | same |
| `src/train_dogface.py` | `dataloader_workers` hardcoded 2 → `--workers`, default **0 on Windows** | Windows has no `fork()`; under `spawn` each worker re-imports the module, and albumentations/OpenCV in spawned workers is a known hang risk |
| `src/superanimal.py` | `DETECTOR` default resnet50_fpn_v2 → mobilenet | every caller already overrode it; the old default made `snapshot_paths()` download a ~170 MB detector nothing uses |
| `run_pipeline.ps1` | new — native port of `run_pipeline.sh` | `.venv/bin/python`, `ls -t`, `cp`, `tee` are POSIX; PowerShell 5.1 also has no `&&`, so native exit codes need explicit checks |

Run the pipeline with `.\run_pipeline.ps1` (params: `-P1Epochs`, `-P2Epochs`, `-BatchSize`,
`-Threads`, `-Workers`). `run_pipeline.sh` is left in place for the Linux box.

Weights cache to `.venv\Lib\site-packages\deeplabcut\modelzoo\checkpoints\` and persist —
the `dlc_hf_*` temp dirs in the download log are staging only. HuggingFace warns about
symlinks on Windows; harmless, silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

---

### Original setup, aarch64 Linux (kept for the other box)
The environment is awkward and the reasons are load-bearing: this box is **CPU-only aarch64
(Asahi Linux), no CUDA**, on Python 3.13. DeepLabCut 3.0.1 pins `numpy<2`, which has no
cp313 wheel and would try to build from source. So:

```bash
python3 -m venv --system-site-packages .venv     # reuse system torch / numpy 2.4 / timm
.venv/bin/python -m pip install --no-deps deeplabcut==3.0.1
.venv/bin/python -m pip install "dlclibrary>=0.0.12" matplotlib einops filterpy networkx \
    pydantic tqdm imageio-ffmpeg scikit-learn scikit-image statsmodels tables pycocotools \
    numba "albumentations<=1.4.3"
```
`--no-deps` is what dodges the numpy pin. The PyTorch engine runs fine against numpy 2;
the pin is legacy from the TensorFlow engine. `requirements-venv.txt` lists what ended up
installed, but **do not `pip install -r` it directly** — it omits the system packages.

Verify: `.venv/bin/python -c "import deeplabcut; print(deeplabcut.__version__)"`

### One-shot
```bash
./run_pipeline.sh          # steps 1-7: correspondence -> COCO -> surgery -> train -> eval -> video
```
Override epochs: `P1_EPOCHS=1 P2_EPOCHS=3 ./run_pipeline.sh`

`run_pipeline.sh` assumes `src/extract_dogflw.py` and `src/build_dogboxes.py` have already
run. Both have (their outputs are on disk), and `build_dogboxes.py` takes ~42 min, so do
not re-run it casually.

### Step by step (what a peer probably wants)
```bash
# Steps 1-3 are done; outputs are on disk. Re-run only if you change the merge rules.
.venv/bin/python src/analyze_correspondence.py     # -> data/keypoint_map.json
.venv/bin/python src/make_coco.py                  # -> dlc_project/annotations/*.json
.venv/bin/python src/extend_head.py                # -> model_weights/..._dogface.pt

# Phase 1 - head only, backbone frozen. ~1.2 s/iter, 482 iters/epoch => ~10 min/epoch
OMP_NUM_THREADS=8 .venv/bin/python src/train_dogface.py \
  --run-name phase1 --epochs 1 --batch-size 8 --unfreeze none \
  --lr-head 1e-3 --save-epochs 1 \
  --snapshot model_weights/superanimal_quadruped_hrnet_w32_dogface.pt

# Phase 2 - HRNet stage4 + head. ~20 min/epoch (estimated, NEVER MEASURED)
OMP_NUM_THREADS=8 .venv/bin/python src/train_dogface.py \
  --run-name phase2 --epochs 3 --batch-size 8 --unfreeze stage4 \
  --lr-backbone 1e-5 --lr-head 2e-4 --save-epochs 1 \
  --snapshot dlc_project/phase1/snapshot-001.pt

# Phase 2 RESUMES from epoch 1, so with --epochs 3 it writes snapshots 002/003/004 and
# max_snapshots=2 keeps only the last two. Grab the newest rather than hardcoding a number:
FINAL=$(ls -t dlc_project/phase2/snapshot-*.pt | head -1)

# Evaluate against the baseline in outputs/evaluation_baseline.json
.venv/bin/python src/evaluate.py \
  --config dlc_project/phase2/pytorch_config.yaml \
  --snapshot "$FINAL" \
  --out outputs/evaluation.json

# The deliverable video (frames 36-216 = 1.5-9.0 s; outside that the clip is
# blown out by lens flare at the start and the dog turns away at the end)
.venv/bin/python src/run_video.py --video videos/mixkit_1476.mp4 \
  --out outputs/dog_walk_comparison.mp4 \
  --also-solo outputs/dog_walk_dogface.mp4 \
  --config dlc_project/phase2/pytorch_config.yaml \
  --snapshot "$FINAL" \
  --compare --width 960 --smooth 3 --start-frame 36 --max-frames 180
```

`--smooth 3` is a 3-frame temporal median, cosmetic only — pass `--smooth 1` to see raw
per-frame output.

### Two footguns that cost real time here
- **`pkill -f train_dogface` kills your own shell**, because the pattern matches the
  invoking bash command line. Use
  `ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /dogface/ {print $1}' | xargs -r kill`.
- **DeepLabCut reports progress through `logging`.** Without
  `setup_file_logging(model_folder / "log.txt")` the training run is completely silent and
  looks hung. `train_dogface.py` now calls it; anything new built on `apis.training.train`
  must too.

---

## 9. Known issues / next steps, roughly in priority order

1. **Train the model.** Nothing downstream is meaningful until Phase 1 + Phase 2 finish.
2. ~~**Phase 2's ~20 min/epoch is an estimate that was never measured.**~~ **MEASURED
   2026-08-26** on the Windows box (6 cores, CPU). The old guess assumed backward through
   `stage4` costs ~2× a forward pass; the real ratio is **1.84×**, so the estimate was
   sound but slightly pessimistic.

   Relative cost of each lever, HRNet-w32 @ 76 outputs, forward+backward (ratios are the
   trustworthy part — these synthetic runs come out ~30% slower than a real training
   iteration, which measured **0.80 s/iter** for phase 1):

   | lever | setting | rel. cost |
   |---|---|---|
   | crop | 256 / 224 / 192 / 160 | 1.00 / 0.75 / 0.56 / 0.41 |
   | batch (per *image*) | 4 / 8 / 16 | 1.03 / 1.00 / 0.78 |
   | backbone | none / stage4 / stage3,4 / all | 1.00 / 1.84 / 2.69 / 3.41 |

   Applying the 1.84× to the real 0.80 s/iter gives **phase 2 ≈ 1.5 s/iter, ~11.8
   min/epoch** — so 3 phase-2 epochs is ~35 min, not the ~60 the old estimate implied.

   Keypoint *count* is not a lever: trimming 76→63 outputs saves 429 parameters
   (0.0015% of the model) and 2.6 ms of an 800 ms iteration. The head is a 1×1 conv; the
   29.3M-parameter backbone is the entire cost.

   Note on crop: it is the largest raw lever but **not free**. 256×256 is what
   `get_inference_runners` uses for the released SuperAnimal config, so lowering it makes
   training crops disagree with inference crops *and* with the crops the memory-replay
   pseudo-labels were generated on. It also shrinks the heatmap grid (64→48 px at crop
   192), and the face already occupies only ~39 px of that grid — bad for 46 dense
   landmarks. Batch 16 is the lever with no correctness cost; scale the LR by ~√2 with it.

   Still true: benchmark on a quiet machine. The earlier estimates in this project were
   wrong twice, both times because something else was using the cores.
3. **Validate against the §7 baseline**, not against zero.
4. **Hyperparameters are unswept.** `lr-head 1e-3`/`2e-4`, `lr-backbone 1e-5`,
   `PSEUDO_THRESH 0.4`, `MERGE_THRESH 0.05`, 3 epochs — all first guesses, none tuned.
5. **`pos_dist_thresh=17`** gives a Gaussian σ ≈ 11.3 px in the 256×256 crop. That is
   SuperAnimal's own setting and was deliberately left alone for consistency with the
   pretrained head, but it is likely too wide for 46 dense facial landmarks. Sharpening it
   is the most promising accuracy lever — it would also retrain the 39 existing channels
   toward sharper heatmaps, so evaluate the forgetting metric if you touch it.
6. **The detector was not fine-tuned.** `run_pipeline.sh` uses the stock SuperAnimal
   detector. Fine, since the box definition has not changed.
7. **Disk is at 99% (1.4 GB free).** Each snapshot is 118 MB; `max_snapshots` is 2 per run
   folder. A long run with `--save-epochs 1` will fill the disk.
8. **Only one demo video.** `videos/dog_walk_sf.ogv` (Wikimedia, CC BY-SA 3.0) is a second
   candidate, but the dog is small and dark.

## 10. Licences

DogFLW is **CC BY-NC 4.0 — non-commercial**. Mixkit clip 1476 is Mixkit Free License.
`dog_walk_sf.ogv` is CC BY-SA 3.0. SuperAnimal weights follow the DeepLabCut Model Zoo
terms. The non-commercial restriction on DogFLW propagates to any model trained on it.

---

## 11. Four inference-time fixes — where the accuracy actually came from

After training, the rendered video was still poor. Every cause turned out to be fixable
**after the CNN rather than inside it**, and together these matter more than the
fine-tuning did. All are on by default in `run_video.py`; each has an opt-out flag.

### (a) `src/subpixel.py` — the largest bug in the project
`HeatmapPredictor.get_pose_prediction` decodes each keypoint as `torch.argmax` over the
64×64 heatmap and then adds a learned offset **from the locref map**. These models were
trained with `generate_locref: false`, so that offset is always `None` and every keypoint
snaps to a cell centre — 4 px in the 256 crop.

For dense facial landmarks that is fatal: two landmarks peaking in the same cell decode to
*byte-identical* coordinates and render as one dot. Measured on video, the 46 face
channels occupied **16 distinct pixel positions**.

Fitting a parabola through each peak and its neighbours (standard HRNet / DARK
post-processing) gives **44 of 46 distinct**, and ~7% better NME everywhere because the
rounding was real error, not just a display artefact. Disable with `--no-subpixel`.

### (b) `src/ear_correct.py` — ear NME 0.0882 → 0.0631 (−28.5%)
One output channel serves both erect and floppy ears, so the model averages the two
geometries. That makes its ear error **systematic and opposite by type** (§7), and a
systematic error can be subtracted.

Two pieces, both fitted on the train split: a logistic classifier that recovers ear type
from the model's *own* predicted ear landmarks (80% on held-out data, 51.8% majority
baseline — nothing external needed at inference), and a per-type mean residual in a
similarity frame defined by the 30 reliable landmarks. Every ear landmark improved 12–42%.
Predicted type matches an oracle type to within 0.0002, because misclassifications land
between adjacent types whose biases are similar. `--no-ear-correct`; refit with
`python src/ear_correct.py --fit --eval`.

### (c) `src/shape_refine.py` — skull top, NME 0.130 → 0.074 (−43%)
`head_top_left/right` sit on featureless fur: PCK@5% of 0.6%, i.e. chance. But *no local
texture* is not *undetermined* — the crown is implied by the rest of the head. A ridge
shape model on the 30 reliable landmarks derives them from the CNN's own predictions.

The same trick **fails on ears** (0.100 → 0.180): an ear can be perked or flopped
independently of the face, so geometry cannot predict it. Only landmarks that are globally
determined *and* locally featureless belong in `TARGET`. `--no-refine`.

### (d) Two-pass head crop — available, off by default
Re-cutting the box so the face fills 55% of it (the measured optimum; a whole-dog box on
video gives ~25%) is worth ~18% NME on the reliable landmarks. It is **off by default**
because it costs more than it buys for a human viewer: it lowers confidence, and with
sub-pixel decoding on it drops distinct face points from 44 to 25. Use it when measuring,
not when looking. `--head-crop 0.55`, gated by `--gate scale`.

### What generalised
Three of these came from the same move: **an error that looks like noise in aggregate is
often systematic once conditioned on the right variable.** Before calling a landmark
unlearnable, check whether its error has a consistent direction given something
observable. `keypoint_scheme.UNRELIABLE` was 14 landmarks, then 2, and is now empty.

Also worth carrying: on this model **confidence runs opposite to accuracy** across crop
scales (0.807 at the worst, 0.589 at the best). It is a visibility control, never a
quality signal — do not gate or tune anything by watching it rise.
---

## 12. Stable landmark — branch `stable-pre-model-separation`

Cut **2026-09-02**, immediately before splitting the single shared-backbone model into
separate face and body networks. Everything below is measured on this branch, not
projected. **If the two-model architecture does not beat these numbers, come back here.**

Branch name note: the request was `stable - pre model seperation`; git forbids spaces in
ref names, so it is hyphenated, and "separation" is spelled correctly.

### What "stable" means, exactly

| | |
|---|---|
| base | SuperAnimal-Quadruped `hrnet_w32` + `fasterrcnn_mobilenet_v3_large_fpn`, DLC 3.0.1 / dlclibrary 0.0.12 |
| checkpoint | `superanimal_quadruped_dogface_final.pt` — σ=17, 4 epochs |
| architecture | **one** shared HRNet-W32 backbone (29,363,185 params) → **one** 1×1 `ConvTranspose2d(32→76)` head (2,508 params). 99.99% of the model is shared between face and body; a keypoint is a 32-number column. |
| inference stack | sub-pixel decode → ear-type bias correction → skull-top shape model → 3-frame temporal median. Head crop **off**. |
| fitted params, in repo | `data/ear_bias.pkl`, `data/shape_model.npz` — nothing needs refitting |

### Accuracy — 479 held-out DogFLW test images, shipping config

Reproduce with the scratch script pattern in §8; ear correction and shape refine both
applied on 479/479.

| region | n | NME ↓ | median | PCK@5% | PCK@10% | before ear/shape fix | change |
|---|---|---|---|---|---|---|---|
| eye | 8 | 0.0261 | 0.0252 | 96.7% | 99.6% | 0.0261 | — |
| nose | 7 | 0.0269 | 0.0225 | 90.2% | 99.5% | 0.0269 | — |
| mouth | 11 | 0.0325 | 0.0234 | 81.9% | 96.3% | 0.0325 | — |
| muzzle | 4 | 0.0505 | 0.0506 | 49.0% | 98.6% | 0.0505 | — |
| ear | 14 | 0.0631 | 0.0472 | 53.2% | 84.0% | 0.0882 | **−28.5%** |
| head (skull top) | 2 | 0.0880 | 0.0884 | 8.9% | 65.1% | 0.1240 | **−29.0%** |
| **ALL 46** | 46 | **0.0438** | 0.0319 | **71.0%** | **92.5%** | 0.0531 | **−17.4%** |
| 37 added channels | 37 | 0.0458 | 0.0342 | 68.4% | 91.9% | | |
| 9 merged channels | 9 | 0.0357 | 0.0215 | 81.4% | 94.9% | | |

Read the last column carefully: it is **ear correction + shape model only**. Sub-pixel
decoding (§11a) is already inside *both* columns, so its contribution is not shown here —
measured separately it was NME 0.0329 → 0.0306 and face points 16 → 44 distinct.

Best landmark `philtrum` 0.0117; worst `head_top_right` 0.0903. SuperAnimal body drift
remains **0.0000**.

### What is not on this branch

The checkpoint (113 MB, over GitHub's per-file limit — it is a Release asset), DogFLW
imagery (CC BY-NC 4.0, not ours to redistribute), `dlc_project/`, `.venv/`, demo videos.
Everything needed to *run* the app is here except the checkpoint; see SETUP.md.

### The bar the next architecture has to clear

§11d is the only direct evidence for what a face-specific model buys: feeding tighter
crops to *this* network was worth **~18% NME on the reliable landmarks**. Two caveats to
judge the split against:

- It **understates** the gain. This model was trained on whole-dog crops, so a tight crop
  is out of distribution — that is why confidence fell and visible face points dropped
  44 → 25. A network actually trained on face crops pays no such penalty.
- It **does not touch the two worst regions**, which are 16 of the 46 landmarks. Ear error
  moves 4% across a 3× crop range but 20% by ear type; the skull top has no local texture
  at any resolution. Both were fixed after the CNN (§11b, §11c) and those fixes carry over
  to any new architecture unchanged — so a face model should be compared against
  **0.0438**, not against the raw CNN's 0.0531.

