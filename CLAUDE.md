# CLAUDE.md — handoff notes

Project state as of **2026-08-26, 18:30 EDT**. Work was **stopped mid-training on purpose**
for this handoff. Read "Status" before running anything.

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

### NOT done — this is the remaining work
| item | state |
|---|---|
| **A trained model** | **Does not exist.** Phase 1 was killed at iteration 375/482, before its end-of-epoch save. `dlc_project/phase1/` has a config and a log but **no `.pt`**. |
| Phase 2 (backbone `stage4` fine-tune) | never started |
| Final evaluation of a fine-tuned model | never run |
| The demo video | **never produced** — this is the actual deliverable the prompt asked for |
| Figure 3 (qualitative test predictions) | never run; needs a trained snapshot |
| Whether the added keypoints actually *improve* | **unknown and unvalidated** — see §7 |

The only model file present is the **untrained** extended one. Running the video script
against it today will render a dog with 39 correct body keypoints and 37 face keypoints
that are just copies of their warm-start donors.

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
and a fine-tuned model must be compared against this row, not against zero. The whole
question this project has not yet answered is whether training clears that bar.

The `0.0000` drift row is meaningful: it proves the head surgery preserved the original
39 channels exactly.

---

## 8. Running it

### Setup (already done in `.venv/`; these are the steps to rebuild)
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
2. **Phase 2's ~20 min/epoch is an estimate that was never measured.** It was extrapolated
   from Phase 1's frozen-backbone rate assuming backward through `stage4` costs ~2× a
   forward pass. Measure the first 25 iterations before trusting any schedule built on it.
   The earlier estimates in this project were wrong twice, both times because the
   benchmark ran while something else was using the cores — benchmark on a quiet machine.
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
