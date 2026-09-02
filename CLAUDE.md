# CLAUDE.md — handoff notes

Project state as of **2026-09-02**. The unified 76-channel model has been **replaced by a
two-stage cascade**. The code is written and the data is built; **the face model has not
been trained yet** — that is the next step and it was deliberately not run.

The previous architecture, with its full results, is frozen on the branch
**`stable-pre-model-separation`**. If the cascade does not beat it, go back there.

---

## 1. The original request

> download the deeplabcut superanimalquadraped model (most recent iteration). then,
> re-train the model to add addiitonal face keypoints from the docFLW dataset. you may
> use the entire dataset to re-train. then, downlaod a sample video of dog walking and
> run the new model (with teh face keypoints added) and save to demonstrate your work.

Two interpretations a peer should sanity-check:

- **"docFLW" was read as DogFLW** ("Dog Facial Landmarks in the Wild", Martvel et al.).
  Nothing called "docFLW" exists. Confident, but it is an inference.
- **"most recent iteration"** was read as *the checkpoint the current DeepLabCut release
  actually resolves*, i.e. `hrnet_w32`. See §3.

---

## 2. Architecture

```
frame
  -> SuperAnimal detector (Faster R-CNN)        -> whole-dog box
  -> SuperAnimal-Quadruped pose, STOCK/FROZEN   -> 39 body keypoints
  -> derive_face_box() from the head anchors    -> face box
  -> crop + resize to 256x256                   -> face crop
  -> DogFLW face model                          -> 46 landmarks
  -> CropTransform.to_image                     -> image coordinates
```

Stage 1 is the released checkpoint run unmodified. **No head surgery, no fine-tuning of
SuperAnimal, no memory replay, and no forgetting to measure** — the body weights are the
published ones, so the drift metric the old architecture needed does not apply.

### The invariant

`src/facebox.py::derive_face_box` is the only place a face box is produced, and both the
training data builder and video inference call it with the same config. Training on
DogFLW's shipped face boxes and inferring on derived ones would mean the face model never
sees its deployment box distribution. `tests/test_facebox.py` asserts both call sites
resolve to the same function object, and that no second implementation exists in `src/`.

Boxes are derived from SuperAnimal's *predictions* even at training time, so first-stage
error is part of the training distribution rather than a deployment surprise.

### Why the cascade

The old model shared **99.99% of its parameters** between body and face: one HRNet-W32
backbone (29,363,185 params) into a single 1x1 `ConvTranspose2d` head (2,508 params), so
every keypoint was a 32-number linear readout of the same feature vector.

Evidence that this was binding, all measured on this project:

| finding | number |
|---|---|
| within-image crop sweep, 30 reliable landmarks, at 25% / 45% / 55% face fill | NME 0.0411 / 0.0362 / 0.0328 — **monotonic** |
| face span on the 64x64 heatmap grid, DogFLW | median 35.9 cells (p5 21.4, p95 60.4) |
| face span on video (whole standing dog) | ~16 cells |
| argmax decoding collapsed 46 face channels onto | **16 distinct pixels** |

Note the sweep is monotonic across everything tested — 55% is the *highest fill measured*,
not a measured optimum. A face-filling crop extrapolates past the end of it.

### What the rebuild actually bought, measured

Do not overstate the resolution gain. On the built dataset:

| | old (whole-dog crop) | new (derived face box) |
|---|---|---|
| face span, DogFLW test | median 35.9 cells | **median 39.7** (p25 36.8, p75 44.2) |
| face span, video | ~16 cells | ~40, independent of dog distance |
| spread across the split | 21 → 60 cells | 37 → 44 cells |
| landmark supervision | 60.5% GT / 22.9% pseudo / 16.6% masked | **99.8% ground truth** |
| box derivation failures, train | n/a | 1 fallback, 0 failures in 3,279 |

On DogFLW the resolution gain is only **+11%**, because `pad=1.8` spends most of the crop
guaranteeing a flopped ear stays inside it. The gains that are large are the **variance
collapse** (scale is now normalised) and the **supervision jump**. On video, where the old
model got ~16 cells, the change is ~2.5x.

`--pad` is a config field so it can be swept on validation: pad 1.4 would give ~51 cells
but contains all 46 landmarks in only 78.6% of images, against 97.7% at 1.8.

---

## 3. Which SuperAnimal checkpoint, and why

`mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped` on HuggingFace holds more pose
checkpoints than DeepLabCut ships support for. As of dlclibrary 0.0.12 the registry lists
only `hrnet_w32`, `resnet_50`, `rtmpose_s` for `superanimal_quadruped`. `hrnet_w48`,
`rtmpose_m` and `rtmpose_x` exist in the repo (uploaded 2025-06-30) but are **not
registered**, so `hrnet_w32` is what a current install resolves.

If "most recent iteration" meant *newest file in the repo*, that points at `hrnet_w48`
and needs a manual download — `dlclibrary` will not resolve it. Worth confirming.

Detector: `fasterrcnn_mobilenet_v3_large_fpn`, benchmarked against
`fasterrcnn_resnet50_fpn_v2` on 12 DogFLW images — **0.31 s/img and 12/12** against
**3.64 s/img and 9/12**. 12 images, so indicative rather than settled.

---

## 4. The face-box rule, and how it was set

`FaceBoxConfig` in `src/faceconfig.py`. Both values were measured over all 4,335 images,
not chosen. The table is the box side needed to contain every one of the 46 ground-truth
landmarks, as a multiple of the anchor hull's side:

| anchor_conf | p50 | p90 | p95 | pad 1.6x | pad 1.8x | pad 2.0x |
|---|---|---|---|---|---|---|
| 0.1 | 1.202 | 1.536 | 1.662 | 93.1% | **97.7%** | 98.9% |
| 0.3 | 1.223 | 1.609 | 1.800 | 89.6% | 95.0% | 97.0% |
| 0.5 | 1.270 | 1.814 | 2.221 | 82.0% | 89.7% | 92.6% |

Two conclusions, and the second is counterintuitive:

* **pad 1.8** contains all 46 landmarks in 97.7% of images while leaving the face at ~67%
  of the box. 2.0 buys 1.2 points of containment for a smaller face; 1.6 loses 4.6.
* **A LOW `anchor_conf` is better.** Raising it drops marginal ear tips, which *shrinks*
  the hull, which means MORE padding is needed — p95 goes 1.66 → 1.80 → 2.22. Filter
  anchors loosely and let the pad absorb the error.

Anchors are the 11 non-antler head keypoints. The 4 antler keypoints are excluded: on a
dog the pretrained model puts `antler_base` on top of `earbase`, adding noise without
adding extent.

Fallback when fewer than 3 anchors clear the threshold: the upper 55% of the dog box,
squared off. Measured rate on DogFLW at `anchor_conf=0.1`: **1 in 3,279 train images**. On
video, where a dog may turn away, it will be common — hence it degrades to a bad crop
rather than to no prediction. `--fallback skip` is available if a gap is preferable.

---

## 5. The two parity traps in the crop

Both were found by tests, and both would have been invisible in the rendered output.

**DeepLabCut rounds box corners to integers.** `top_down_crop` computes corners as
`int(round(cx ± w/2))`. For a fractional box that is not symmetric: a 33.75 px square at
(100.5, 60.25) becomes 84..117 by 43..77 — **33 x 34**, so the crop scale comes out
anisotropic (0.1289 vs 0.1328), a 3% stretch that grows as the box shrinks. A model
trained on subtly stretched crops learns a subtly stretched face. `crop.to_dlc_bbox`
snaps the box to the integer grid first, making DLC's rounding the identity.

**DeepLabCut truncates the bbox again in the dataloader** (`bboxes.astype(int)`, before
cropping). So `build_face_coco.py` writes the integers `to_dlc_bbox` produces, and
training and inference land on the same pixels by construction.

`src/crop.py` therefore **wraps** DLC's cropper rather than reimplementing it. An earlier
draft did its own exact-float `warpAffine` and agreed with training only to within half a
pixel — precisely the bug the module exists to prevent.

---

## 6. Landmark naming is inferred, not authoritative

DogFLW ships coordinates only; the landmark manual is behind a dead link shortener
(`rb.gy/r6srv9` → 403). The names in `src/keypoint_scheme.py` were derived from the data:
each face normalised into its own bbox and averaged over all 4,335 images gives a
near-perfectly bilaterally symmetric mean shape — 20 mirror pairs and 6 midline points.

**Indices are exact; names are inference.** Two cross-checks passed: the pretrained
model's `right_eye` matched the point independently labelled `eye_right_lower_lid`, and
`right_earend` matched `ear_right_tip` — which also confirms "right" = the animal's right
= image-left. Anyone publishing off this should get the real manual from the authors.

---

## 7. Splits, and what identity is actually available

DogFLW is built on **Stanford Dogs**. Ids look like `n02085620_11477`, where the prefix is
a WordNet synset — `n02085620` is Chihuahua. So:

- **breed is exactly inferable** (120 breeds)
- **individual dog is not**
- all 4,335 filenames are unique, so there is no image-level leakage
- the shipped split shares all 120 breeds between train and test, i.e. it is random by
  image, not by breed

The validation split is therefore **stratified by breed** (default), matching how test was
built, so thresholds tuned on val transfer. `--strategy breed_disjoint` exists and is a
*different, harder question*, not a better one — a breed-disjoint val would be
systematically harder than the test split being reported on.

Current partition: **train 3,281 / val 574 / test 480**, all 120 breeds in each.

Residual risk that cannot be fixed here: Stanford Dogs may hold several photos of the same
individual dog within a breed, so near-duplicates may straddle any split. That is equally
true of the shipped test split and therefore of the 0.0438 baseline, so it does not bias
the comparison — but absolute DogFLW numbers may be mildly optimistic.

---

## 8. Running it

### Windows setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install --no-deps deeplabcut==3.0.1
.venv\Scripts\python.exe -m pip install "dlclibrary>=0.0.12" matplotlib einops filterpy networkx `
    pydantic tqdm imageio-ffmpeg scikit-learn scikit-image statsmodels tables pycocotools `
    numba "albumentations<=1.4.3"
.venv\Scripts\python.exe -m pip install timm pytest
```

Python **3.11**, not 3.13. `timm` is explicit — a Windows venv inherits no system
packages and DeepLabCut's HRNet backbone needs it. `imgaug` is declared by DeepLabCut but
deliberately **not** installed; it is a TensorFlow-era dependency and `import deeplabcut`
succeeds without it. pip will warn about it plus numpy/matplotlib/pandas/filelock — all
expected under `--no-deps`.

**CPU-only.** PyTorch's ROCm builds are Linux-only, so the Radeon 840M is unusable and
`device="cpu"` throughout is correct, not a leftover.

### The pipeline

```powershell
.\run_pipeline.ps1              # tests -> splits -> COCO -> train -> eval on val
.\run_pipeline.ps1 -SkipTrain   # stop before the long step
```

Step by step:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
.venv\Scripts\python.exe src\splits.py
.venv\Scripts\python.exe src\build_face_coco.py
.venv\Scripts\python.exe src\train_face.py --run-name face1 --epochs 4 --eval-every 1
.venv\Scripts\python.exe src\evaluate_face.py --split val --snapshot face_project\face1\snapshot-004.pt
.venv\Scripts\python.exe src\run_video.py --video "Happy lab.mov" --snapshot face_project\face1\snapshot-004.pt
```

`src/extract_dogflw.py` and `src/build_dogboxes.py` have already run; their outputs are on
disk and `build_dogboxes.py` takes ~42 min, so do not re-run it casually.

### Footguns that cost real time here

- **`pkill -f train_face` kills your own shell** on Linux, because the pattern matches the
  invoking command line. Use
  `ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /train_face/ {print $1}' | xargs -r kill`.
- **DeepLabCut reports progress through `logging`.** Without
  `setup_file_logging(model_folder / "log.txt")` a run is completely silent and looks
  hung. `train_face.py` calls it.
- **`np.load` on an `.npz` decompresses on every `__getitem__`.** `z["poses"][k]` inside a
  4,335-iteration loop re-reads the whole array each time; a probe that should have taken
  a second ran for two minutes. Materialise the array once.
- **Train loss is not a stopping signal on this project.** On the old model it went flat
  while test NME was still falling. Use `--eval-every 1`.
- **Windows sleep.** A 3-epoch run that should have taken 43 minutes took 14 hours because
  the machine slept. `train_face.py` suppresses it for the duration.

---

## 9. Quarantined: the post-hoc corrections

Three fixes carried most of the old architecture's accuracy. All are **default OFF** and
must be re-measured before being switched on — each learns a correction to one specific
model's residuals, and this is a different model.

| fix | old effect | expectation now |
|---|---|---|
| `ear_correct.py` | ear NME 0.0882 → 0.0631 (−28.5%) | **likely still needed.** Ear-type multimodality is a property of the labels, not the crop: error moved 4% across a 3x crop range and 20% by ear type |
| `shape_refine.py` | head top 0.1240 → 0.0880 (−29.0%) | **may be obsolete.** Those points partly failed because they sat near the edge of a low-resolution crop |
| two-pass head crop | ~18% NME on reliable landmarks | **obsolete by construction.** The cascade *is* the head crop |

Sub-pixel decoding is different in kind and stays **on**: argmax-only decoding is a defect,
not a baseline. It is now an explicit tested function (`src/decode.py`), not the
import-time monkeypatch it used to be.

Refit and measure:

```powershell
.venv\Scripts\python.exe src\postfit.py --snapshot <ckpt> --fit-shape --fit-ear
.venv\Scripts\python.exe src\evaluate_face.py --split val --snapshot <ckpt> --ear-correct
```

---

## 10. Success criteria

Report on the same 479 held-out test images, against the frozen baseline
(`stable-pre-model-separation`, CLAUDE.md §12 on that branch):

| metric | baseline (76-ch unified) |
|---|---|
| NME mean | 0.0438 |
| NME median | 0.0319 |
| PCK@5% | 71.0% |
| PCK@10% | 92.5% |
| eye | 0.0261 |
| nose | 0.0269 |
| mouth | 0.0325 |
| muzzle | 0.0505 |
| ear | 0.0631 |
| head top | 0.0880 |

`evaluate_face.py` prints this comparison automatically, plus the face-box failure rate
and NME broken out by face-box size.

**Stated in advance, so this stays an experiment:** expect substantial gains on
eye/nose/mouth (precision-limited by quantisation), modest on muzzle, and **little or none
on ear** — ear-type multimodality is orthogonal to resolution. If ear error improves
dramatically, be suspicious and check for split leakage first.

Also worth watching: the old model's per-image NME *rose* with face size (r = +0.310),
because face size was confounded with real camera resolution. If the cascade is doing its
job that correlation should flatten, since every face now arrives at the same scale.

---

## 11. Still not done

| item | state |
|---|---|
| **Train the face model** | not run. Everything upstream is built and tested |
| Sweep `pad` on val | 1.8 is measured-for-containment, not measured-for-accuracy |
| Sweep `pos_dist_thresh` | defaults to 8. On the old whole-dog crop 8 was worse than 17, but the landmarks are ~1.8x further apart in grid units now, so that result does not transfer |
| Re-measure the quarantined fixes | §9 |
| Qualitative figures | `make_figures.py` was deleted with the architecture it documented |
| The DogFLW landmark manual | still behind a dead link; §6 naming remains inferred |

---

## 12. Licences

DogFLW is **CC BY-NC 4.0 — non-commercial**, and that restriction propagates to any model
trained on it. Mixkit clip 1476 is Mixkit Free License; `dog_walk_sf.ogv` is CC BY-SA 3.0.
SuperAnimal weights follow the DeepLabCut Model Zoo terms.
