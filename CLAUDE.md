# CLAUDE.md — handoff notes

Project state as of **2026-10-11**.

The pipeline is a **two-stage cascade**: stock SuperAnimal-Quadruped for the body, a
dedicated DogFLW model for the face. On top of it sits a **feature layer** that turns
keypoints into geometry, a **classification pilot** that is blocked on labelled data,
and **Pupchat** — a prototype that hands the measurements plus the video to a language
model and gets an owner-facing answer back (§15).

| layer | state |
|---|---|
| Body keypoints (39) | stock SuperAnimal, **measured**: NME 0.0243 on 2,545 images |
| Face landmarks (38) | `face38`, trained. **Validated only — never scored on test** |
| Features (20 live) | implemented, tested, error measured against ground truth |
| Classification | pilot only, 24 labelled rows, no trained model exists |
| Pupchat prototype | end to end and tested; the measurement half verified on real video |

Model in use: `face_project/face38/snapshot-best-008.pt`.
Its predecessor `face1` (46 landmarks) is backed up under `backups/face1-46kp-2026-09-03/`.
The pre-cascade architecture is frozen on branch **`stable-pre-model-separation`**.

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
  -> DogFLW face model (face38)                 -> 38 landmarks
  -> CropTransform.to_image                     -> image coordinates
  -> features.frame_features()                  -> per-frame geometry
  -> aggregate.aggregate()                      -> one row per clip
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

### Why the cascade replaced the unified model

The old model shared **99.99% of its parameters** between body and face: one HRNet-W32
backbone (29,363,185 params) into a single 1x1 `ConvTranspose2d` head (1,518 params at 46
channels), so every keypoint was a 32-number linear readout of the same feature vector.

| finding | number |
|---|---|
| within-image crop sweep, 30 reliable landmarks, at 25% / 45% / 55% face fill | NME 0.0411 / 0.0362 / 0.0328 — **monotonic** |
| face span on the 64x64 heatmap grid, DogFLW | median 35.9 cells (p5 21.4, p95 60.4) |
| face span on video (whole standing dog) | ~16 cells |
| argmax decoding collapsed 46 face channels onto | **16 distinct pixels** |

What the rebuild bought:

| | old (whole-dog crop) | new (derived face box) |
|---|---|---|
| face span, DogFLW test | median 35.9 cells | **median 39.7** (p25 36.8, p75 44.2) |
| face span, video | ~16 cells | ~40, independent of dog distance |
| spread across the split | 21 → 60 cells | 37 → 44 cells |
| landmark supervision | 60.5% GT / 22.9% pseudo / 16.6% masked | **99.9% ground truth** |

On DogFLW the resolution gain is only **+11%**, because `pad=1.8` spends most of the crop
guaranteeing a flopped ear stays inside it. The gains that are large are the **variance
collapse** and the **supervision jump**. On video the change is ~2.5x.

---

## 3. Which SuperAnimal checkpoint, and why

`mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped` on HuggingFace holds more pose
checkpoints than DeepLabCut ships support for. As of dlclibrary 0.0.12 the registry lists
only `hrnet_w32`, `resnet_50`, `rtmpose_s` for `superanimal_quadruped`. `hrnet_w48`,
`rtmpose_m` and `rtmpose_x` exist in the repo but are **not registered**, so `hrnet_w32` is
what a current install resolves.

Detector: `fasterrcnn_mobilenet_v3_large_fpn`, benchmarked against
`fasterrcnn_resnet50_fpn_v2` on 12 DogFLW images — **0.31 s/img and 12/12** against
**3.64 s/img and 9/12**. 12 images, so indicative rather than settled.

---

## 4. The face-box rule, and how it was set

`FaceBoxConfig` in `src/faceconfig.py`. Both values were measured over all 4,335 images.
The table is the box side needed to contain every one of the 46 ground-truth landmarks, as
a multiple of the anchor hull's side:

| anchor_conf | p50 | p90 | p95 | pad 1.6x | pad 1.8x | pad 2.0x |
|---|---|---|---|---|---|---|
| 0.1 | 1.202 | 1.536 | 1.662 | 93.1% | **97.7%** | 98.9% |
| 0.3 | 1.223 | 1.609 | 1.800 | 89.6% | 95.0% | 97.0% |
| 0.5 | 1.270 | 1.814 | 2.221 | 82.0% | 89.7% | 92.6% |

Two conclusions, and the second is counterintuitive:

* **pad 1.8** contains all 46 landmarks in 97.7% of images while leaving the face at ~67%
  of the box.
* **A LOW `anchor_conf` is better.** Raising it drops marginal ear tips, which *shrinks*
  the hull, which means MORE padding is needed — p95 goes 1.66 → 1.80 → 2.22. Filter
  anchors loosely and let the pad absorb the error.

Anchors are the 11 non-antler head keypoints. The 4 antler keypoints are excluded: on a
dog the pretrained model puts `antler_base` on top of `earbase`.

Fallback when fewer than 3 anchors clear the threshold: the upper 55% of the dog box,
squared off. Measured rate on DogFLW: **1 in 3,279 train images**. `--fallback skip` is
available if a gap is preferable to a bad crop.

---

## 5. The two parity traps in the crop

Both were found by tests, and both would have been invisible in the rendered output.

**DeepLabCut rounds box corners to integers.** A 33.75 px square at (100.5, 60.25) becomes
84..117 by 43..77 — **33 x 34**, so the crop scale comes out anisotropic (0.1289 vs
0.1328), a 3% stretch that grows as the box shrinks. A model trained on subtly stretched
crops learns a subtly stretched face. `crop.to_dlc_bbox` snaps the box to the integer grid
first, making DLC's rounding the identity.

**DeepLabCut truncates the bbox again in the dataloader** (`bboxes.astype(int)`). So
`build_face_coco.py` writes the integers `to_dlc_bbox` produces, and training and inference
land on the same pixels by construction.

`src/crop.py` therefore **wraps** DLC's cropper rather than reimplementing it.

---

## 6. Landmark naming is inferred, not authoritative

DogFLW ships coordinates only; the landmark manual is behind a dead link shortener
(`rb.gy/r6srv9` → 403). The names in `src/keypoint_scheme.py` were derived from the data:
each face normalised into its own bbox and averaged over all 4,335 images gives a
near-perfectly bilaterally symmetric mean shape — 20 mirror pairs and 6 midline points.

**Indices are exact; names are inference.** Two cross-checks passed: the pretrained model's
`right_eye` matched the point independently labelled `eye_right_lower_lid`, and
`right_earend` matched `ear_right_tip` — which also confirms "right" = the animal's right =
image-left.

---

## 7. Splits, and what identity is actually available

DogFLW is built on **Stanford Dogs**. Ids look like `n02085620_11477`, where the prefix is
a WordNet synset. So **breed is exactly inferable** (120 breeds), **individual dog is not**,
and the shipped split is random by image, not by breed.

The validation split is therefore **stratified by breed** (default), matching how test was
built. `--strategy breed_disjoint` exists and is a *different, harder question*, not a
better one.

Current partition: **train 3,279 / val 574 / test 480**, all 120 breeds in each.

Residual risk: Stanford Dogs may hold several photos of the same individual within a breed,
so near-duplicates may straddle any split. That is equally true of the published baseline,
so it does not bias the comparison — but absolute DogFLW numbers may be mildly optimistic.

---

## 8. The landmark set: 46 → 38

Ears stayed the worst region and the cascade barely moved them (−5.5%). Decomposing ear
error against the local tangent of the ear contour showed why: error ran **1.3–1.9x larger
ALONG** the ear edge than across it. The model finds the edge and cannot know where along
it a human placed point five. That is annotation indeterminacy, not resolution, and no
amount of training fixes it.

`keypoint_scheme.DROP_EAR_CONTOUR` removes the eight mid-contour samples, keeping
`inner_base`, `tip`, `outer_base` per ear — the points that carry ear orientation.

```python
DROP_EAR_CONTOUR = [2, 3, 4, 5, 8, 9, 10, 11]
ACTIVE = [i for i in range(46) if i not in DROP_EAR_CONTOUR]   # 38
CH_OF  = {d: k for k, d in enumerate(ACTIVE)}
```

**The indexing hazard this introduced.** While all 46 were active, a DogFLW index and a
model channel were the same number. They are not any more, and a stale `kpts[14]` now
*silently reads a different landmark* instead of failing. `ACTIVE` / `CH_OF` are the single
source of truth; every consumer was converted to channel space and
`tests/test_landmark_set.py` pins the mapping.

Three guards exist because this failure is silent:

- `cascade.checkpoint_n_keypoints` reads `num_heatmaps` from the **checkpoint's own**
  config and refuses a mismatch. Decoding a 46-channel model as 38 gives the wrong heatmap
  stride and hands every landmark a neighbour's identity — and still renders a plausible face.
- `shape_refine.Refiner` and `ear_correct.EarCorrector` both verify the landmark set of the
  model they load. The ear fit's 46-space indices `[0..13]` are all *valid* on a 38-channel
  array; without the guard it would subtract an ear bias from the eyes.

---

## 9. Results

### Face — `face1`, 46 landmarks, TEST split, scored once

479 held-out images, sub-pixel decoding, both post-hoc corrections on.
**This is the only test-split number that exists for any face model.**

| region | n | cascade | old unified | change | PCK@5% |
|---|---|---|---|---|---|
| eye | 8 | **0.0131** | 0.0261 | −49.6% | 98.7% |
| nose | 7 | **0.0135** | 0.0269 | −49.8% | 99.1% |
| muzzle | 4 | **0.0218** | 0.0505 | −56.9% | 97.2% |
| mouth | 11 | **0.0240** | 0.0325 | −26.1% | 91.1% |
| head top | 2 | **0.0262** | 0.0880 | **−70.2%** | 94.3% |
| ear | 14 | **0.0596** | 0.0631 | −5.5% | 57.6% |
| **all 46** | 46 | **0.0313** | 0.0438 | **−28.6%** | **84.1%** |

**The scale confound is gone.** The old model's per-image NME *rose* with face size
(r = +0.310) because face size was confounded with camera resolution. Now **r = −0.078**,
essentially flat across 31–2233 px.

### Face — `face38`, 38 landmarks, VALIDATION only

8 epochs. Note the wall-clock: 7.5 hours for ~2.4 hours of compute, the rest lost to
Windows Modern Standby (see §12).

NME over 38 landmarks is **not** comparable to NME over 46 — dropping the hardest points
lowers the mean without improving a single prediction. `evaluate_face.py` prints `n/a` on
those rows rather than a flattering number. Scored on the **same 38 landmarks**, same val
split, corrections off both sides:

| region (shared points) | face1 | face38 | change |
|---|---|---|---|
| eye (8) | 0.0123 | 0.0112 | −9.2% |
| nose (7) | 0.0125 | 0.0113 | −9.8% |
| muzzle (4) | 0.0200 | 0.0178 | −11.2% |
| mouth (11) | 0.0234 | 0.0240 | +2.6% |
| head top (2) | 0.0443 | 0.0269 | −39.2% |
| **all 38 shared** | **0.0250** | **0.0223** | **−10.7%** |

The gain is **not** in the ears. Five of the six surviving ear points moved by under 2.5%,
which is noise. The one large mover (`ear_left_outer_base`, −46.3%) has a mirror that did
not move at all, so it is more plausibly a 574-image fluke than an effect. The real gain
landed on eye, nose, muzzle and head top — landmarks nowhere near the ears, which is the
signature of indeterminate supervision having polluted the shared backbone.

`head_top` at 0.0269 with **no correction** now beats face1's 0.0277 *with* the shape model,
so `shape_refine` may finally be obsolete. Untested — see §11.

### Body — stock SuperAnimal vs StanfordExtra v12

2,545 overlapping images (both datasets are built on Stanford Dogs, so ids join directly),
38,656 scored keypoints, normalised by the dog-box diagonal.

| | value |
|---|---|
| NME | **0.0243** |
| median | 0.0171 |
| PCK@5% | 92.2% |
| PCK@10% | 97.9% |

Laterality was verified from the data, not assumed: median error **0.0165** direct against
**0.1667** with the sides swapped, a 10x gap.

**14 of the 39 keypoints have no accuracy figure at all** — StanfordExtra never annotates
`upper_jaw, mouth_end_right, mouth_end_left, right_eye, left_eye, neck_base, neck_end,
throat_base, throat_end, back_base, back_middle, belly_bottom, body_middle_right,
body_middle_left`. Two of those matter: `back_base` (withers) and `back_middle` define `L`,
the body length **every body feature divides by**.

**Face NME and body NME are not comparable to each other** — different normalisers.

---

## 10. Features

`src/features.py` computes per-frame geometry; `src/aggregate.py` turns a clip into one
row, smoothing in milliseconds (converted with the clip's own fps), differentiating per
second, and handling angles circularly throughout — `head_tilt`'s apparent 145° spread was
1.23° once wrap was handled.

Two rules exist to prevent specific silent failures:

- **Look up keypoints by NAME, never index.** Runtime channel order differs from the test
  fixtures: `back_base` is 19, `back_end` 20, `back_middle` 21, and the rear legs interleave.
- **A feature is NaN unless every keypoint it needs clears the confidence cut — never 0.**
  Zero is a real value: 0 tuck means the tail is against the hips, i.e. maximum fear.

### How feature accuracy is measured

A feature's error cannot be inferred from its keypoints. Features are mostly *differences*
between nearby landmarks, and pose error is largely **common-mode** — when the face box
lands slightly off, every landmark shifts together and the shared part cancels. An earlier
Monte-Carlo approach perturbed landmarks independently, destroying that cancellation, and
declared eye aperture and the lip measures dead *by assumption*.

The current method needs no simulation. DogFLW ships ground truth, so each feature is
computed twice on the same image and the difference **is** the error:

```
d        = feature(prediction) − feature(truth)     # angles wrapped to +/-180
error_sd = sd(d)        bias = mean(d)
true_sd  = sd(feature(truth))  across 574 val images

SNR = max(0, signal_var − error_var) / error_var    # a VARIANCE ratio
  snr_across_dogs : signal = true_sd      -> can it separate one DOG from another?
  snr_within_clip : signal = observed_sd  -> can it resolve MOTION inside one clip?
```

SNR `0.00` is not rounding — it means the observed variation was entirely explained by
measurement error. Take a square root for an amplitude ratio.

**Body features have no ground truth** (no withers), so they keep the Monte-Carlo estimate,
which is an upper bound on noise and therefore a lower bound on their SNR.

### The 20 live features

Ground truth, 574 val images. Coverage is across 4 video clips / 480 frames.

| face feature | err_sd | across dogs | within clip | cov |
|---|---|---|---|---|
| `ear_height` | 0.036 | 49.87 | 0.33 | 71% |
| `ear_angle_L_basevec` | 13.01° | 27.08 | 0.00 | 91% |
| `ear_angle_R_basevec` | 15.63° | 23.27 | 0.00 | 66% |
| `view_yaw` | 0.010 | 20.38 | 1.61 | 92% |
| `ear_angle_R_eyeframe` | 11.44° | 18.45 | 0.00 | 71% |
| `ear_angle_L_eyeframe` | 12.93° | 13.60 | 0.00 | 91% |
| `view_dir` | 0.043 | 11.94 | 0.00 | 92% |
| `ear_len_R` | 0.039 | 8.98 | 0.00 | 66% |
| `ear_len_L` | 0.040 | 7.93 | 0.00 | 96% |
| `ear_closeness_iod` | 0.239 | 4.55 | 0.00 | 71% |
| `mouth_open_jaw` | 0.020 | 3.76 | 0.00 | 74% |

| body feature (Monte-Carlo only) | obs_sd | noise_sd | within clip | cov |
|---|---|---|---|---|
| `head_pitch` | 21.32° | 5.00° | 17.15 | 68% |
| `tail_height` | 40.25° | 13.80° | 7.51 | 73% |
| `head_tilt` | 18.64° | 7.04° | 6.01 | 64% |
| `head_height` | 0.248 | 0.147 | 1.86 | 67% |
| `tail_tuck` | 0.090 | 0.055 | 1.70 | 46% |
| `tail_lateral` | 0.072 | 0.097 | **0.00** | 68% |
| `tail_len` | 0.085 | 0.091 | **0.00** | 68% |
| `posture` | 0.076 | 0.137 | **0.00** | 44% |
| `motion_energy` | — | — | never validated | 99% |

Bookkeeping columns, reported as coverage and never summarised as signal:
`body_L, face_px, head_tilt_used_ground, n_paws, tail_conf`.

### What this feature set is, and is not

**It is a between-dog, clip-level instrument.** Almost every face feature reads 0.00
within-clip — the variation inside a clip is fully explained by measurement error. Only six
features resolve motion: `head_pitch`, `tail_height`, `head_tilt`, `head_height`,
`tail_tuck`, `view_yaw`. Of the clip-level features, **eight of nine affect-bearing ones are
ears**. Any classifier built on this is largely an ear-posture classifier; say so.

`tail_lateral` is the wag source and sits below its own noise floor, which is why
`wag_rate_hz` returns nothing on real footage. That gate **fails closed** by design:
counting zero-crossings of noise always yields a confident, meaningless frequency, and an
earlier version without the floor reported a 9.97 Hz "wag" against a real range of 3–5 Hz.

### Six features measured, then deleted

| removed | across dogs | note |
|---|---|---|
| `mouth_open_lips` | 2.12 | the jaw form beat it on error, SNR and coverage |
| `ear_closeness_basesep` | 1.02 | the iod form won |
| `lip_raise_philtrum` | 0.69 | below its own error |
| `eye_width` | 0.05 | dead; bias −0.0096 against true_sd 0.0125 |
| `eyelid_open` | 0.03 | dead; true_sd 0.0116 against err_sd 0.0114 |
| `lip_raise_upperlip` | 0.00 | dead; bias −0.0088 against true_sd 0.0127 |

Two capabilities die with them: **no usable eye-aperture signal and no usable snarl
signal** (a snarl holds the mouth nearly shut, so `mouth_open_jaw` cannot see it either).
`eyelid_open` rides the two most accurate landmarks on the whole face (0.0096–0.0107) and
still fails, because the lid gap is a few pixels. Reinstating either needs better
landmarks, not better arithmetic.

---

## 11. Classification pilot

`src/pilot_extract.py` (still images → feature rows) and `src/pilot_model.py` (k-means with
labels withheld from the fit, plus leave-one-dog-out logistic regression, plus a
permutation baseline).

Smoke run: **24 rows, 5 dogs, 2 classes.** Adjusted Rand −0.025, supervised accuracy 0.375
against chance 0.500, permutation p = 0.78.

**Those numbers mean nothing yet.** With 24 rows and 11 features, p = 0.78 is what *absent
data* looks like, not an absent effect. The run also dropped 8 of 19 features for near-zero
coverage — every tail, posture and head-height feature — because the source is cropped face
images with no body in frame.

No trained classifier exists on disk. The pilot prints a result and saves no model.

---

## 12. Running it

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

Python **3.11**, not 3.13. `timm` is explicit — DeepLabCut's HRNet backbone needs it and a
Windows venv inherits no system packages. `imgaug` is declared by DeepLabCut but
deliberately **not** installed. **CPU-only**: PyTorch's ROCm builds are Linux-only, so the
Radeon 840M is unusable and `device="cpu"` throughout is correct, not a leftover.
pandas is pinned to 2.2.3 — Smart App Control blocks the pandas 3.0.5 DLL.

### The app

Double-click **`Dog Keypoints App.bat`**, or:

```powershell
.venv\Scripts\python.exe src\video_app.py
```

It lists every checkpoint with a config beside it, labels each with its landmark count,
marks incompatible ones, and preselects one that can actually run.

### Command line

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q                       # 235 tests

# video -> labelled MP4 + per-frame features + clip row
.venv\Scripts\python.exe src\run_video.py --video "Happy lab.mov" `
    --snapshot face_project\face38\snapshot-best-008.pt --ransac --features

# same, with the live feature readout drawn on the video
.venv\Scripts\python.exe src\run_video.py --video "Happy lab.mov" `
    --snapshot face_project\face38\snapshot-best-008.pt --ransac --feature-overlay

.venv\Scripts\python.exe src\evaluate_face.py --split val --snapshot <ckpt>
.venv\Scripts\python.exe src\check_body_accuracy.py
.venv\Scripts\python.exe src\check_feature_error.py --snapshot <ckpt>
```

`--config` is no longer passed: it is derived from the directory the snapshot lives in.
`--feature-pcut` is separate from `--pcut` on purpose — `--pcut` is a *display* control, and
raising it to tidy up a video must not silently change a measurement.

**Where features are computed matters.** `run_video.py` taps the keypoints **after RANSAC
outlier rejection and before the rendering median**. `aggregate.oscillation` reads the raw
series deliberately, because smoothing is a low-pass filter and a wag is exactly the
high-frequency content it removes; `aggregate()` then smooths again at its own fps-aware
window. Tapping after the median low-passes twice and silently flattens every rate and
oscillation feature.

`src/extract_dogflw.py` and `src/build_dogboxes.py` have already run; `build_dogboxes.py`
takes ~42 min, so do not re-run it casually.

### Footguns that cost real time here

- **Windows Modern Standby defeats the sleep suppression.** `train_face.py::keep_awake`
  calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, which prevents legacy
  S3 sleep but **not** Modern Standby. It prints "sleep suppressed" and then does not. The
  face38 run lost 115 and 221 minutes to two standby periods on battery (DC timeout 180 s;
  AC is set to never). **Train on AC power.**
- **DeepLabCut reports progress through `logging`.** Without
  `setup_file_logging(model_folder / "log.txt")` a run is completely silent and looks hung.
- **`np.load` on an `.npz` decompresses on every `__getitem__`.** `z["poses"][k]` inside a
  4,335-iteration loop re-reads the whole array each time. Materialise it once.
- **Train loss is not a stopping signal on this project.** Use `--eval-every 1`.
- **git is not on PATH.** Prepend `C:\Program Files\Git\cmd`.
- **Pushing needs an interactive sign-in.** Git Credential Manager opens a browser, which a
  non-interactive shell cannot provide. Double-click `push_to_github.bat`.
- **`pkill -f train_face` kills your own shell** on Linux, because the pattern matches the
  invoking command line.

---

## 13. Open items

| item | state |
|---|---|
| **face38 never scored on test** | Validation only. The quotable 0.0313 belongs to `face1`. Val informed decisions, so face38's 0.0223 is optimistic by construction. One run closes this. |
| **Post-hoc corrections never refitted for 38 channels** | `data/shape_model.npz` and `data/ear_bias.pkl` are still the 46-channel fits, so face38 runs with both OFF. On face1 they were worth 15% of NME. `shape_refine` may now be genuinely obsolete (§9) — measure before refitting. `--fit-shape` is seconds; `--fit-ear` is ~65 min of inference over 3,279 images. |
| **No labelled emotion dataset** | 24 rows. This blocks the entire classification half. `src/fetch_emotion_videos.py` exists; its output does not. |
| **Withers unlabelled** | ~60 clicks, `src/label_withers.py` written and never run. Unblocks per-FEATURE body accuracy and the whole body half of the SNR table. |
| **Right-ear coverage is systematically lower than left** | 66–71% against 91–96% across 4 clips. Probably viewpoint, but any left/right asymmetry feature would inherit the bias. Check on footage with varied head direction. |
| Sweep `pad` | 1.8 was measured for containment, never for accuracy. 1.4 would give ~51 heatmap cells instead of ~40 but contains all landmarks in only 78.6% of images. |
| Sweep `pos_dist_thresh` | 8 was reasoned, never tested against 17 or 4 on this architecture. |
| More epochs | the curve had not flattened at 8. Resume from `snapshot-best-008.pt`. |
| The DogFLW landmark manual | still behind a dead link; §6 naming remains inferred. |

---

## 14. Licences

Full text in [ATTRIBUTION.md](ATTRIBUTION.md), which is also what `/api/health` serves
and what the web page renders.

DogFLW is **CC BY-NC 4.0 — non-commercial**, and that restriction propagates to any model
trained on it. StanfordExtra v12 carries its own terms and is **not** committed to this
repo. Mixkit clip 1476 is Mixkit Free License; `dog_walk_sf.ogv` is CC BY-SA 3.0.
SuperAnimal weights are a **Modified MIT**: academic and non-commercial use, copyright
notice must travel with copies, commercial licensing via EPFL-TTO.

**A correction worth knowing.** `LICENCE_NOTICE` used to say this "must not be deployed
publicly", which was stricter than either licence actually is. Both restrict
**commercial** use, not public hosting: CC BY-NC explicitly permits sharing for
non-commercial purposes *with attribution*, and the Modified MIT permits non-commercial
use with the notice carried along. So a free demo is allowed and the credit is a
**condition** of it. Selling it, running ads on it, or using it to promote a business
is not. The notice now says that, and the attribution travels with the API so a front
end cannot leave it behind.

---

## 15. Pupchat: the prototype on top

`video + question -> keypoints -> features -> evidence -> provider -> answer`.
Full documentation in [docs/PUPCHAT_PROTOTYPE.md](docs/PUPCHAT_PROTOTYPE.md).

A layer, not a change: it imports `run_video.clip_features` rather than recomputing
anything, so Pupchat's numbers and the research pipeline's numbers come from one
function. No model, training or evaluation code was touched.

```powershell
.venv\Scripts\python.exe -m pupchat ask --video "<file>" --question "<text>"
.venv\Scripts\python.exe -m pupchat evidence --video "<file>"     # no key needed
.venv\Scripts\python.exe -m pupchat.web                            # gradio page
.venv\Scripts\python.exe -m pupchat.api                            # the Stitch web app
```

Or double-click: **`Pupchat Web App.bat`** for the local page, **`Share Pupchat.bat`**
to put it on the internet behind a Cloudflare tunnel (see below).

### Two providers, one prompt

The only real difference is what media each can accept. OpenAI has **no video input** —
checked against [the vision guide](https://developers.openai.com/api/docs/guides/images-vision),
which documents exactly `input_text` and `input_image` and no `video` type — so frames
are sampled client-side at **2 fps under 15 s**, 1 fps above, capped at 32. Gemini takes
the file whole ([docs](https://ai.google.dev/gemini-api/docs/video-understanding)):
inline under 100 MB, Files API above.

The system prompt and the evidence text are byte-identical across providers, and a test
asserts it. Otherwise a comparison between them measures two prompts rather than two
models. `PUPCHAT_PROVIDER` in `.env` selects; `--provider` overrides for one run.

### The evidence summary, and what it refuses to say

Two tiers, derived from the measurement files rather than hard-coded:

* **Timestamped moments** only where within-clip SNR >= 1 — `head_pitch`, `tail_height`,
  `head_tilt`, `head_height`, `tail_tuck`, `view_yaw`. An excursion registers when the
  series leaves its clip baseline by more than 2x the feature's **own measured error**.
* **Clip-level descriptions, no timing**, for everything that separates dogs but not
  moments.

Where `feature_error.json` and `feature_snr.json` disagree, ground truth wins — the same
rule `aggregate.load_noise_floors` uses. That promotes `view_yaw` (1.61 vs 0.94) and
demotes `ear_height` (0.33 vs 1.42), and reproduces the six above without naming them.

Three refusals are load-bearing:

* **No emotion words.** `config.BANNED_WORDS` covers affect and also interpretation
  wearing body-language clothes — `relaxed`, `alert`, `submissive`. The model does the
  interpreting; handing it a conclusion dressed as a measurement makes the loop circular.
* **Unavailable features are always listed with a reason, and never trimmed.** The first
  version dropped bullets off the end when over budget, which on a real clip silently
  removed 8 of 10 unavailable entries — including the wag rate. Long lists are grouped
  by reason now, and tests assert every entry survives a tight budget. A reader cannot
  tell a trimmed entry from one that never existed, which makes that the single most
  dangerous truncation order available.
* **Wag rate is reported as unavailable whenever the oscillation gate refuses**, which
  on real footage is almost always (§10). That is not evidence the dog did not wag, and
  the system prompt says so explicitly.

### Verified on real video

`videos/whie pom sample 15.mp4`, 7.5 s: 94 keypoint frames at 12.5 fps (strided from
25), 16 vision frames at 2 fps, 10 features reported unavailable with reasons, three
timestamped `view_yaw` moments. Both providers' requests inspected via `--dry-run`:
question last in each, no key in either payload.

**Gemini verified live (2026-10-11)**: `gemini-3.8-flash`, 1,874 in / 896 out, 13.1 s,
$0.0048, plus a follow-up at 2,409 in with no re-measurement. The answer recommended a
behaviourist and a vet unprompted, quoted the ear angles, and stated that the body and
tail were not visible. **OpenAI is still unverified live** — no key available; its wire
format is covered by tests and `--dry-run` only.

### Reasoning tokens are billed, and the providers disagree about reporting

The first live call exposed a money bug. Gemini reports `candidates_token_count` and
`thoughts_token_count` separately and **charges for both** — measured 214 visible
against 682 thinking, so billing the visible count alone understated the output charge
roughly fourfold. OpenAI already folds reasoning into `output_tokens`, where adding it
again would overstate. `normalise_usage` makes `output_tokens` the billed total on both.

Same cause truncated the first answer mid-word: at `max_output_tokens=900`, thinking ate
682 before the answer started. Default is now 3000, and truncation is surfaced rather
than returned as though it were a complete answer.

### Seeing a feature's working

`pupchat trace --video <file> [--time N] [--failed-only]` joins keypoint confidences to
feature values for one frame, from the cache, recomputing nothing. A NaN feature stops
being a dead end: the trace names the keypoint that fell below the cut. It also prints
clip-wide keypoint reliability, weakest first, which is what a coverage figure means -
a feature is NaN unless EVERY keypoint it needs clears the cut.

`pipeline.py` caches `keypoints.npz` for this (CACHE_VERSION 4); older caches need
`--force`.

### Sharing it: what changes when it is not just you

`Share Pupchat.bat` starts the server, opens a **Cloudflare quick tunnel** and prints a
random `trycloudflare.com` address. No account, no card, and the address dies with the
window. The tunnel was chosen over hosting on the measurement: a cold analysis of a
7.5 s clip is **93 s on 12 cores**, so on a 2-vCPU free tier it would be eight to ten
minutes per visitor. The host's own machine is the fastest cheap option available. (For
the record, that door is also shut: since July 2026 Hugging Face free accounts cannot
create Docker Spaces, and a CPU Gradio Space needs PRO.)

`pupchat/guard.py` holds the four things that only matter once someone else can reach
it. **Localhost is exempt from all of them**, so a local run behaves exactly as it did
before the module existed — which is also why they need tests rather than manual
checking.

| | why |
|---|---|
| **Spend caps**, per visitor and global | An ask is ~$0.0048 measured, so 300/day bounds the bill at about $1.44. The **global** cap is the load-bearing one: per-IP counting rests on a forwarding header the visitor controls, and a cap that can be rotated away is not a budget. |
| **Streaming size limit** | The 80 MB cap existed but ran after `await file.read()`, so a 10 GB POST was a 10 GB allocation before the 413. `read_capped` stops mid-stream. |
| **One analysis at a time, in a thread** | `analyse()` is 93 s of blocking CPU, and `/api/analyse` *and* `/api/ask` both called it from an `async def` — so one measurement stalled every other request, the page included, for its full duration. The waiting room is bounded at 6: being refused instantly beats waiting ten minutes behind people the page never mentioned. |
| **24 h retention** | Strangers' videos land on the host's disk. Swept only in share mode — deleting the researcher's own test clips on a timer would be a surprise, and locally there is no stranger whose video needs removing. Legacy uploads under the old temp path are never swept. |

Two honest limits. A visitor can forge `CF-Connecting-IP` and land in a fresh bucket
each request, which is why the global cap exists and why the per-IP cap is documented
as a courtesy rather than a control. And the per-IP header is *trusted* here, which
would be wrong behind a general proxy — behind a quick tunnel it is the only source of
the visitor's address.

`PUPCHAT_SHARE=1` gates retention and the visitor notice on the page. It does **not**
gate the caps or the queue, which are always on and simply never fire for 127.0.0.1.

### Footgun carried over

`pipeline.sample_frames` strides the video down to ~10 fps and hands the **strided** rate
to `aggregate()`. Passing the source rate instead would scale every per-second
derivative and the oscillation rate by the stride factor, silently. The vision pass has
its own, much sparser rate; conflating the two would either blow the token budget or
starve the temporal features.
