# Running the Dog Keypoints app

Three steps from a fresh clone to a working app on Windows. No terminal needed for
step 1 or 3.

---

## 1. Build the environment — double-click `setup.bat`

Takes 5–15 minutes and downloads about 1 GB (PyTorch is most of it). Safe to re-run if it
fails partway — it keeps whatever already installed.

It needs **Python 3.11** on the machine first. Not 3.12 or 3.13: DeepLabCut 3.0.1's
dependency wheels are not all available for those. Get it from
[python.org](https://www.python.org/downloads/release/python-3119/) — pick *Windows
installer (64-bit)* and tick **Add python.exe to PATH**.

`setup.bat` tells you if Python 3.11 is missing rather than failing obscurely.

## 2. Get a face model

**There is no trained face-model checkpoint in this repository yet.** The architecture was
rebuilt as a two-stage cascade and training has not been run. Two options:

**a) Train one** (about a day on a 6-core CPU):

```powershell
.venv\Scripts\python.exe src\splits.py
.venv\Scripts\python.exe src\build_face_coco.py
.venv\Scripts\python.exe src\train_face.py --run-name face1 --epochs 4 --eval-every 1
```

This needs the DogFLW imagery — see the bottom of this file. Snapshots land in
`face_project\face1\` and the app finds them automatically.

**b) Download one**, if a release has been published since:

> **https://github.com/gabe-udel/dog-emotions/releases**

Put the `.pt` **and its `pytorch_config.yaml`** into `model_weights\`. The app only lists
a checkpoint when a matching config sits beside it, because the architecture has to match
the weights or loading fails obscurely.

The **body** model needs nothing — stage 1 is the stock SuperAnimal-Quadruped checkpoint
and downloads itself on first run.

## 3. Run it — double-click `Dog Keypoints App.bat`

Choose a video, press Run. Output lands in `outputs\`.

If something is missing the app says which piece and where to get it, rather than failing
silently.

---

## What the app does per frame

```
your video
   ↓  detector (stock SuperAnimal, downloads itself on first run)
   ↓  SuperAnimal-Quadruped pose, unmodified   -> 39 body keypoints
   ↓  derive_face_box() from the head anchors  -> face box
   ↓  crop + resize to 256x256
   ↓  DogFLW face model                        -> 46 landmarks
   ↓  sub-pixel heatmap decoding    src/decode.py
   ↓  temporal median
labelled MP4
```

Roughly **2 seconds per frame** on a 6-core CPU — two HRNet passes, body then face.

## Settings worth knowing

| control | default | note |
|---|---|---|
| Display | Points + colour legend | contour lines exaggerate a single bad landmark into a visible spike |
| Confidence | 0.35 | a **visibility** control, not a quality knob. On the previous model confidence ran *opposite* to accuracy; that has not been re-checked here |
| Ear-type bias correction | **off** | fitted against the previous architecture. Re-measure before trusting it |
| Shape model for skull top | **off** | same. May be obsolete now the face crop is larger |

`outputs\keypoint_legend.png` maps every colour and index to a landmark name.

---

## Only if you are retraining or evaluating

Running the app needs none of this.

**DogFLW imagery** is not in the repo — it is CC BY-NC 4.0 and not ours to redistribute.
Download `georgemartvel/dogflw` from Kaggle and run `src\extract_dogflw.py`, which writes
`data\dogflw\`.

**`data\sa_dogboxes.npz` is** in the repo. It holds the SuperAnimal detector boxes and
39-keypoint predictions over all 4,335 images, which takes 42 minutes to recompute, and
the face boxes are derived from it.

**`face_project\`** (the generated COCO dataset and training snapshots) is not in the repo;
it regenerates from `src\build_face_coco.py`.

Full detail is in `CLAUDE.md`.

## Licence

DogFLW is **CC BY-NC 4.0 — non-commercial**, and that restriction propagates to any model
trained on it.
