# Running the Dog Keypoints app

Three steps from a fresh clone to a working app on Windows. No terminal needed.

---

## 1. Build the environment — double-click `setup.bat`

Takes 5–15 minutes and downloads about 1 GB (PyTorch is most of it). Safe to re-run if
it fails partway — it keeps whatever already installed.

It needs **Python 3.11** on the machine first. Not 3.12 or 3.13: DeepLabCut 3.0.1's
dependency wheels are not all available for those. Get it from
[python.org](https://www.python.org/downloads/release/python-3119/) — pick *Windows
installer (64-bit)* and tick **Add python.exe to PATH**.

`setup.bat` tells you if Python 3.11 is missing rather than failing obscurely.

## 2. Get the trained model

**The checkpoint is not in this repository.** It is 113 MB and GitHub rejects any file
over 100 MB, so it ships as a release asset instead:

> **https://github.com/gabe-udel/dog-emotions/releases**

Download **`superanimal_quadruped_dogface_final.zip`** (104 MB), then **unzip it** and put
`superanimal_quadruped_dogface_final.pt` into the `model_weights/` folder, beside the
`.yaml` files already there.

It is zipped because GitHub Releases only accepts a fixed list of file extensions and
`.pt` is not among them. Nothing else about it is special — the archive holds one file.

`model_weights/` should then look like:

```
model_weights/
  pytorch_config.yaml                        (in the repo)
  superanimal_quadruped_dogface_sigma8.yaml  (in the repo)
  superanimal_quadruped_dogface_final.pt     (you just added this)
```

That is the only file you need. The other two checkpoints are a failed experiment
(`_sigma8`) and an untrained reference (`_hrnet_w32_dogface`), kept for comparison and
not worth downloading unless you are reproducing the evaluation.

## 3. Run it — double-click `Dog Keypoints App.bat`

Choose a video, press Run. Output lands in `outputs/`.

If something is missing the app says which piece and where to get it, rather than
failing silently.

---

## What the app does per frame

```
your video
   ↓  detector (stock SuperAnimal, downloads itself on first run)
   ↓  pose model — 76 heatmaps: 39 body + 37 added facial
   ↓  sub-pixel peak fitting          src/subpixel.py
   ↓  ear-type bias correction        src/ear_correct.py
   ↓  shape model for the skull top   src/shape_refine.py
   ↓  temporal smoothing
labelled MP4, 76 keypoints
```

The three correction stages are the difference between a usable result and a smear —
see `CLAUDE.md` §11 for what each fixes and the measurements behind it. They are on by
default and their fitted parameters (`data/shape_model.npz`, `data/ear_bias.pkl`) *are*
in the repository, so nothing needs refitting.

Roughly **1 second per frame** on a 6-core CPU. The default 150 frames is about 2½
minutes.

## Settings worth knowing

| control | default | note |
|---|---|---|
| Model | sigma 17 | the σ=8 model scores worse and hides its own weak points; keep this |
| Display | Points + colour legend | connecting lines exaggerate a single bad point into a spike across the face |
| Confidence | 0.35 | a **visibility** control, not a quality knob — on this model confidence runs *opposite* to accuracy |
| Zoom to head | off | ~18% better NME but drops visible face points from 44 to 25; use it for measuring, not looking |

`outputs/keypoint_legend.png` maps every colour and index to a landmark name.

---

## Only if you are retraining or evaluating

Running the app needs none of this.

**DogFLW imagery** is not in the repo — it is CC BY-NC 4.0 and not ours to redistribute.
Download `georgemartvel/dogflw` from Kaggle and run `src/extract_dogflw.py`, which takes
the zip and writes `data/dogflw/`.

**Derived training data** (`dlc_project/`, 909 MB) is not in the repo either; it
regenerates in seconds from `src/make_coco.py`.

**`data/sa_dogboxes.npz` is** in the repo — it holds the SuperAnimal detector boxes and
pose predictions over all 4,335 images, which takes 42 minutes to recompute. The
evaluation scripts need it.

Full detail, including why the stock DeepLabCut fine-tuning path cannot add keypoints,
is in `CLAUDE.md`.

## Licence

DogFLW is **CC BY-NC 4.0 — non-commercial**, and that restriction propagates to any model
trained on it, including the released checkpoint.
