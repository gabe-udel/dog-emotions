"""Measure what a typical dog's face looks like, once.

    .venv\\Scripts\\python.exe tools\\build_reference.py

Writes `outputs/pupchat/reference.json`: per feature, the 10th/25th/50th/75th/90th
percentiles across DogFLW's hand-annotated landmarks. The evidence layer uses them to say
"higher than most dogs" instead of handing the model a bare number it cannot judge.

**Ground truth, not predictions.** The landmarks are the human annotations, so the spread
is the spread of real dogs rather than the model's error on top of it. The machinery
already exists: src/check_feature_error.py computes features from ground truth to measure
error, and this is the same call over all 4,335 images rather than the 574 validation ones.

Two whole groups of features are deliberately absent, and both absences are the honest
answer rather than a gap to fill later.

**Body features.** Every one divides by body length, which runs withers to tail base, and
StanfordExtra leaves the withers slot empty in all 2,545 overlapping images
(src/check_body_accuracy.py:21-23). There is no calibrated population range to be had. A
dog is compared with its own baseline clips instead, which is the better comparison for
them anyway: same dog, same camera, so the body-length error is common-mode.

**Face features that divide by the face-box side.** At inference the side comes from
`derive_face_box()`, which needs SuperAnimal's head keypoints - it cannot be computed from
the annotations. Using DogFLW's own shipped box instead would put the reference on a
different scale from every runtime value, so the percentiles would be quietly wrong rather
than missing. The builder DETECTS these by recomputing each feature at two scales and
dropping whatever moves; `--with-model` runs the cascade and includes them properly, at
roughly an hour of CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "pupchat" / "reference.json"
ANNOTATIONS = ROOT / "data" / "dogflw" / "annotations.json"
PERCENTILES = (10, 25, 50, 75, 90)

MIN_N = 200
"""Below this a percentile is noise dressed as a population, so the feature is left out
and the evidence simply makes no comparison for it."""

BOOKKEEPING = {"body_L", "face_px", "head_tilt_used_ground", "n_paws", "tail_conf"}
"""Coverage columns, never signal (CLAUDE.md §10).

They survive the scale-free test because they are constants here - body is None, so
n_paws and tail_conf are 0 in every image - and a percentile over a column of zeros is
a comparison that means nothing while looking exactly like one that does.
"""


def scale_free(names: list[str], sample: np.ndarray) -> set[str]:
    """Which features mean the same thing without a true face-box side.

    Measured rather than listed: computed at two very different scales, anything that
    moves depends on the side and anything that does not is a pure angle or ratio. A
    hard-coded list would rot the first time a feature changed its normaliser.
    """
    from features import frame_features

    a = frame_features(None, sample, 100.0, None, names, 0.35)
    b = frame_features(None, sample, 400.0, None, names, 0.35)
    out = set()
    for k, va in a.items():
        vb = b.get(k)
        if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
            continue
        if math.isnan(va) and math.isnan(vb):
            continue
        if math.isfinite(va) and math.isfinite(vb) and abs(va - vb) < 1e-9:
            out.add(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-model", action="store_true",
                    help="run the cascade for a true face-box side, so the features "
                         "that divide by it are included too (~1 hour of CPU)")
    ap.add_argument("--snapshot",
                    default="face_project/face38/snapshot-best-008.pt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import keypoint_scheme as ks
    from features import frame_features

    from pupchat.reference import NO_POPULATION_BAND as NO_BAND

    if not ANNOTATIONS.exists():
        raise SystemExit(f"no DogFLW annotations at {ANNOTATIONS}")
    rows = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} annotated images")

    names = ks.ACTIVE_NAMES
    cascade = None
    if a.with_model:
        from cascade import Cascade, config_beside
        cascade = Cascade(config_beside(a.snapshot), a.snapshot)
        print("running the cascade for a true face-box side; this is the slow path")

    keep: set[str] | None = None
    values: dict[str, list[float]] = {}
    skipped = 0

    for i, row in enumerate(rows):
        lm = np.asarray(row.get("landmarks", []), dtype=float)
        if lm.shape[0] < 46 or not np.isfinite(lm).all():
            skipped += 1
            continue
        # Confidence 1 everywhere: these are human annotations, so nothing is gated out.
        gt = np.concatenate([lm[ks.ACTIVE], np.ones((len(ks.ACTIVE), 1))], axis=1)

        side = math.nan
        if cascade is not None:
            path = str(ROOT / "data" / "dogflw" / row["file"])
            try:
                res = cascade.run_image(path)
                side = res.face_box.side if res.face_box is not None else math.nan
            except Exception:                       # noqa: BLE001 - one bad image
                skipped += 1
                continue
            if not math.isfinite(side):
                skipped += 1
                continue

        if keep is None:
            keep = (None if cascade is not None else scale_free(names, gt))
            if keep is not None:
                print(f"{len(keep)} features are scale-free and can be built from the "
                      f"annotations alone")

        feats = frame_features(None, gt, side if cascade else 100.0, None, names, 0.35)
        for k, v in feats.items():
            if k in BOOKKEEPING or k in NO_BAND:
                continue
            if keep is not None and k not in keep:
                continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                values.setdefault(k, []).append(float(v))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")

    out: dict[str, dict] = {}
    for name, series in sorted(values.items()):
        if len(series) < MIN_N:
            print(f"  skipping {name}: only {len(series)} values")
            continue
        arr = np.asarray(series)
        entry = {"n": int(arr.size)}
        for p in PERCENTILES:
            entry[f"p{p}"] = round(float(np.percentile(arr, p)), 6)
        out[name] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "DogFLW ground-truth landmarks (photographs, not video)",
        "images": len(rows) - skipped,
        "with_model": bool(a.with_model),
        "note": "Face features only, and without --with-model only those that do not "
                "divide by the face-box side. Body features need the withers, which "
                "StanfordExtra never annotates.",
        "features": out,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(out)} features over "
          f"{len(rows) - skipped} images ({skipped} skipped)")
    for name, e in out.items():
        print(f"   {name:26s} n={e['n']:5d}  p10={e['p10']:9.4f}  "
              f"p50={e['p50']:9.4f}  p90={e['p90']:9.4f}")


if __name__ == "__main__":
    main()
