"""Does keypoint-derived geometry carry affective signal? Two tests, one strict.

Reads the feature rows written by pilot_extract.py and runs:

1. **Unsupervised.** Standardise, then k-means with k = number of classes. The labels are
   NOT passed to the fit - they are used afterwards only to score how well the clusters
   happen to line up. This is the honest form of "the machine found it on its own": if
   emotion is the dominant structure in these features, clustering finds it without being
   told what to look for.

2. **Supervised.** Logistic regression, leave-one-dog-out. Answers a different question -
   how much signal is present when you *are* allowed to fit to the labels.

Two guards, because with a few hundred rows and a dozen features it is easy to read noise
as a result:

* **Splitting is by DOG, never by frame.** Frames from one video are near-duplicates; let
  them straddle train and test and the score measures memorisation. Without a dog column
  the script still runs but says plainly that the number is inflated.
* **A permutation baseline.** The same pipeline on shuffled labels, repeated. That is what
  "no signal" looks like on this exact data, and the real score has to clear it.

Expect to land well below the 89% a DINO-ViT reached on the same images - that model uses
every pixel and this uses about a dozen numbers. Meaningfully above chance AND above the
permutation baseline is the result that matters.

    python src/pilot_model.py --features outputs/pilot_features.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = {"id", "label", "dog"}
# Bookkeeping columns describe measurement quality, not the dog. Feeding them to a
# classifier lets it key on how the frame was captured instead of what the dog was doing.
NOT_FEATURES = {"tail_conf", "head_tilt_used_ground", "face_px", "body_L", "n_paws"}


def cluster_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Best accuracy over all one-to-one assignments of cluster ids to labels."""
    from itertools import permutations
    classes = np.unique(y)
    best = 0.0
    for perm in permutations(range(len(classes))):
        mapped = np.empty_like(pred)
        for ci, p in enumerate(perm):
            mapped[pred == p] = ci
        best = max(best, float((classes[mapped] == y).mean()))
    return best


def prepare(df: pd.DataFrame, min_coverage: float) -> tuple[pd.DataFrame, list[str], dict]:
    cols = [c for c in df.columns if c not in META and c not in NOT_FEATURES]
    cov = {c: float(df[c].notna().mean()) for c in cols}
    keep = [c for c in cols if cov[c] >= min_coverage]
    dropped = {c: round(cov[c], 3) for c in cols if c not in keep}
    X = df[keep].copy()
    n_missing = int(X.isna().sum().sum())
    X = X.fillna(X.median(numeric_only=True))
    return X, keep, {"dropped_low_coverage": dropped, "imputed_cells": n_missing,
                     "coverage": {c: round(cov[c], 3) for c in keep}}


def supervised(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
               seed: int) -> tuple[float, str]:
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    if groups is not None and len(np.unique(groups)) >= 3:
        splitter, gs, how = LeaveOneGroupOut(), groups, "leave-one-dog-out"
    else:
        splitter, gs, how = StratifiedKFold(5, shuffle=True, random_state=seed), None, \
            "stratified 5-fold (NO dog column - score is INFLATED)"
    correct = total = 0
    for tr, te in splitter.split(X, y, gs):
        if len(np.unique(y[tr])) < 2:
            continue
        model.fit(X[tr], y[tr])
        correct += int((model.predict(X[te]) == y[te]).sum())
        total += len(te)
    return (correct / total if total else float("nan")), how


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("outputs/pilot_features.csv"))
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--permutations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/pilot_result.json"))
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    df = df[df["label"].notna() & (df["label"].astype(str) != "")]
    X_df, keep, prep = prepare(df, a.min_coverage)
    classes, y = np.unique(df["label"].astype(str), return_inverse=True)
    groups = df["dog"].astype(str).values if df["dog"].notna().any() else None
    if groups is not None and set(groups) <= {"", "nan"}:
        groups = None
    X = X_df.to_numpy(dtype=float)
    chance = float(pd.Series(y).value_counts(normalize=True).max())

    # ---- unsupervised: labels withheld from the fit --------------------------------
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=len(classes), n_init=20, random_state=a.seed).fit(Z)
    ari = float(adjusted_rand_score(y, km.labels_))
    cacc = cluster_accuracy(y, km.labels_)

    # ---- supervised ----------------------------------------------------------------
    acc, how = supervised(X, y, groups, a.seed)

    # ---- what no signal looks like on this same data --------------------------------
    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.permutations):
        yp = rng.permutation(y)
        null.append(supervised(X, yp, groups, a.seed)[0])
    null = np.array([v for v in null if np.isfinite(v)])
    p_val = float((null >= acc).mean()) if null.size else float("nan")

    out = {
        "n_rows": int(len(df)), "classes": classes.tolist(),
        "n_features": len(keep), "features": keep, "prep": prep,
        "n_dogs": int(len(np.unique(groups))) if groups is not None else 0,
        "chance": chance,
        "unsupervised": {"adjusted_rand": ari, "cluster_accuracy": cacc},
        "supervised": {"accuracy": acc, "scheme": how},
        "permutation_baseline": {
            "mean": float(null.mean()) if null.size else None,
            "p95": float(np.percentile(null, 95)) if null.size else None,
            "p_value": p_val, "n": int(null.size)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))

    print(f"\n{'='*72}\nPILOT: can features derived from keypoints separate the classes?")
    print(f"{'='*72}")
    print(f"rows {len(df)}   classes {list(classes)}   features {len(keep)}   "
          f"dogs {out['n_dogs'] or 'NONE'}")
    if prep["dropped_low_coverage"]:
        print(f"dropped below {a.min_coverage:.0%} coverage: "
              f"{list(prep['dropped_low_coverage'])}")
    print(f"imputed cells: {prep['imputed_cells']}")
    print()
    print(f"  chance (majority class)        {chance:6.1%}")
    print(f"  UNSUPERVISED cluster accuracy  {cacc:6.1%}   (labels never seen while fitting)")
    print(f"     adjusted Rand index         {ari:+6.3f}   (0 = clusters unrelated to labels)")
    print(f"  SUPERVISED accuracy            {acc:6.1%}   [{how}]")
    if null.size:
        print(f"  shuffled-label baseline        {null.mean():6.1%} mean, "
              f"{np.percentile(null,95):.1%} at p95   (n={null.size})")
        print(f"     p-value                     {p_val:.3f}")
    print()
    if null.size and p_val <= 0.05 and acc > chance:
        print("  VERDICT: signal present - above chance and clears the shuffled baseline.")
    elif acc <= chance:
        print("  VERDICT: no signal - at or below chance.")
    else:
        print("  VERDICT: not separable from the shuffled baseline; treat as no result.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
