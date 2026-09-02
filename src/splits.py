"""Train / val / test partition, with the leakage checks the old pipeline never had.

The old code tuned every threshold on the same 479 images it reported. This carves a
validation split out of DogFLW's train split and leaves test untouched until the end.

What identity is available, measured rather than assumed:

    DogFLW is built on Stanford Dogs. Every id looks like `n02085620_11477`, where the
    prefix is a WordNet synset - n02085620 is Chihuahua. So BREED is exactly inferable
    and INDIVIDUAL DOG is not. All 4,335 filenames are unique, so there is no image-level
    leakage in the shipped split.

    The shipped split shares all 120 breeds between train and test, i.e. it is a random
    split by image, not by breed.

That last fact drives the default. A breed-disjoint validation split would be
systematically harder than the test split we report on, so thresholds tuned against it
would be tuned for the wrong distribution. `stratified` matches the test split's
construction and is the default; `breed_disjoint` is available for a stricter
generalisation check, and should be read as a different question, not a better one.

Residual risk that cannot be resolved here: Stanford Dogs may hold several photographs
of the same individual dog within a breed, so near-duplicates may straddle any split.
That is equally true of the shipped test split, and therefore of the 0.0438 baseline, so
it does not bias the comparison this project makes - but it does mean absolute numbers
on DogFLW may be mildly optimistic in a way no split strategy here can fix.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Splits:
    train: list[str]
    val: list[str]
    test: list[str]

    def summary(self) -> str:
        return (f"train {len(self.train)}  val {len(self.val)}  test {len(self.test)}  "
                f"(total {len(self.train) + len(self.val) + len(self.test)})")


def breed_of(image_id: str) -> str:
    """Stanford Dogs synset prefix, e.g. 'n02085620_11477' -> 'n02085620'."""
    return image_id.split("_", 1)[0]


def make_splits(records: list[dict], val_frac: float = 0.15, seed: int = 42,
                strategy: str = "stratified") -> Splits:
    """Partition DogFLW records, carving val out of the shipped train split.

    Args:
        records: DogFLW annotation records, each with 'id' and 'split'.
        val_frac: fraction of the shipped train split to hold out.
        seed: RNG seed. Splits are a function of (records, val_frac, seed, strategy).
        strategy: 'stratified' (default, matches how test was built) or 'breed_disjoint'.

    Returns:
        Splits with disjoint id lists. `verify` is called before returning, so a
        malformed partition raises here rather than surfacing as an inflated score.
    """
    train_ids = [r["id"] for r in records if r["split"] == "train"]
    test_ids = [r["id"] for r in records if r["split"] == "test"]
    if not train_ids or not test_ids:
        raise ValueError("records contain no train or no test split")

    rng = np.random.default_rng(seed)
    if strategy == "stratified":
        tr, va = _stratified(train_ids, val_frac, rng)
    elif strategy == "breed_disjoint":
        tr, va = _breed_disjoint(train_ids, val_frac, rng)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    s = Splits(train=sorted(tr), val=sorted(va), test=sorted(test_ids))
    verify(s)
    return s


def _stratified(ids: list[str], val_frac: float,
                rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Hold out val_frac of each breed, so val and train have the same breed mix."""
    by_breed: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_breed[breed_of(i)].append(i)
    train, val = [], []
    for breed in sorted(by_breed):
        members = sorted(by_breed[breed])
        rng.shuffle(members)
        # At least one val image per breed, but never the whole breed.
        k = min(max(1, int(round(len(members) * val_frac))), len(members) - 1)
        val += members[:k]
        train += members[k:]
    return train, val


def _breed_disjoint(ids: list[str], val_frac: float,
                    rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Hold out whole breeds. A harder, different question - see module docstring."""
    breeds = sorted({breed_of(i) for i in ids})
    rng.shuffle(breeds)
    k = max(1, int(round(len(breeds) * val_frac)))
    held = set(breeds[:k])
    train = [i for i in ids if breed_of(i) not in held]
    val = [i for i in ids if breed_of(i) in held]
    return train, val


def verify(splits: Splits) -> None:
    """Fail loudly on any overlap. Called by make_splits; call again after any edit."""
    tr, va, te = set(splits.train), set(splits.val), set(splits.test)
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        common = a & b
        if common:
            raise AssertionError(
                f"{len(common)} ids appear in both {na} and {nb}, e.g. "
                f"{sorted(common)[:5]}"
            )
    for name, ids in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{name} contains duplicate ids")
        if not ids:
            raise AssertionError(f"{name} split is empty")


def load(path: str | Path) -> Splits:
    d = json.loads(Path(path).read_text())
    s = Splits(train=d["train"], val=d["val"], test=d["test"])
    verify(s)
    return s


def save(splits: Splits, path: str | Path, meta: dict | None = None) -> None:
    verify(splits)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"train": splits.train, "val": splits.val,
                             "test": splits.test, "meta": meta or {}}, indent=1))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="data/dogflw/annotations.json")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", choices=["stratified", "breed_disjoint"],
                    default="stratified")
    a = ap.parse_args()

    recs = json.loads(Path(a.annotations).read_text())
    s = make_splits(recs, val_frac=a.val_frac, seed=a.seed, strategy=a.strategy)
    save(s, a.out, meta={"strategy": a.strategy, "val_frac": a.val_frac, "seed": a.seed})
    print(s.summary())
    for name, ids in (("train", s.train), ("val", s.val), ("test", s.test)):
        print(f"  {name:5s} {len(ids):5d} images, {len({breed_of(i) for i in ids}):3d} breeds")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
