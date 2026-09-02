"""Split construction and leakage.

The old pipeline had no validation split at all and tuned thresholds on the reported
test images. These tests exist so that cannot silently come back.
"""
from __future__ import annotations

import pytest

from splits import Splits, breed_of, make_splits, verify


def _records(n_breeds: int = 12, per_breed: int = 20, test_per_breed: int = 4):
    recs = []
    for b in range(n_breeds):
        syn = f"n{2085620 + b:07d}"
        for k in range(per_breed):
            recs.append({"id": f"{syn}_{k:05d}",
                         "split": "test" if k < test_per_breed else "train"})
    return recs


def test_breed_of_parses_stanford_dogs_ids() -> None:
    assert breed_of("n02085620_11477") == "n02085620"
    assert breed_of("n02112137_61") == "n02112137"


def test_splits_are_disjoint() -> None:
    s = make_splits(_records())
    assert not (set(s.train) & set(s.val))
    assert not (set(s.train) & set(s.test))
    assert not (set(s.val) & set(s.test))


def test_test_split_is_untouched() -> None:
    """Val must be carved out of train only - never out of the reported test split."""
    recs = _records()
    expected = {r["id"] for r in recs if r["split"] == "test"}
    s = make_splits(recs)
    assert set(s.test) == expected


def test_every_train_image_lands_somewhere() -> None:
    recs = _records()
    train_ids = {r["id"] for r in recs if r["split"] == "train"}
    s = make_splits(recs)
    assert set(s.train) | set(s.val) == train_ids


def test_stratified_keeps_every_breed_in_both_train_and_val() -> None:
    s = make_splits(_records(), strategy="stratified")
    assert {breed_of(i) for i in s.val} == {breed_of(i) for i in s.train}


def test_breed_disjoint_shares_no_breed() -> None:
    s = make_splits(_records(), strategy="breed_disjoint")
    assert not ({breed_of(i) for i in s.val} & {breed_of(i) for i in s.train})


def test_splits_are_deterministic() -> None:
    a = make_splits(_records(), seed=7)
    b = make_splits(_records(), seed=7)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seeds_give_different_val() -> None:
    a = make_splits(_records(), seed=1)
    b = make_splits(_records(), seed=2)
    assert a.val != b.val


def test_verify_catches_overlap() -> None:
    bad = Splits(train=["a", "b"], val=["b"], test=["c"])
    with pytest.raises(AssertionError, match="both train and val"):
        verify(bad)


def test_verify_catches_duplicates() -> None:
    with pytest.raises(AssertionError, match="duplicate ids"):
        verify(Splits(train=["a", "a"], val=["b"], test=["c"]))


def test_verify_catches_empty_split() -> None:
    with pytest.raises(AssertionError, match="empty"):
        verify(Splits(train=["a"], val=[], test=["c"]))


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        make_splits(_records(), strategy="by_vibes")


def test_records_without_a_test_split_raise() -> None:
    recs = [{"id": "n1_1", "split": "train"}]
    with pytest.raises(ValueError, match="no train or no test"):
        make_splits(recs)
