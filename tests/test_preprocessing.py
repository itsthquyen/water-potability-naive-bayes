"""Kiểm thử tiền xử lý bằng dữ liệu tổng hợp và đáp án tính tay độc lập."""

import math
import random

import pytest

from src.preprocessing import stratified_folds, stratified_holdout


def test_holdout_rounding_and_invariants():
    """n0=7, n1=10, f=.25: số test lần lượt floor(2.25)=2 và floor(3)=3."""
    labels = [0] * 7 + [1] * 10
    dev, test = stratified_holdout(labels, 0.25)
    assert sum(labels[index] == 0 for index in test) == 2
    assert sum(labels[index] == 1 for index in test) == 3
    assert not set(dev) & set(test)
    assert sorted(dev + test) == list(range(len(labels)))
    assert {labels[index] for index in dev} == {0, 1}
    assert {labels[index] for index in test} == {0, 1}
    assert (dev, test) == stratified_holdout(labels, 0.25)
    assert (dev, test) != stratified_holdout(labels, 0.25, seed=43)
    assert labels == [0] * 7 + [1] * 10


def test_holdout_half_rounds_up_not_bankers_rounding():
    _, test = stratified_holdout([0] * 5 + [1] * 5, 0.5)
    assert len(test) == 6


def test_fold_invariants_and_seed():
    labels = [0] * 13 + [1] * 11
    folds = stratified_folds(labels)
    assert len(folds) == 5
    validation_occurrences = []
    counts = {0: [], 1: []}
    for train, validation in folds:
        assert not set(train) & set(validation)
        assert sorted(train + validation) == list(range(len(labels)))
        assert {labels[index] for index in train} == {0, 1}
        assert {labels[index] for index in validation} == {0, 1}
        validation_occurrences.extend(validation)
        for label in (0, 1):
            counts[label].append(sum(labels[index] == label for index in validation))
    assert sorted(validation_occurrences) == list(range(len(labels)))
    for values in counts.values():
        assert max(values) - min(values) <= 1
    assert folds == stratified_folds(labels)
    assert folds != stratified_folds(labels, seed=43)


def test_split_does_not_change_global_random_state():
    before = random.getstate()
    stratified_holdout([0] * 10 + [1] * 10)
    stratified_folds([0] * 10 + [1] * 10)
    assert random.getstate() == before


@pytest.mark.parametrize("labels", [[], [0, 0], [1], [0, 2], [False, 1],
                                        [0.0, 1], [None, 1], [[0], [1]], None, "01"])
@pytest.mark.parametrize("splitter", [stratified_holdout, stratified_folds])
def test_invalid_labels(splitter, labels):
    with pytest.raises(ValueError):
        splitter(labels)


@pytest.mark.parametrize("seed", [True, -1, 1.5, None, "42"])
@pytest.mark.parametrize("splitter", [stratified_holdout, stratified_folds])
def test_invalid_seed(splitter, seed):
    with pytest.raises(ValueError):
        splitter([0] * 10 + [1] * 10, seed=seed)


@pytest.mark.parametrize("fraction", [True, 0, 1, -0.2, float("nan"), float("inf"), "0.2"])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        stratified_holdout([0] * 10 + [1] * 10, fraction)


@pytest.mark.parametrize("n_folds", [True, 0, 1, 2.5, "5"])
def test_invalid_fold_count(n_folds):
    with pytest.raises(ValueError):
        stratified_folds([0] * 10 + [1] * 10, n_folds)


def test_small_class_failures_and_two_fold_boundary():
    with pytest.raises(ValueError):
        stratified_holdout([0] + [1] * 10)
    with pytest.raises(ValueError):
        stratified_holdout([0] * 2 + [1] * 10, 0.2)
    with pytest.raises(ValueError):
        stratified_holdout([0] * 2 + [1] * 10, 0.9)
    with pytest.raises(ValueError):
        stratified_folds([0] * 4 + [1] * 10)
    assert len(stratified_folds([0, 0, 1, 1], n_folds=2)) == 2
