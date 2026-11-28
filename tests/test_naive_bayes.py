"""Đáp án tính tay cho Naive Bayes đếm, kể cả tần suất 0 và chia khoảng."""

import pytest

from src.data_processing.discretization import Discretizer
from src.algorithms.naive_bayes import NaiveBayes


def test_hand_calculated_counts_and_posterior():
    model = NaiveBayes().fit([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]],
                            [0, 0, 0, 1, 1], [2, 2])
    assert model.to_state()["feature_counts"] == [[[2, 1], [2, 1]], [[0, 2], [1, 1]]]
    # Với (1, 0): lớp 0 có (3/5)(1/3)(2/3)=2/15; lớp 1 có (2/5)(2/2)(1/2)=1/5.
    labels, probabilities = model.predict_with_proba([[1, 0], [0, 1]])
    assert labels == [1, 0]
    assert probabilities[0] == pytest.approx([0.4, 0.6])
    assert probabilities[1] == [1.0, 0.0]  # Giữ tần suất 0, không smoothing.


def test_equal_positive_scores_and_zero_evidence_are_different():
    tie = NaiveBayes().fit([[0], [0]], [0, 1], [1])
    assert tie.predict_with_proba([[0]]) == ([0], [[0.5, 0.5]])
    disjoint = NaiveBayes().fit([[0, 0], [1, 1]], [0, 1], [2, 2])
    with pytest.raises(ValueError, match="cả hai điểm"):
        disjoint.predict_with_proba([[0, 1]])


def test_discretization_train_only_median_and_constant_column():
    discretizer = Discretizer().fit([[1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [None, 8]])
    assert discretizer.to_state()["cut_points"] == [[2.5, 4.5], []]
    assert discretizer.to_state()["fill_values"] == [3.5, 8.0]
    assert discretizer.category_sizes == [3, 1]
    assert discretizer.transform([[2.5, 8], [4.5, 8], [None, 8], [1e308, 9]]) == [
        [0, 0], [1, 0], [1, 0], [2, 0],
    ]
    assert discretizer.to_state()["cut_points"] == [[2.5, 4.5], []]


def test_does_not_split_equal_values_or_accept_missing_training_column():
    assert Discretizer().fit([[1], [1], [1], [1], [2], [3]]).category_sizes == [2]
    with pytest.raises(ValueError, match="thiếu hoàn toàn"):
        Discretizer().fit([[1, None], [2, None]])


def test_midpoint_rounds_once_for_subnormal_median_and_avoids_overflow():
    smallest = float.fromhex("0x0.0000000000001p-1022")
    lower = float.fromhex("0x1.4p+1023")
    upper = float.fromhex("0x1.cp+1023")
    state = Discretizer().fit([[2 * smallest, lower], [3 * smallest, upper]]).to_state()
    assert state["fill_values"] == [2 * smallest, float.fromhex("0x1.8p+1023")]



def test_invalid_count_state_is_rejected():
    with pytest.raises(ValueError, match="Bảng đếm"):
        NaiveBayes.from_state({"category_sizes": [2], "class_counts": [1, 1],
                               "feature_counts": [[[1, 0]], [[1, 1]]]})
