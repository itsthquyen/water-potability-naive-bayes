"""Kiểm chứng thuật toán tự cài bằng đáp án tính tay."""

import math
import pytest

from src.comparison_models import NumericPreprocessor, DecisionTree, entropy, KNearestNeighbors


def test_numeric_fit_does_not_read_validation():
    processor = NumericPreprocessor().fit([[1, 8], [3, 8], [None, 8]])
    assert processor.state["medians"] == [2, 8]
    assert processor.state["means"] == [2, 8]
    assert processor.state["scales"] == [math.sqrt(2 / 3), 1]
    snapshot = repr(processor.state)
    assert processor.transform([[2, 8]]) == [[0, 0]]
    processor.transform([[1000, 1000]])
    assert repr(processor.state) == snapshot


def test_entropy_and_both_tree_criteria():
    assert entropy([2, 2]) == 1
    assert entropy([0, 4]) == 0
    X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 1, 1]
    for criterion in ("gain", "gain_ratio"):
        tree = DecisionTree(criterion).fit(X, y)
        assert tree.root["column"] == 0
        assert tree.root["gain"] == 1
        assert tree.predict_with_scores(X) == (y, [0, 0, 1, 1])
        assert tree.predict_with_scores([[2, 0]]) == ([0], [0.5])


def test_knn_distances_and_votes_are_hand_calculated():
    model = KNearestNeighbors(3).fit([[0], [2], [5], [8]], [0, 1, 1, 0])
    assert model.predict_with_scores([[3]]) == ([1], [2 / 3])
    tie = KNearestNeighbors(2).fit([[0], [2]], [0, 1])
    assert tie.predict_with_scores([[1]]) == ([0], [0.5])
    with pytest.raises(ValueError):
        KNearestNeighbors(5).fit([[1]], [0])
