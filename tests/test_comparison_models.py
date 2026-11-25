"""Kiểm chứng thuật toán tự cài bằng đáp án tính tay."""

import math
import pytest

from src.comparison_models import NumericPreprocessor, DecisionTree, entropy


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
    for criterion in ("gain",):
        tree = DecisionTree(criterion).fit(X, y)
        assert tree.root["column"] == 0
        assert tree.root["gain"] == 1
        assert tree.predict_with_scores(X) == (y, [0, 0, 1, 1])
        assert tree.predict_with_scores([[2, 0]]) == ([0], [0.5])
