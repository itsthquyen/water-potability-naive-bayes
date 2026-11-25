"""Kiểm chứng độc lập các bộ phân lớp so sánh và thống kê train-only."""

import math
import pytest

from src.comparison_models import DecisionTree, KNearestNeighbors, LogisticRegression, LinearSVM, NumericPreprocessor, entropy
from src.six_experiments import ComparisonPipeline, ALGORITHMS


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


def test_one_logistic_step_matches_gradient_at_origin():
    model = LogisticRegression(epochs=1, learning_rate=1, l2=0).fit([[-1], [1]], [0, 1])
    assert model.weights == [0.5] and model.bias == 0
    assert model.predict_with_scores([[-1], [1]])[0] == [0, 1]
    assert model.predict_with_scores([[1]])[1][0] == pytest.approx(1 / (1 + math.exp(-0.5)))


def test_one_svm_step_matches_hinge_gradient():
    model = LinearSVM(epochs=1, learning_rate=0.1, l2=0.5).fit([[2]], [1])
    assert model.weights == [0.2] and model.bias == 0.1
    assert model.predict_with_scores([[2]]) == ([1], [0.5])


def test_numeric_fit_does_not_read_validation():
    processor = NumericPreprocessor().fit([[1, 8], [3, 8], [None, 8]])
    assert processor.state["medians"] == [2, 8]
    assert processor.state["means"] == [2, 8]
    assert processor.state["scales"] == [math.sqrt(2 / 3), 1]
    snapshot = repr(processor.state)
    assert processor.transform([[2, 8]]) == [[0, 0]]
    processor.transform([[1000, 1000]])
    assert repr(processor.state) == snapshot


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_pipeline_serialization_preserves_predictions(algorithm):
    X = [[float(i)] * 9 for i in range(12)]
    pipeline = ComparisonPipeline(algorithm).fit(X, [0] * 6 + [1] * 6)
    restored = ComparisonPipeline.from_state(pipeline.to_state())
    assert restored.predict_with_scores(X) == pipeline.predict_with_scores(X)
