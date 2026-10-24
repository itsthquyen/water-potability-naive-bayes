"""Đáp án tính tay độc lập trên dữ liệu tổng hợp nhỏ, không phải thực nghiệm."""

import json

import pytest

from src.metrics import evaluate_predictions, roc_auc_score, summarize_scores


def test_confusion_and_hand_calculated_metrics():
    result = evaluate_predictions([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert result[name] == pytest.approx(2 / 3)
    assert [row["label"] for row in result["per_class"]] == [0, 1]
    for row in result["per_class"]:
        assert row["support"] == 3
        for name in ("precision", "recall", "f1"):
            assert row[name] == pytest.approx(2 / 3)
            assert row["undefined"][name] is False
    assert result["roc_auc"] is None
    assert result["roc_curve"] is None


def test_asymmetric_counts():
    result = evaluate_predictions([0, 0, 0, 1, 1], [0, 0, 1, 0, 0])
    assert result["confusion_matrix"] == [[2, 1], [2, 0]]
    assert result["accuracy"] == pytest.approx(2 / 5)
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 7)
    assert result["per_class"][0]["precision"] == pytest.approx(1 / 2)
    assert result["per_class"][0]["recall"] == pytest.approx(2 / 3)


def test_baseline_precision_undefined_but_f1_defined_zero():
    result = evaluate_predictions([0, 0, 0, 1], [0, 0, 0, 0], [0.25] * 4)
    positive = result["per_class"][1]
    assert positive["precision"] is None
    assert positive["recall"] == positive["f1"] == 0.0
    assert positive["undefined"] == {"precision": True, "recall": False, "f1": False}
    assert result["macro_f1"] == pytest.approx(3 / 7)
    assert result["balanced_accuracy"] == 0.5
    assert result["roc_auc"] == 0.5


def test_absent_class_is_not_removed_from_macro_and_json_is_strict():
    result = evaluate_predictions([0, 0], [0, 0], [0.1, 0.2])
    assert result["per_class"][1]["undefined"] == dict.fromkeys(
        ["precision", "recall", "f1"], True)
    assert result["per_class"][1]["f1"] is None
    assert result["macro_f1"] == result["balanced_accuracy"] == 0.5
    assert result["roc_auc"] is None
    assert result["roc_curve"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scores, expected", [([0, 0, 1, 1], 1.0),
    ([1, 1, 0, 0], 0.0), ([0.5] * 4, 0.5), ([0, 1, 1, 2], 0.875)])
def test_auc_hand_rankings_and_ties(scores, expected):
    assert roc_auc_score([0, 0, 1, 1], scores) == pytest.approx(expected)


def test_roc_groups_ties_before_adding_points():
    result = evaluate_predictions([0, 1], [0, 1], [0.5, 0.5])
    assert result["roc_curve"] == [
        {"threshold": None, "fpr": 0.0, "tpr": 0.0},
        {"threshold": 0.5, "fpr": 1.0, "tpr": 1.0}]
    json.dumps(result, allow_nan=False)
    assert roc_auc_score([1], [4]) is None


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 1], [0]),
    ([0, 2], [0, 1]), ([0], [float("nan")]), ([0], [True])])
def test_invalid_auc_inputs(labels, scores):
    with pytest.raises(ValueError):
        roc_auc_score(labels, scores)


@pytest.mark.parametrize("actual,predicted,scores", [([], [], None),
    ([0], [], None), ([0], [1, 0], None), ([2], [0], None),
    ([True], [0], None), (["0"], [0], None), ([0], [float("nan")], None),
    ([0], [0], []), ([0], [0], [float("inf")]), ([0], [0], ["0.1"])])
def test_invalid_evaluation_inputs(actual, predicted, scores):
    with pytest.raises(ValueError):
        evaluate_predictions(actual, predicted, scores)


def test_score_mean_population_std_hand_example():
    result = summarize_scores([1, 2, 3, 4, 5])
    assert result == {"mean": 3.0, "std": pytest.approx(2 ** 0.5)}
    assert summarize_scores([0.7]) == {"mean": 0.7, "std": 0.0}


@pytest.mark.parametrize("values", [[], [None], [float("nan")], [float("inf")],
    [True], ["1"], [-1e308, 1e308]])
def test_invalid_or_overflowing_score_summary(values):
    with pytest.raises(ValueError):
        summarize_scores(values)
