"""Kiểm chứng EDA bằng số liệu tổng hợp và đáp án tính tay độc lập."""

import json

import pytest

from src.eda import histogram_counts, pearson_correlation, summarize_features


def test_even_median_linear_quartiles_and_missing():
    result = summarize_features([[1], [2], [3], [4], [None], [float("nan")]], ["a"])["a"]
    assert result["count"] == 4
    assert result["missing"] == 2
    assert result["min"] == 1
    assert result["max"] == 4
    assert result["mean"] == result["median"] == 2.5
    assert result["quartiles"] == {"q1": 1.75, "q2": 2.5, "q3": 3.25}
    assert result["boxplot"] == {"lower_fence": -0.5, "upper_fence": 5.5,
                                 "whisker_low": 1, "whisker_high": 4, "outliers": []}


def test_odd_median_boxplot_outlier_and_whiskers_are_observations():
    result = summarize_features([[1], [2], [3], [4], [100]], ["a"])["a"]
    assert result["median"] == 3
    assert result["mean"] == pytest.approx(22)
    assert result["quartiles"] == {"q1": 2, "q2": 3, "q3": 4}
    assert result["boxplot"] == {"lower_fence": -1, "upper_fence": 7,
                                 "whisker_low": 1, "whisker_high": 4, "outliers": [100]}


def test_empty_and_constant_columns_are_json_safe():
    result = summarize_features([[None, 2], [float("nan"), 2]], ["empty", "constant"])
    assert result["empty"]["count"] == 0
    assert result["empty"]["missing"] == 2
    for name in ("min", "max", "mean", "median"):
        assert result["empty"][name] is None
    assert result["empty"]["quartiles"] == dict.fromkeys(["q1", "q2", "q3"])
    assert result["constant"]["boxplot"]["whisker_low"] == 2
    assert result["constant"]["boxplot"]["outliers"] == []
    assert summarize_features([], ["a"])["a"]["count"] == 0
    json.dumps(result, allow_nan=False)


def test_histogram_internal_edge_right_and_last_edge_inclusive():
    result = histogram_counts([0, 1, 2, 3, 4, None, float("nan")], bins=2)
    assert result == {"edges": [0, 2, 4], "counts": [2, 3]}
    assert histogram_counts([-1, 0, 1, 2, 3], bins=[0, 1, 2]) == {
        "edges": [0, 1, 2], "counts": [1, 2]}
    assert histogram_counts([2, 2], bins=2) == {"edges": [1.5, 2, 2.5], "counts": [0, 2]}
    assert histogram_counts([], bins=2) == {"edges": [0, 0.5, 1], "counts": [0, 0]}


def test_pearson_pairwise_missing_hand_values():
    value, count = pearson_correlation([1, 2, None, 3, 100], [2, 4, 7, 6, None])
    assert count == 3
    assert value == pytest.approx(1)
    value, count = pearson_correlation([1, 2, 3], [3, 2, 1])
    assert (value, count) == (pytest.approx(-1), 3)
    value, count = pearson_correlation([1, 2, 3], [1, 3, 2])
    assert value == pytest.approx(0.5)
    assert count == 3


def test_pearson_constant_insufficient_and_empty():
    assert pearson_correlation([0.1] * 11, [0.1] * 11) == (None, 11)
    assert pearson_correlation(list(range(11)), [0.1] * 11) == (None, 11)
    assert pearson_correlation([0.1] * 11, list(range(11))) == (None, 11)
    assert pearson_correlation([1, 1], [2, 3]) == (None, 2)
    assert pearson_correlation([1, None], [2, 3]) == (None, 1)
    assert pearson_correlation([float("nan")], [2]) == (None, 0)
    assert pearson_correlation([], []) == (None, 0)


@pytest.mark.parametrize("rows,names", [([[1, 2]], ["a"]), ([[1]], ["a", "a"]),
    ([[1]], [""]), ([], []), ([["1"]], ["a"]), ([[float("inf")]], ["a"]),
    ([[True]], ["a"]), ([[-1e308], [1e308]], ["a"])])
def test_invalid_feature_summary(rows, names):
    with pytest.raises(ValueError):
        summarize_features(rows, names)


@pytest.mark.parametrize("bins", [0, -1, True, 1.5, [1], [1, 1], [2, 1],
    [0, float("inf")], [0, None]])
def test_invalid_histogram_bins(bins):
    with pytest.raises(ValueError):
        histogram_counts([1, 2], bins)


def test_invalid_pearson_inputs_and_overflow():
    with pytest.raises(ValueError):
        pearson_correlation([1], [])
    with pytest.raises(ValueError):
        pearson_correlation([None], [float("inf")])
    with pytest.raises(ValueError):
        pearson_correlation([-1e308, 1e308], [1, 2])
