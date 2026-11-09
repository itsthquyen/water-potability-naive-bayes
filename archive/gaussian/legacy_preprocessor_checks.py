"""Kiểm thử tiền xử lý bằng dữ liệu tổng hợp và đáp án tính tay độc lập."""

import copy
import json
import math
import random

import pytest

from archive.gaussian.legacy_preprocessor import Preprocessor


@pytest.mark.parametrize("strategy, values, expected", [
    ("mean", [1, 2, 9, None], 4),
    ("median", [9, 1, None, 2], 2),
    ("median", [8, None, 2, 4, 10], 6),
])
def test_imputation_hand_calculated(strategy, values, expected):
    """Mean=12/3; median lẻ=2; median chẵn=(4+8)/2."""
    matrix = [[value] for value in values]
    preprocessor = Preprocessor(strategy)
    assert preprocessor.fit(matrix) is preprocessor
    assert preprocessor.transform([[None], [float("nan")]]) == [[expected], [expected]]
    state = preprocessor.to_state()
    assert state["fill_values"] == [expected]
    assert state["raw_min"] == [min(value for value in values if value is not None)]
    assert state["raw_max"] == [max(value for value in values if value is not None)]


def test_standardization_after_imputation_ddof_zero():
    """Train [1,3,missing] thành [1,3,2], variance=2/3, cột hằng scale=1."""
    preprocessor = Preprocessor(standardize=True)
    output = preprocessor.fit_transform([[1, 7], [3, 7], [None, 7]])
    scale = math.sqrt(2 / 3)
    assert output[0] == pytest.approx([-1 / scale, 0])
    assert output[1] == pytest.approx([1 / scale, 0])
    assert output[2] == [0, 0]
    assert preprocessor.to_state()["means"] == [2, 7]
    assert preprocessor.to_state()["scales"] == pytest.approx([scale, 1])
    assert preprocessor.transform([[5, 9]])[0] == pytest.approx([3 / scale, 2])


def test_no_leakage_and_no_mutation():
    """Giá trị test rất lớn không đổi fill/min/max hay dữ liệu train."""
    train = [[2, 10], [4, None]]
    original = copy.deepcopy(train)
    preprocessor = Preprocessor().fit(train)
    before = preprocessor.to_state()
    assert preprocessor.transform([[1000, None]]) == [[1000, 10]]
    assert preprocessor.transform([[None, None]]) == [[3, 10]]
    assert preprocessor.to_state() == before
    assert train == original


def test_decimal_constant_and_median_center_after_imputation():
    """Cột hằng .1 không sinh scale nhỏ do sai số cộng; mean sau điền != median."""
    constant = Preprocessor(standardize=True)
    assert constant.fit_transform([[0.1], [0.1], [None]]) == [[0], [0], [0]]
    assert constant.to_state()["scales"] == [1]
    median = Preprocessor("median").fit([[1], [2], [9], [None]])
    assert median.to_state()["fill_values"] == [2]
    assert median.to_state()["means"] == [3.5]


@pytest.mark.parametrize("matrix", [
    [], [[]], [[1], [2, 3]], [1, 2], "12", None,
    [[None], [float("nan")]], [[True]], [["3"]], [[object()]],
    [[float("inf")]], [[-float("inf")]], [[1 + 2j]],
])
def test_invalid_training_matrix(matrix):
    with pytest.raises(ValueError):
        Preprocessor().fit(matrix)


@pytest.mark.parametrize("kwargs", [{"strategy": "mode"}, {"standardize": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Preprocessor(**kwargs)


def test_unfitted_wrong_width_and_numeric_overflow():
    with pytest.raises(ValueError):
        Preprocessor().transform([[1]])
    with pytest.raises(ValueError):
        Preprocessor().to_state()
    preprocessor = Preprocessor(standardize=True).fit([[1], [1]])
    with pytest.raises(ValueError):
        preprocessor.transform([[1, 2]])
    with pytest.raises(ValueError):
        Preprocessor().fit([[1e308], [-1e308]])
    wide_offset = Preprocessor(standardize=True).fit([[-1e308]])
    with pytest.raises(ValueError):
        wide_offset.transform([[1e308]])


def test_failed_refit_preserves_previous_state():
    preprocessor = Preprocessor().fit([[2], [4]])
    state = preprocessor.to_state()
    with pytest.raises(ValueError):
        preprocessor.fit([[None]])
    assert preprocessor.to_state() == state


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("standardize", [False, True])
def test_json_state_roundtrip_and_isolation(strategy, standardize):
    preprocessor = Preprocessor(strategy, standardize).fit([[1, 8], [3, 8], [None, 8]])
    state = json.loads(json.dumps(preprocessor.to_state(), allow_nan=False))
    restored = Preprocessor.from_state(state)
    sample = [[None, 10], [7, None]]
    assert restored.transform(sample) == preprocessor.transform(sample)
    state["fill_values"][0] = 999
    exported = restored.to_state()
    exported["means"][0] = 999
    assert restored.transform(sample) == preprocessor.transform(sample)


@pytest.mark.parametrize("key, value", [
    ("version", True), ("version", 2), ("n_features", 0),
    ("n_features", True), ("n_features", 1.0), ("strategy", "bad"),
    ("standardize", 1), ("fill_values", []), ("means", [float("nan")]),
    ("scales", [0]), ("scales", [-1]), ("scales", [float("inf")]),
    ("scales", [True]), ("raw_min", [5]), ("raw_max", [0]),
    ("fill_values", [100]), ("fill_values", [1]),
    ("means", [-100]), ("raw_min", (1,)),
])
def test_invalid_state_fields(key, value):
    state = Preprocessor().fit([[1], [3]]).to_state()
    state[key] = value
    with pytest.raises(ValueError):
        Preprocessor.from_state(state)


def test_invalid_state_schema_and_constant_scale():
    state = Preprocessor().fit([[2], [2]]).to_state()
    for invalid in (None, [], {}, {**state, "unknown": 1}):
        with pytest.raises(ValueError):
            Preprocessor.from_state(invalid)
    del state["means"]
    with pytest.raises(ValueError):
        Preprocessor.from_state(state)
    state = Preprocessor().fit([[2], [2]]).to_state()
    state["scales"] = [2]
    with pytest.raises(ValueError):
        Preprocessor.from_state(state)


def test_fit_transform_iterator():
    assert Preprocessor().fit_transform(iter([[1], [3], [None]])) == [[1], [3], [2]]
