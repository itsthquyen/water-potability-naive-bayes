"""Kiểm thử dữ liệu tổng hợp, đáp án tính tay độc lập; không dùng ML oracle."""

import json
import math

import pytest

from src.config import DEFAULT_VAR_SMOOTHING, VARIANCE_FLOOR
from src.gaussian_nb_numpy import GaussianNaiveBayes


def hand_model():
    """Hai lớp có mean 1/5, MLE 1; global MLE 5, epsilon 1."""
    return GaussianNaiveBayes(var_smoothing=0.2, variance_floor=0.01).fit(
        [[0], [2], [4], [6]], [0, 0, 1, 1]
    )


def test_hand_calculated_parameters_and_posteriors():
    model = hand_model()
    state = model.to_state()
    assert state["class_counts"] == [2, 2]
    assert state["class_priors"] == [0.5, 0.5]
    assert state["means"] == [[1.0], [5.0]]
    assert state["variances_mle"] == [[1.0], [1.0]]
    assert state["global_variances_mle"] == [5.0]
    assert state["epsilon"] == 1.0
    assert state["variances_effective"] == [[2.0], [2.0]]
    # Ở x=1, tỉ số likelihood lớp 1/lớp 0 = exp(-16/(2*2)) = exp(-4).
    expected = [1 / (1 + math.exp(-4)), 1 / (1 + math.exp(4))]
    assert model.predict_proba([[1]])[0] == pytest.approx(expected, abs=1e-14)
    assert model.predict_log_proba([[1]])[0] == pytest.approx(
        [math.log(value) for value in expected], abs=1e-14
    )
    expected_joint = math.log(0.5) - 0.5 * math.log(4 * math.pi)
    assert model._joint_log_scores([[1]])[0] == pytest.approx(
        [expected_joint, expected_joint - 4], abs=1e-14
    )
    assert model.predict([[1], [5], [3]]) == [0, 1, 0]


def test_multiple_features_unequal_priors_and_constant_column():
    model = GaussianNaiveBayes(var_smoothing=0.1, variance_floor=1).fit(
        [[0, 7], [2, 7], [0, 7], [2, 7], [4, 7], [6, 7]],
        [0, 0, 0, 0, 1, 1],
    )
    state = model.to_state()
    assert state["class_priors"] == [2 / 3, 1 / 3]
    assert state["means"] == [[1, 7], [5, 7]]
    assert state["variances_mle"] == [[1, 0], [1, 0]]
    # Tại trung điểm 3, hai likelihood bằng nhau: hậu nghiệm bằng prior.
    assert model.predict_proba([[3, 7]])[0] == pytest.approx([2 / 3, 1 / 3])


def test_all_constant_singletons_floor_and_tie():
    model = GaussianNaiveBayes().fit([[9, 9], [9, 9]], [1, 0])
    state = model.to_state()
    assert state["classes"] == [0, 1]
    assert state["global_variances_mle"] == [0, 0]
    assert state["epsilon"] == VARIANCE_FLOOR
    assert state["var_smoothing"] == DEFAULT_VAR_SMOOTHING
    assert model.predict_proba([[9, 9]]) == [[0.5, 0.5]]
    assert model.predict([[9, 9]]) == [0]


def test_decimal_constant_and_large_offset_roundtrip():
    model = GaussianNaiveBayes().fit([[0.1]] * 10, [0] * 3 + [1] * 7)
    state = model.to_state()
    assert state["variances_mle"] == [[0], [0]]
    assert state["global_variances_mle"] == [0]
    assert state["epsilon"] == VARIANCE_FLOOR
    assert GaussianNaiveBayes.from_state(state).predict_proba([[0.1]])[0] == pytest.approx([0.3, 0.7])
    model.fit([[1e16], [1e16 + 2], [1e16 + 4], [1e16 + 6]], [0, 0, 1, 1])
    assert GaussianNaiveBayes.from_state(model.to_state()).to_state() == model.to_state()


def test_stable_normalization_and_probability_properties():
    model = hand_model()
    probabilities = model.predict_proba([[1000], [-1000], [3]])
    for row in probabilities:
        assert sum(row) == pytest.approx(1, abs=1e-14)
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in row)
    assert probabilities[:2] == [[0.0, 1.0], [1.0, 0.0]]
    assert all(math.isfinite(value) for row in model.predict_log_proba([[1000]]) for value in row)


def test_row_order_invariance_and_fit_returns_self():
    model = hand_model()
    reversed_model = GaussianNaiveBayes(0.2, 0.01)
    assert reversed_model.fit([[6], [4], [2], [0]], [1, 1, 0, 0]) is reversed_model
    assert reversed_model.to_state() == model.to_state()


def test_json_roundtrip_and_state_copy_isolation():
    original = hand_model()
    state = json.loads(json.dumps(original.to_state()))
    restored = GaussianNaiveBayes.from_state(state)
    assert restored.to_state() == original.to_state()
    assert restored.predict_proba([[1], [3], [5]]) == original.predict_proba([[1], [3], [5]])
    assert restored.predict_log_proba([[1], [3]]) == original.predict_log_proba([[1], [3]])
    assert restored.predict([[1], [3], [5]]) == [0, 0, 1]
    state["means"][0][0] = 999
    exported = restored.to_state()
    exported["variances_effective"][0][0] = 999
    assert restored.to_state() == original.to_state()


@pytest.mark.parametrize("value", [0, -1, True, False, "1", None, float("nan"), float("inf"), 10 ** 1000])
@pytest.mark.parametrize("parameter", ["var_smoothing", "variance_floor"])
def test_invalid_parameters(parameter, value):
    with pytest.raises(ValueError):
        GaussianNaiveBayes(**{parameter: value})


@pytest.mark.parametrize("X,y", [
    ([], []), ([[], []], [0, 1]), ([[1], [2, 3]], [0, 1]),
    ([1, 2], [0, 1]), (None, [0, 1]), ("12", [0, 1]),
    ([[True], [2]], [0, 1]), ([["1"], [2]], [0, 1]),
    ([[None], [2]], [0, 1]), ([[float("nan")], [2]], [0, 1]),
    ([[1], [float("inf")]], [0, 1]), ([[1], [2]], [0]),
    ([[1], [2]], [0, 0]), ([[1], [2]], [1, 1]),
    ([[1], [2]], [False, 1]), ([[1], [2]], [0, True]),
    ([[1], [2]], [0, 2]), ([[1], [2]], [0, 0.5]),
    ([[1], [2]], [0, "1"]), ([[1], [2]], [0, float("nan")]),
    ([[1], [2]], None), ([[1], [2]], [[0], [1]]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        GaussianNaiveBayes().fit(X, y)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "predict_log_proba", "to_state"])
def test_before_fit(method):
    with pytest.raises(ValueError, match="chưa"):
        function = getattr(GaussianNaiveBayes(), method)
        function() if method == "to_state" else function([[1]])


@pytest.mark.parametrize("X", [[], [[]], [[1, 2]], [1], [[True]], [[float("inf")]], [["1"]]])
@pytest.mark.parametrize("method", ["predict", "predict_proba", "predict_log_proba"])
def test_invalid_prediction_data(X, method):
    with pytest.raises(ValueError):
        getattr(hand_model(), method)(X)


def test_overflow_is_an_explicit_error_and_failed_fit_is_atomic():
    model = hand_model()
    old_state = model.to_state()
    for rows in ([[1e308], [9e307]], [[-1e200], [1e200]]):
        with pytest.raises(ValueError, match="hữu hạn|tràn"):
            model.fit(rows, [0, 1])
    assert model.to_state() == old_state
    for method in (model.predict, model.predict_proba, model.predict_log_proba):
        with pytest.raises(ValueError, match="hữu hạn|tràn"):
            method([[1e308]])
    with pytest.raises(ValueError):
        GaussianNaiveBayes(var_smoothing=1e308).fit([[0], [4]], [0, 1])


@pytest.mark.parametrize("key,value", [
    ("version", True), ("version", 2), ("classes", [1, 0]), ("classes", [False, 1]),
    ("n_features", 0), ("n_features", True), ("n_features", 2),
    ("class_counts", [0, 2]), ("class_counts", [True, 2]), ("class_counts", [2.0, 2]),
    ("class_counts", [1, 1]),
    ("class_priors", [0.4, 0.6]), ("class_priors", [0, 1]),
    ("means", [[1]]), ("means", [[1], [float("nan")]]),
    ("variances_mle", [[-1], [1]]), ("variances_mle", [[True], [1]]),
    ("variances_effective", [[3], [3]]), ("variances_effective", [[0], [2]]),
    ("global_variances_mle", [-5]), ("global_variances_mle", [6]),
    ("global_variances_mle", [float("inf")]), ("epsilon", 2), ("epsilon", 0),
    ("var_smoothing", -1), ("variance_floor", 0),
])
def test_invalid_state_fields(key, value):
    state = hand_model().to_state()
    state[key] = value
    with pytest.raises(ValueError):
        GaussianNaiveBayes.from_state(state)


@pytest.mark.parametrize("state", [None, [], {}, {"version": 1}])
def test_invalid_state_structure(state):
    with pytest.raises(ValueError):
        GaussianNaiveBayes.from_state(state)


def test_missing_extra_state_keys_and_inconsistent_global_moments():
    state = hand_model().to_state()
    del state["epsilon"]
    with pytest.raises(ValueError):
        GaussianNaiveBayes.from_state(state)
    state = hand_model().to_state()
    state["extra"] = 1
    with pytest.raises(ValueError):
        GaussianNaiveBayes.from_state(state)
    state = hand_model().to_state()
    state["means"] = [[1], [7]]
    with pytest.raises(ValueError, match="moments"):
        GaussianNaiveBayes.from_state(state)


def test_numpy_array_inputs_without_statistical_helpers():
    numpy = pytest.importorskip("numpy")
    model = GaussianNaiveBayes(0.2, 0.01).fit(
        numpy.array([[0], [2], [4], [6]], dtype=float), numpy.array([0, 0, 1, 1])
    )
    assert model.predict(numpy.array([[1], [5]])) == [0, 1]
    for X, y in ((numpy.array([[False], [True]]), [0, 1]),
                 ([[0], [1]], numpy.array([False, True]))):
        with pytest.raises(ValueError):
            model.fit(X, y)
