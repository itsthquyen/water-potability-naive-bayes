"""Kiểm chứng thực nghiệm bằng dữ liệu tổng hợp và bảng điểm tính tay."""

from copy import deepcopy

import pytest

from src import experiments
from src.config import FEATURE_COLUMNS, SMOOTHING_VALUES, VARIANCE_FLOOR
from src.preprocessing import stratified_folds


def configuration(**changes):
    """Cấu hình nhỏ cho test, không dùng dataset thực nghiệm thật."""
    config = {"strategy": "mean", "standardize": False,
              "var_smoothing": 1e-9, "variance_floor": VARIANCE_FLOOR}
    config.update(changes)
    return config


def result(score, **changes):
    """Bảng điểm độc lập cho kiểm thử quy tắc chọn, không gọi CV làm oracle."""
    return {"config": configuration(**changes), "fold_metrics": [],
            "summary": {"balanced_accuracy": {"mean": score, "std": 0},
                        "macro_f1": {"mean": 0.4, "std": 0}}}


def sample_data():
    """Hai fold có train/val đủ hai lớp, dữ liệu tách lớp rõ ràng."""
    X = [[value] * len(FEATURE_COLUMNS) for value in (0, 1, 10, 11, 2, 3, 12, 13)]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    folds = [([4, 5, 6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]
    return X, y, folds


def test_real_cv_hand_expected_perfect_scores():
    X, y, folds = sample_data()
    outcome = experiments.cross_validate(X, y, folds, configuration())
    assert outcome["config"] == configuration()
    assert len(outcome["fold_metrics"]) == 2
    assert outcome["summary"] == {
        "balanced_accuracy": {"mean": 1, "std": 0},
        "macro_f1": {"mean": 1, "std": 0},
    }
    for metrics in outcome["fold_metrics"]:
        assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
        assert metrics["roc_auc"] == 1


def test_each_fold_fresh_pipeline_and_training_only_statistics(monkeypatch):
    X, y, folds = sample_data()
    X[0][0] = None
    created = []
    real_pipeline = experiments.WaterPipeline

    class SpyPipeline(real_pipeline):
        """Giữ instance và dữ liệu fit để phát hiện tái sử dụng hoặc leakage."""

        def __init__(self, **config):
            super().__init__(**config)
            created.append(self)

        def fit(self, rows, labels):
            self.fit_rows = deepcopy(rows)
            return super().fit(rows, labels)

    monkeypatch.setattr(experiments, "WaterPipeline", SpyPipeline)
    experiments.cross_validate(X, y, folds, configuration())
    assert len(created) == 2 and created[0] is not created[1]
    assert created[0].preprocessor is not created[1].preprocessor
    assert created[0].model is not created[1].model
    assert created[0].fit_rows == [X[index] for index in folds[0][0]]
    assert created[1].fit_rows == [X[index] for index in folds[1][0]]
    # Train fold 0: (2+3+12+13)/4=7.5; train fold 1 cột 0: (1+10+11)/3.
    assert created[0].preprocessor.to_state()["fill_values"][0] == 7.5
    assert created[1].preprocessor.to_state()["fill_values"][0] == pytest.approx(22 / 3)


def test_baseline_train_prior_tie_and_auc_half():
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    folds = [([4, 5, 6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]
    outcome = experiments.baseline_cross_validate(y, folds)
    assert outcome["config"] == {"baseline": "majority"}
    assert outcome["summary"]["balanced_accuracy"] == {"mean": 0.5, "std": 0}
    assert outcome["summary"]["macro_f1"] == {"mean": 0.2, "std": 0}
    assert outcome["fold_metrics"][0]["confusion_matrix"] == [[0, 3], [0, 1]]
    assert outcome["fold_metrics"][1]["confusion_matrix"] == [[1, 0], [3, 0]]
    for metric, prior in zip(outcome["fold_metrics"], [0.75, 0.25]):
        assert metric["roc_auc"] == 0.5
        assert metric["roc_curve"][1]["threshold"] == prior
    _, y, folds = sample_data()
    tied = experiments.baseline_cross_validate(y, folds)
    assert tied["fold_metrics"][0]["confusion_matrix"] == [[2, 0], [2, 0]]


def test_unweighted_summary_nonzero_population_std(monkeypatch):
    # Bảng độc lập: mean(.25,.75)=.5, std ddof0=.25; macro mean=.4/std=.2.
    table = iter([{"balanced_accuracy": 0.25, "macro_f1": 0.2},
                  {"balanced_accuracy": 0.75, "macro_f1": 0.6}])
    monkeypatch.setattr(experiments, "evaluate_predictions", lambda *args: next(table))
    _, y, folds = sample_data()
    summary = experiments.baseline_cross_validate(y, folds)["summary"]
    assert summary["balanced_accuracy"] == {"mean": 0.5, "std": 0.25}
    assert summary["macro_f1"] == pytest.approx({"mean": 0.4, "std": 0.2})


@pytest.mark.parametrize("preferred,other", [
    ({"standardize": False, "strategy": "median"}, {"standardize": True}),
    ({"strategy": "mean", "var_smoothing": 1e-12}, {"strategy": "median"}),
    ({"var_smoothing": 1e-9}, {"var_smoothing": 1e-8}),
    ({"var_smoothing": 1e-10}, {"var_smoothing": 1e-8}),
])
def test_selection_preference_order(preferred, other):
    first, second = result(0.795, **preferred), result(0.8, **other)
    assert experiments.select_configuration([second, first]) == first


def test_selection_boundary_unrounded_and_copy():
    boundary = result(0.795)
    best = result(0.8, standardize=True)
    assert experiments.select_configuration([best, boundary]) == boundary
    outside = result(0.794999999)
    assert experiments.select_configuration([best, outside]) == best
    selected = experiments.select_configuration([best, boundary])
    selected["config"]["strategy"] = "median"
    assert boundary["config"]["strategy"] == "mean"
    assert experiments.select_configuration([best, boundary], tolerance=0) == best


def test_default_grid_28_and_140_fresh_fits(monkeypatch):
    created = []

    class CountingPipeline:
        """Double xác định nhãn từ dữ liệu tổng hợp, không làm ML oracle."""

        def __init__(self, **config):
            created.append(self)
            self.fit_count = 0

        def fit(self, X, y):
            self.fit_count += 1
            return self

        def predict(self, X):
            return [int(row[0] > 5) for row in X]

        def predict_proba(self, X):
            return [[0, 1] if row[0] > 5 else [1, 0] for row in X]

    monkeypatch.setattr(experiments, "WaterPipeline", CountingPipeline)
    y = [0] * 5 + [1] * 5
    X = [[label * 10] * len(FEATURE_COLUMNS) for label in y]
    outcome = experiments.search_configurations(X, y, stratified_folds(y, 5))
    assert len(outcome["results"]) == 28
    assert len(created) == 140
    assert all(pipeline.fit_count == 1 for pipeline in created)
    combinations = {(entry["config"]["strategy"], entry["config"]["standardize"],
                     entry["config"]["var_smoothing"]) for entry in outcome["results"]}
    assert combinations == {(strategy, scale, smoothing) for strategy in ("mean", "median")
                            for scale in (False, True) for smoothing in SMOOTHING_VALUES}
    assert outcome["best"]["config"] == configuration()


@pytest.mark.parametrize("folds", [
    [], [([0, 1, 2, 3], [4, 5, 6, 7])],
    [([0, 1, 2, 3], [0, 5, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])],
    [([0, 1, 2, 3], [4, 4, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])],
    [([0, 1, 2, 3], [4, 5, 6, 8])],
    [([0, 1, 2, 3], [4, 5, 6, -1])],
    [([0, 1, 2, 3], [False, 5, 6, 7])],
    [([0, 1, 2, 3], [4.0, 5, 6, 7])],
    [([0, 1, 2], [4, 5, 6, 7])],
    [([], [0, 1, 2, 3, 4, 5, 6, 7])],
    [([0, 1], [2, 3, 4, 5, 6, 7])],
    [([0, 1, 2, 3], [4, 5, 6, 7])] * 2,
    [None], None,
])
def test_invalid_partitions(folds):
    X, y, _ = sample_data()
    with pytest.raises(ValueError):
        experiments.cross_validate(X, y, folds, configuration())
    with pytest.raises(ValueError):
        experiments.baseline_cross_validate(y, folds)


@pytest.mark.parametrize("y", [[], [0] * 8, [1] * 8, [False] + [1] * 7,
                                   [0] * 7 + [2], [0] * 7 + ["1"], None])
def test_invalid_labels(y):
    X, _, folds = sample_data()
    with pytest.raises(ValueError):
        experiments.cross_validate(X, y, folds, configuration())


@pytest.mark.parametrize("changes", [{"strategy": "bad"}, {"standardize": 1},
                                    {"var_smoothing": 0}, {"variance_floor": True},
                                    {"var_smoothing": float("nan")}])
def test_invalid_configuration(changes):
    X, y, folds = sample_data()
    with pytest.raises(ValueError):
        experiments.cross_validate(X, y, folds, configuration(**changes))


def test_missing_config_keys_length_and_empty_search():
    X, y, folds = sample_data()
    for config in ({}, None, {**configuration(), "extra": 1}):
        with pytest.raises(ValueError):
            experiments.cross_validate(X, y, folds, config)
    with pytest.raises(ValueError):
        experiments.cross_validate(X[:-1], y, folds, configuration())
    with pytest.raises(ValueError):
        experiments.search_configurations(X, y, folds, configs=[])


@pytest.mark.parametrize("stage", ["fit", "predict", "predict_proba", "metric"])
def test_fold_errors_propagate_without_skipping(monkeypatch, stage):
    X, y, folds = sample_data()

    def fail(*args, **kwargs):
        raise ValueError("lỗi fold cố ý")

    if stage == "metric":
        monkeypatch.setattr(experiments, "evaluate_predictions", fail)
    else:
        monkeypatch.setattr(experiments.WaterPipeline, stage, fail)
    with pytest.raises(ValueError, match="lỗi fold cố ý"):
        experiments.search_configurations(X, y, folds, [configuration()])


@pytest.mark.parametrize("tolerance", [-1, 2, True, float("nan"), "0.005"])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        experiments.select_configuration([result(0.8)], tolerance)


def test_invalid_results_and_summary():
    bad_mean = result(0.8)
    bad_mean["summary"]["balanced_accuracy"]["mean"] = float("nan")
    bad_std = result(0.8)
    bad_std["summary"]["macro_f1"]["std"] = -1
    for results in ([], None, [{}], [bad_mean], [bad_std]):
        with pytest.raises(ValueError):
            experiments.select_configuration(results)


def test_search_reuses_generator_inputs_safely():
    X, y, folds = sample_data()
    outcome = experiments.search_configurations(iter(X), iter(y), iter(folds),
                                               [configuration(), configuration(strategy="median")])
    assert len(outcome["results"]) == 2
    assert outcome["best"]["config"]["strategy"] == "mean"
