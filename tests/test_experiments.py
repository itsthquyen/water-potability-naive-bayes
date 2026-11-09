"""Kiểm chứng thực nghiệm bằng dữ liệu tổng hợp và bảng điểm tính tay."""

from copy import deepcopy

import pytest

from src import experiments
from src.config import FEATURE_COLUMNS, MODEL_ALGORITHM, N_BINS
from src.preprocessing import stratified_folds


def sample_data():
    """Hai fold có train/val đủ hai lớp, dữ liệu tách lớp rõ ràng."""
    X = [[value] * len(FEATURE_COLUMNS) for value in (0, 1, 10, 11, 2, 3, 12, 13)]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    folds = [([4, 5, 6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]
    return X, y, folds


def test_real_cv_hand_expected_perfect_scores():
    X, y, folds = sample_data()
    outcome = experiments.cross_validate(X, y, folds)
    assert outcome["config"] == {
        "algorithm": MODEL_ALGORITHM, "max_bins": N_BINS, "missing": "median", "smoothing": False,
    }
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
    experiments.cross_validate(X, y, folds)
    assert len(created) == 2 and created[0] is not created[1]
    assert created[0].preprocessor is not created[1].preprocessor
    assert created[0].model is not created[1].model
    assert created[0].fit_rows == [X[index] for index in folds[0][0]]
    assert created[1].fit_rows == [X[index] for index in folds[1][0]]
    # Ngưỡng/median chỉ đọc giá trị quan sát ở fold train, bỏ ô thiếu.
    assert created[0].preprocessor.to_state()["cut_points"][0] == [7.5, 12.5]
    assert created[1].preprocessor.to_state()["cut_points"][0] == [5.5, 10.5]
    assert created[0].preprocessor.to_state()["fill_values"][0] == 7.5
    assert created[1].preprocessor.to_state()["fill_values"][0] == 10


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
        experiments.cross_validate(X, y, folds)
    with pytest.raises(ValueError):
        experiments.baseline_cross_validate(y, folds)


@pytest.mark.parametrize("y", [[], [0] * 8, [1] * 8, [False] + [1] * 7,
                                   [0] * 7 + [2], [0] * 7 + ["1"], None])
def test_invalid_labels(y):
    X, _, folds = sample_data()
    with pytest.raises(ValueError):
        experiments.cross_validate(X, y, folds)


def test_feature_count_must_match_labels():
    X, y, folds = sample_data()
    with pytest.raises(ValueError, match="cùng số mẫu"):
        experiments.cross_validate(X[:-1], y, folds)


@pytest.mark.parametrize("stage", ["fit", "predict_with_proba", "metric"])
def test_fold_errors_propagate_without_skipping(monkeypatch, stage):
    X, y, folds = sample_data()

    def fail(*args, **kwargs):
        raise ValueError("lỗi fold cố ý")

    if stage == "metric":
        monkeypatch.setattr(experiments, "evaluate_predictions", fail)
    else:
        monkeypatch.setattr(experiments.WaterPipeline, stage, fail)
    with pytest.raises(ValueError, match="lỗi fold cố ý"):
        experiments.cross_validate(X, y, folds)
