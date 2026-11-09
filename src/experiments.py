"""Kiểm định chéo Naive Bayes và baseline trên dữ liệu phát triển."""

from numbers import Integral, Real

from .config import MODEL_ALGORITHM, N_BINS
from .metrics import evaluate_predictions, summarize_scores
from .pipeline import WaterPipeline


def _sequence(values, name):
    """Sao chép dãy hữu hạn để dùng lặp lại; từ chối scalar/chuỗi/dict."""
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(f"{name} phải là dãy.")
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(f"{name} phải là dãy.") from error


def _validated_folds(y, folds):
    """Kiểm tra y nhị phân và phân hoạch CV hoàn chỉnh, trả labels và cặp chỉ số.

    Mỗi fold phải có train/validation không rỗng, đủ hai lớp, không lặp chỉ số,
    không giao nhau và hợp đúng toàn bộ y. Mỗi mẫu validation đúng một lần
    qua ít nhất hai fold. Không tự sửa hoặc bỏ fold lỗi; chỉ số bool bị cấm.
    """
    labels = []
    for label in _sequence(y, "y"):
        if isinstance(label, bool) or not isinstance(label, Real) or label not in (0, 1):
            raise ValueError("y chỉ nhận nhãn số 0/1, không nhận bool.")
        labels.append(int(label))
    if set(labels) != {0, 1}:
        raise ValueError("y phải không rỗng và đủ hai lớp.")
    validated = []
    validation_counts = [0] * len(labels)
    for fold in _sequence(folds, "folds"):
        parts = _sequence(fold, "fold")
        if len(parts) != 2:
            raise ValueError("Mỗi fold phải là cặp (train_indices, val_indices).")
        index_groups = []
        for part in parts:
            indices = _sequence(part, "Chỉ số fold")
            if not indices:
                raise ValueError("Train/validation không được rỗng.")
            for index in indices:
                if (isinstance(index, bool) or not isinstance(index, Integral)
                        or index < 0 or index >= len(labels)):
                    raise ValueError("Chỉ số fold phải là số nguyên trong giới hạn y.")
            indices = [int(index) for index in indices]
            if len(set(indices)) != len(indices):
                raise ValueError("Chỉ số bị lặp trong fold.")
            if {labels[index] for index in indices} != {0, 1}:
                raise ValueError("Mỗi phần train/validation phải đủ hai lớp.")
            index_groups.append(indices)
        train, validation = index_groups
        if set(train) & set(validation) or len(train) + len(validation) != len(labels):
            raise ValueError("Train/validation phải phân hoạch toàn bộ y, không giao nhau.")
        for index in validation:
            validation_counts[index] += 1
        validated.append((train, validation))
    if len(validated) < 2 or any(count != 1 for count in validation_counts):
        raise ValueError("Cần ít nhất hai fold, mỗi mẫu validation đúng một lần.")
    return labels, validated


def _result(config, fold_metrics):
    """Ghép kết quả và mean/std ddof=0 không trọng số; không bỏ điểm lỗi."""
    return {"config": config, "fold_metrics": fold_metrics,
            "summary": {name: summarize_scores([fold[name] for fold in fold_metrics])
                        for name in ("balanced_accuracy", "macro_f1")}}


def cross_validate(X, y, folds):
    """Đánh giá cấu hình cố định trên các cặp chỉ số; trả config/fold_metrics/summary.

    X (n, 9) có thể thiếu, y (n,) nhị phân. Mỗi fold tạo WaterPipeline mới và
    chỉ fit phần train. fold_metrics là list dict từ evaluate_predictions;
    summary chứa balanced_accuracy/macro_f1, mỗi mục {mean, std}. Lỗi dữ liệu,
    tiền xử lý, mô hình hay metric được truyền ra, không âm thầm bỏ fold.
    """
    configuration = {"algorithm": MODEL_ALGORITHM, "max_bins": N_BINS, "missing": "median", "smoothing": False}
    labels, partitions = _validated_folds(y, folds)
    rows = _sequence(X, "X")
    if len(rows) != len(labels):
        raise ValueError("X và y phải cùng số mẫu.")
    fold_metrics = []
    for train, validation in partitions:
        pipeline = WaterPipeline()
        pipeline.fit([rows[index] for index in train], [labels[index] for index in train])
        validation_X = [rows[index] for index in validation]
        predictions, probabilities = pipeline.predict_with_proba(validation_X)
        fold_metrics.append(evaluate_predictions(
            [labels[index] for index in validation], predictions,
            [row[1] for row in probabilities],
        ))
    return _result(configuration, fold_metrics)


def baseline_cross_validate(y, folds):
    """CV baseline đa số train, hòa chọn 0; điểm lớp 1 là prior train hằng.

    Trả cùng cấu trúc cross_validate, config={"baseline": "majority"}. Không
    dùng nhãn validation để học prior; ROC-AUC mỗi fold đủ hai lớp bằng 0.5.
    """
    labels, partitions = _validated_folds(y, folds)
    fold_metrics = []
    for train, validation in partitions:
        positive_count = 0
        for index in train:
            positive_count += labels[index]
        prior_one = positive_count / len(train)
        majority = 1 if positive_count > len(train) - positive_count else 0
        fold_metrics.append(evaluate_predictions(
            [labels[index] for index in validation], [majority] * len(validation),
            [prior_one] * len(validation),
        ))
    return _result({"baseline": "majority"}, fold_metrics)
