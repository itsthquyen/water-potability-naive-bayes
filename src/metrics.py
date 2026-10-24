"""Chỉ số nhị phân tự cài; hàng/cột ma trận luôn theo lớp [0, 1]."""

from math import isfinite, sqrt
from numbers import Real


def _validate_labels(values, name):
    """Đổi dãy nhãn không rỗng thành int; nhãn ngoài 0/1 gây ValueError."""
    labels = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or value not in (0, 1):
            raise ValueError(f"{name} chỉ nhận nhãn số 0 hoặc 1.")
        labels.append(int(value))
    if not labels:
        raise ValueError(f"{name} không được rỗng.")
    return labels


def _validate_scores(values, expected_length=None):
    """Nhận điểm thực hữu hạn, không ép chuỗi/bool; kiểm tra độ dài nếu có."""
    scores = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Điểm phải là số thực hữu hạn.")
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("Điểm vượt giới hạn số thực hữu hạn.") from error
        if not isfinite(number):
            raise ValueError("Điểm phải là số thực hữu hạn.")
        scores.append(number)
    if not scores or (expected_length is not None and len(scores) != expected_length):
        raise ValueError("Dãy điểm rỗng hoặc không khớp số nhãn.")
    return scores


def _calculate_roc(labels, scores):
    """Gom hòa điểm giảm dần; cộng hình thang ROC, một lớp trả (None, [])."""
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None, []
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    points = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]
    true_positive = 0
    false_positive = 0
    area = 0.0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        false_positive_rate = false_positive / negative_count
        true_positive_rate = true_positive / positive_count
        previous = points[-1]
        area += (false_positive_rate - previous["fpr"]) * (
            true_positive_rate + previous["tpr"]
        ) / 2
        points.append({"threshold": threshold, "fpr": false_positive_rate,
                       "tpr": true_positive_rate})
    return area, points


def roc_auc_score(y_true, scores):
    """Trả diện tích ROC cho điểm lớp 1 (không cần trong [0,1]); một lớp: None.

    Hai dãy một chiều phải cùng độ dài dương; nhãn 0/1, điểm hữu hạn.
    ValueError khi sai dữ liệu. Điểm hòa được chuyển ngưỡng cùng một nhóm.
    """
    labels = _validate_labels(y_true, "y_true")
    validated_scores = _validate_scores(scores, len(labels))
    area, _ = _calculate_roc(labels, validated_scores)
    return area


def evaluate_predictions(y_true, y_pred, scores=None):
    """Đánh giá hai dãy nhãn 0/1 cùng độ dài dương, trả dict JSON-safe.

    confusion_matrix = [[TN, FP], [FN, TP]], per_class theo [0,1]. Mỗi lớp
    có label/support/precision/recall/f1/undefined (dict cờ cho ba chỉ số).
    Mẫu số 0 hiển thị None; balanced_accuracy và macro_f1 vẫn lấy trung bình
    đúng hai lớp, thay chỉ số không xác định bằng 0. F1 dùng 2TP/(2TP+FP+FN).
    scores tùy chọn là điểm hữu hạn lớp 1: roc_curve gồm threshold/fpr/tpr,
    threshold gốc None thay +inf để JSON hợp lệ. Không có scores: hai trường
    ROC là None; chỉ một lớp thật: roc_auc None và roc_curve rỗng.
    Nhãn sai, dãy rỗng, lệch độ dài hoặc điểm không hữu hạn gây ValueError.
    """
    actual = _validate_labels(y_true, "y_true")
    predicted = _validate_labels(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError("Số nhãn thật và dự đoán phải bằng nhau.")
    matrix = [[0, 0], [0, 0]]
    for true_label, predicted_label in zip(actual, predicted):
        matrix[true_label][predicted_label] += 1
    per_class = []
    recall_total = 0.0
    f1_total = 0.0
    for label in (0, 1):
        true_positive = matrix[label][label]
        false_positive = matrix[1 - label][label]
        false_negative = matrix[label][1 - label]
        support = true_positive + false_negative
        denominators = {"precision": true_positive + false_positive,
                        "recall": support,
                        "f1": 2 * true_positive + false_positive + false_negative}
        row = {"label": label, "support": support, "undefined": {}}
        for name, denominator in denominators.items():
            numerator = 2 * true_positive if name == "f1" else true_positive
            row[name] = numerator / denominator if denominator else None
            row["undefined"][name] = denominator == 0
        recall_total += row["recall"] if row["recall"] is not None else 0.0
        f1_total += row["f1"] if row["f1"] is not None else 0.0
        per_class.append(row)
    area, points = None, None
    if scores is not None:
        validated_scores = _validate_scores(scores, len(actual))
        area, points = _calculate_roc(actual, validated_scores)
    return {"accuracy": (matrix[0][0] + matrix[1][1]) / len(actual),
            "balanced_accuracy": recall_total / 2,
            "macro_f1": f1_total / 2, "confusion_matrix": matrix,
            "per_class": per_class, "roc_auc": area, "roc_curve": points}


def summarize_scores(values):
    """Tổng hợp dãy điểm hữu hạn không rỗng thành {mean, std}, std ddof=0.

    Không bỏ fold lỗi/None/NaN. Sai dữ liệu hoặc tràn số gây ValueError.
    Trung bình cộng không trọng số; phương sai là tổng lệch bình phương/n.
    """
    scores = _validate_scores(values)
    count = len(scores)
    mean = sum(value / count for value in scores)
    squared_deviations = 0.0
    for value in scores:
        deviation = value - mean
        squared_deviations += deviation * deviation / count
    if not isfinite(mean) or not isfinite(squared_deviations):
        raise ValueError("Thống kê điểm vượt giới hạn số học hữu hạn.")
    return {"mean": mean, "std": sqrt(squared_deviations)}
