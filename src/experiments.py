"""Vòng lặp CV và chọn cấu hình tự cài, chỉ dùng dữ liệu phát triển."""

from copy import deepcopy
import math
from numbers import Integral, Real

from .config import SMOOTHING_VALUES, VARIANCE_FLOOR
from .metrics import evaluate_predictions, summarize_scores
from .pipeline import WaterPipeline


def _number(value, name):
    """Nhận số thực hữu hạn, không ép bool/chuỗi; sai kiểu gây ValueError."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} phải là số thực hữu hạn.")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} vượt giới hạn số học.") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} phải hữu hạn.")
    return value


def _configuration(config):
    """Chuẩn hóa bốn khóa cấu hình, không bổ sung mặc định ngầm khi thiếu khóa."""
    keys = {"strategy", "standardize", "var_smoothing", "variance_floor"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("Cấu hình phải có đúng strategy/standardize/var_smoothing/variance_floor.")
    if config["strategy"] not in ("mean", "median") or type(config["standardize"]) is not bool:
        raise ValueError("strategy phải mean/median, standardize phải bool.")
    result = dict(config)
    for key in ("var_smoothing", "variance_floor"):
        result[key] = _number(config[key], key)
        if result[key] <= 0:
            raise ValueError(f"{key} phải dương.")
    return result


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


def cross_validate(X, y, folds, config):
    """Đánh giá cấu hình trên các cặp chỉ số; trả config/fold_metrics/summary.

    X (n, 9) có thể thiếu, y (n,) nhị phân. Mỗi fold tạo WaterPipeline mới và
    chỉ fit phần train. fold_metrics là list dict từ evaluate_predictions;
    summary chứa balanced_accuracy/macro_f1, mỗi mục {mean, std}. Lỗi dữ liệu,
    tiền xử lý, mô hình hay metric được truyền ra, không âm thầm bỏ fold.
    """
    configuration = _configuration(config)
    labels, partitions = _validated_folds(y, folds)
    rows = _sequence(X, "X")
    if len(rows) != len(labels):
        raise ValueError("X và y phải cùng số mẫu.")
    fold_metrics = []
    for train, validation in partitions:
        pipeline = WaterPipeline(**configuration)
        pipeline.fit([rows[index] for index in train], [labels[index] for index in train])
        validation_X = [rows[index] for index in validation]
        predictions = pipeline.predict(validation_X)
        probabilities = pipeline.predict_proba(validation_X)
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


def select_configuration(results, tolerance=0.005):
    """Chọn bản sao result hợp lệ trong best BA - tolerance, không làm tròn.

    Ưu tiên không chuẩn hóa, mean, |log10(smoothing)+9| nhỏ nhất rồi smoothing
    nhỏ hơn. Nếu mọi khóa ưu tiên bằng nhau, giữ thứ tự đầu vào. tolerance
    hữu hạn trong [0,1]; thiếu/điểm mean/std sai khiến ValueError, không bỏ qua.
    """
    tolerance = _number(tolerance, "tolerance")
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance phải thuộc [0, 1].")
    candidates = _sequence(results, "results")
    if not candidates:
        raise ValueError("results không được rỗng.")
    checked = []
    for result in candidates:
        if not isinstance(result, dict) or "config" not in result or "summary" not in result:
            raise ValueError("Result thiếu config/summary.")
        configuration = _configuration(result["config"])
        summary = result["summary"]
        if not isinstance(summary, dict):
            raise ValueError("summary phải là dict.")
        for name in ("balanced_accuracy", "macro_f1"):
            statistics = summary.get(name)
            if not isinstance(statistics, dict) or set(statistics) != {"mean", "std"}:
                raise ValueError("Summary phải có mean/std cho BA và macro-F1.")
            for key in ("mean", "std"):
                value = _number(statistics[key], f"{name}.{key}")
                if not 0 <= value <= 1:
                    raise ValueError("Điểm và độ lệch chuẩn phải thuộc [0,1].")
        checked.append((result, configuration, float(summary["balanced_accuracy"]["mean"])))
    best_score = max(item[2] for item in checked)
    # So sánh với biên best - tolerance để biên .8 - .005 = .795 được nhận.
    eligible = [item for item in checked if item[2] >= best_score - tolerance]
    selected = min(eligible, key=lambda item: (
        item[1]["standardize"], item[1]["strategy"] != "mean",
        abs(math.log10(item[1]["var_smoothing"]) + 9), item[1]["var_smoothing"],
    ))
    return deepcopy(selected[0])


def search_configurations(X, y, folds, configs=None):
    """Duyệt tuần tự cấu hình, trả {results, best}; lỗi bất kỳ dừng toàn lượt.

    configs=None dùng 2 strategy x 2 standardize x SMOOTHING_VALUES (28 cấu
    hình), cùng VARIANCE_FLOOR. Không fit mô hình cuối, không đọc dữ liệu test.
    Dãy đầu vào được giữ lại một lần để không cạn generator giữa cấu hình.
    """
    if configs is None:
        configs = [{"strategy": strategy, "standardize": standardize,
                    "var_smoothing": smoothing, "variance_floor": VARIANCE_FLOOR}
                   for strategy in ("mean", "median") for standardize in (False, True)
                   for smoothing in SMOOTHING_VALUES]
    configurations = [_configuration(config) for config in _sequence(configs, "configs")]
    if not configurations:
        raise ValueError("configs không được rỗng.")
    labels, partitions = _validated_folds(y, folds)
    rows = _sequence(X, "X")
    results = [cross_validate(rows, labels, partitions, config) for config in configurations]
    return {"results": results, "best": select_configuration(results)}
