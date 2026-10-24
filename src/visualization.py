"""Dựng hình từ thống kê tự cài; không đọc CSV, fit hoặc chọn tập dữ liệu."""

import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import matplotlib.pyplot as plt

from .config import CLASS_LABELS, FEATURE_COLUMNS
from .eda import histogram_counts, pearson_correlation, summarize_features


def _save_figure(figure, path):
    """Lưu PNG tại đường dẫn caller chọn và luôn đóng figure để tránh rò bộ nhớ."""
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=120)
    finally:
        plt.close(figure)
    return path


def _write_manifest(path, statistics, figures):
    """Ghi thống kê hữu hạn JSON UTF-8 và danh sách tên PNG cùng thư mục."""
    payload = {"statistics": statistics, "figures": [figure.name for figure in figures]}
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
                    encoding="utf-8")


def create_eda_figures(X, y, output_dir):
    """Nhận X (n,9) theo FEATURE_COLUMNS, y nhãn 0/1; trả list Path của 13 PNG.

    Caller phải chọn tập phát triển nếu hình phục vụ lựa chọn phương pháp.
    Không đọc CSV/train. Biên 10 bin được tính một lần mỗi cột trên X, dùng
    chung hai lớp. Boxplot dùng bxp với phân vị/râu do src.eda tính. Pearson
    dùng cặp không thiếu, chú thích r và số cặp; N/A khi không xác định.
    eda_manifest.json lưu raw statistics và tên hình, không nằm trong list
    Path trả về. Tạo output_dir nếu cần; cùng tên file sẽ được ghi đè.
    Rỗng, sai nhãn/shape hoặc số không hợp lệ gây ValueError trước khi vẽ.
    """
    rows = [list(row) for row in X]
    labels = list(y)
    if not rows or len(rows) != len(labels):
        raise ValueError("EDA cần dữ liệu không rỗng và X/y cùng số mẫu.")
    if any(isinstance(label, bool) or not isinstance(label, Real)
           or label not in CLASS_LABELS for label in labels):
        raise ValueError("Nhãn EDA phải là số 0 hoặc 1.")
    summary = summarize_features(rows, FEATURE_COLUMNS)
    class_rows = {label: [row for row, actual in zip(rows, labels) if actual == label]
                  for label in CLASS_LABELS}
    per_class = {str(label): summarize_features(class_rows[label], FEATURE_COLUMNS)
                 for label in CLASS_LABELS}
    class_counts = {str(label): len(class_rows[label]) for label in CLASS_LABELS}
    histograms = {}
    columns = [[row[index] for row in rows] for index in range(len(FEATURE_COLUMNS))]
    for index, name in enumerate(FEATURE_COLUMNS):
        histogram = histogram_counts(columns[index])
        histograms[name] = {"edges": histogram["edges"], "per_class": {}}
        for label in CLASS_LABELS:
            histogram_by_class = histogram_counts(
                [row[index] for row in class_rows[label]], bins=histogram["edges"])
            histograms[name]["per_class"][str(label)] = histogram_by_class["counts"]
    correlations = []
    pair_counts = []
    for left_column in columns:
        correlation_row = []
        count_row = []
        for right_column in columns:
            correlation, count = pearson_correlation(left_column, right_column)
            correlation_row.append(correlation)
            count_row.append(count)
        correlations.append(correlation_row)
        pair_counts.append(count_row)
    statistics = {"feature_names": list(FEATURE_COLUMNS), "sample_count": len(rows),
                  "class_counts": class_counts, "features": summary,
                  "per_class": per_class, "histograms": histograms,
                  "pearson": {"values": correlations, "pair_counts": pair_counts}}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([str(label) for label in CLASS_LABELS],
             [class_counts[str(label)] for label in CLASS_LABELS])
    axis.set(title="Số mẫu theo lớp", xlabel="Nhãn Potability", ylabel="Số mẫu")
    paths.append(_save_figure(figure, destination / "class_counts.png"))

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(FEATURE_COLUMNS, [summary[name]["missing"] for name in FEATURE_COLUMNS])
    axis.set(title="Số ô thiếu theo đặc trưng", ylabel="Số ô thiếu")
    axis.tick_params(axis="x", labelrotation=45)
    paths.append(_save_figure(figure, destination / "missing_counts.png"))

    for name in FEATURE_COLUMNS:
        figure, axis = plt.subplots(figsize=(7, 4))
        edges = histograms[name]["edges"]
        widths = [edges[index + 1] - edges[index] for index in range(len(edges) - 1)]
        for label in CLASS_LABELS:
            axis.bar(edges[:-1], histograms[name]["per_class"][str(label)],
                     width=widths, align="edge", alpha=0.5, label=f"Lớp {label}")
        axis.set(title=f"Phân bố {name} theo lớp", xlabel=name, ylabel="Số mẫu")
        axis.legend()
        paths.append(_save_figure(figure, destination / f"histogram_{name}.png"))

    figure, axes = plt.subplots(3, 3, figsize=(13, 11))
    for index, name in enumerate(FEATURE_COLUMNS):
        axis = axes[index // 3][index % 3]
        boxes = []
        positions = []
        for position, label in enumerate(CLASS_LABELS, start=1):
            feature = per_class[str(label)][name]
            if feature["count"]:
                boxes.append({"label": f"Lớp {label}", "med": feature["median"],
                              "q1": feature["quartiles"]["q1"],
                              "q3": feature["quartiles"]["q3"],
                              "whislo": feature["boxplot"]["whisker_low"],
                              "whishi": feature["boxplot"]["whisker_high"],
                              "fliers": feature["boxplot"]["outliers"]})
                positions.append(position)
        if boxes:
            axis.bxp(boxes, positions=positions, showfliers=True)
        else:
            axis.text(0.5, 0.5, "Không có quan sát", ha="center", transform=axis.transAxes)
        axis.set_xticks([1, 2], ["Lớp 0", "Lớp 1"])
        axis.set_title(name)
    figure.suptitle("Boxplot theo lớp; giữ nguyên các điểm ngoài hàng rào")
    paths.append(_save_figure(figure, destination / "boxplots.png"))

    figure, axis = plt.subplots(figsize=(12, 10))
    # NaN chỉ dùng để che ô ảnh N/A; manifest vẫn lưu None theo JSON chuẩn.
    image_values = [[float("nan") if value is None else value for value in row]
                    for row in correlations]
    image = axis.imshow(image_values, vmin=-1, vmax=1, cmap="coolwarm")
    figure.colorbar(image, ax=axis, label="Tương quan Pearson")
    axis.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    axis.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    axis.set_title("Pearson theo cặp không thiếu; n là số cặp")
    for row_index, row in enumerate(correlations):
        for column_index, value in enumerate(row):
            text = "N/A" if value is None else f"{value:.2f}"
            axis.text(column_index, row_index,
                      f"{text}\nn={pair_counts[row_index][column_index]}",
                      ha="center", va="center", fontsize=7)
    paths.append(_save_figure(figure, destination / "pearson.png"))
    _write_manifest(destination / "eda_manifest.json", statistics, paths)
    return paths


def create_evaluation_figures(metrics, output_dir):
    """Vẽ confusion matrix và ROC có sẵn; trả list Path PNG, không tính metric.

    metrics dùng schema evaluate_predictions: ma trận 2x2 hàng thật/cột dự
    đoán, roc_curve tùy chọn list {fpr,tpr,threshold}. Bỏ hình ROC nếu thiếu
    hoặc rỗng, không tự suy diễn điểm. evaluation_manifest.json lưu nguyên
    metrics JSON-safe cùng tên hình. Sai ma trận/điểm gây ValueError trước
    khi xuất; output_dir caller chọn, được tạo và ghi đè tên file cố định.
    """
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 2:
        raise ValueError("Confusion matrix phải có shape (2,2).")
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("Confusion matrix phải có shape (2,2).")
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
               for value in row):
            raise ValueError("Số đếm confusion matrix phải nguyên không âm.")
    points = metrics.get("roc_curve")
    if points is not None:
        if not isinstance(points, list):
            raise ValueError("roc_curve phải là list hoặc None.")
        previous = (0.0, 0.0)
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Mỗi điểm ROC phải là dict.")
            coordinates = (point.get("fpr"), point.get("tpr"))
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError("FPR/TPR phải hữu hạn trong [0,1].")
            if any(current < old for current, old in zip(coordinates, previous)):
                raise ValueError("Các điểm ROC phải tăng không giảm.")
            previous = coordinates
    try:
        json.dumps(metrics, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metrics phải là dữ liệu JSON-safe.") from error
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, label="Số mẫu")
    axis.set_xticks([0, 1], [str(label) for label in CLASS_LABELS])
    axis.set_yticks([0, 1], [str(label) for label in CLASS_LABELS])
    axis.set(title="Ma trận nhầm lẫn", xlabel="Nhãn dự đoán", ylabel="Nhãn thật")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    paths.append(_save_figure(figure, destination / "confusion_matrix.png"))
    if points:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([point["fpr"] for point in points], [point["tpr"] for point in points],
                  marker="o", label="ROC từ điểm đã tính")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Đường tham chiếu")
        axis.set(title="Đường ROC lớp 1", xlabel="Tỷ lệ dương giả (FPR)",
                 ylabel="Tỷ lệ dương thật (TPR)", xlim=(0, 1), ylim=(0, 1))
        axis.legend()
        paths.append(_save_figure(figure, destination / "roc.png"))
    _write_manifest(destination / "evaluation_manifest.json", metrics, paths)
    return paths
