"""Xuất ARFF và số liệu đã kiểm chứng cho Weka; không fit hoặc chọn lại mô hình."""

import argparse
import csv
import json
import math
from pathlib import Path

from .config import FEATURE_COLUMNS, PROJECT_ROOT, RAW_DATA_PATH, TARGET_COLUMN
from .data_processing.data_validation import clean_training_data, read_csv_records, validate_records
from .evaluate import _confined_path
from .metrics import evaluate_predictions
from .pipeline import load_pipeline, sha256_file
from .train import validate_run_id


def write_arff(path, relation, attributes, rows):
    """Ghi ARFF UTF-8, với None là '?'; tên thuộc tính/quan hệ đến từ mã nội bộ."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"@relation {relation}\n\n")
        for name, kind in attributes:
            stream.write(f"@attribute {name} {kind}\n")
        stream.write("\n@data\n")
        for row in rows:
            stream.write(",".join("?" if value is None else str(value) for value in row) + "\n")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 không khớp: {path.name}")


def _data_rows(X, y):
    return ([*features, label] for features, label in zip(X, y))


def export_weka(run_id, output_dir):
    """Xuất dữ liệu thô, dev/test trước và sau transform, ROC và lỗi đã lưu.

    ARFF gốc có 9 số đo, bản đã xử lý có 9 thuộc tính nominal là mã khoảng.
    Nhãn luôn nominal {0,1}; file lỗi chứa dự đoán, không dùng để huấn luyện.
    """
    validate_run_id(run_id)
    manifest = _read_json(PROJECT_ROOT / "results" / "runs" / f"{run_id}.json")
    if manifest["run_id"] != run_id or manifest["stage"] != "evaluated":
        raise ValueError("Chỉ xuất run đã evaluated và đúng ID.")
    for relative_path, checksum in manifest["files"].items():
        _check_hash(_confined_path(PROJECT_ROOT, relative_path), checksum)

    data_manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    _check_hash(RAW_DATA_PATH, manifest["csv_sha256"])
    _check_hash(data_manifest_path, manifest["data_manifest_sha256"])
    data_manifest = _read_json(data_manifest_path)
    model_path = _confined_path(PROJECT_ROOT, manifest["model_path"])
    pipeline = load_pipeline(model_path, manifest["model_sha256"])
    if pipeline.run_id != run_id:
        raise ValueError("run_id của model không khớp.")

    raw = validate_records(read_csv_records(RAW_DATA_PATH, training=True), training=True)
    if raw["errors"]:
        raise ValueError("CSV gốc có dòng không hợp lệ; không xuất dữ liệu một phần.")
    cleaned = clean_training_data(raw["X"], raw["y"], raw["row_ids"])
    clean_by_id = dict(zip(cleaned["row_ids"], zip(cleaned["X"], cleaned["y"])))
    partitions = {}
    for name in ("development", "test"):
        split = PROJECT_ROOT / "data" / "splits" / f"{name}.txt"
        _check_hash(split, data_manifest["split_checksums"][name])
        row_ids = [int(line) for line in split.read_text(encoding="utf-8").splitlines()]
        if (not row_ids or len(row_ids) != len(set(row_ids))
                or any(row_id not in clean_by_id for row_id in row_ids)):
            raise ValueError(f"ID trong tập {name} không hợp lệ.")
        X, y = zip(*(clean_by_id[row_id] for row_id in row_ids))
        partitions[name] = (row_ids, X, y)
    dev_ids = set(partitions["development"][0])
    test_ids, test_X, test_y = partitions["test"]
    if dev_ids.intersection(test_ids) or dev_ids.union(test_ids) != set(clean_by_id):
        raise ValueError("Hai tập phải không giao nhau và phủ hết dữ liệu đã làm sạch.")

    metrics_dir = PROJECT_ROOT / "results" / "metrics" / run_id
    metrics = _read_json(metrics_dir / "test_metrics.json")
    with (metrics_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != len(test_ids):
        raise ValueError("Số dự đoán khác số mẫu test.")
    predicted, probabilities, error_rows = [], [], []
    for row_id, features, actual, row in zip(test_ids, test_X, test_y, predictions):
        if int(row["row_id"]) != row_id or int(row["actual"]) != actual or row["run_id"] != run_id:
            raise ValueError("Thứ tự ID, nhãn thật hoặc run_id của dự đoán không khớp.")
        label = int(row["predicted"])
        probability_0, score = float(row["probability_0"]), float(row["probability_1"])
        if (label not in (0, 1)
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in (probability_0, score))
                or not math.isclose(probability_0 + score, 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Nhãn dự đoán hoặc xác suất không hợp lệ.")
        if (score > probability_0 and label != 1) or (score < probability_0 and label != 0):
            raise ValueError("Nhãn dự đoán không khớp thứ tự xác suất đã lưu.")
        predicted.append(label)
        probabilities.append(score)
        error_type = ("TP" if label else "FN") if actual else ("FP" if label else "TN")
        error_rows.append((score, features[0], error_type))

    # Kiểm lại từ dự đoán đã lưu bằng chính luồng metric tự cài; không suy luận lại.
    recomputed = evaluate_predictions(test_y, predicted, probabilities)
    if any(recomputed[key] != metrics[key] for key in
           ("confusion_matrix", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "roc_curve")):
        raise ValueError("Chỉ số tính lại từ dự đoán không khớp kết quả run.")

    output_dir.mkdir(parents=True, exist_ok=True)
    attributes = [(name, "numeric") for name in FEATURE_COLUMNS] + [(TARGET_COLUMN, "{0,1}")]
    processed_attributes = [
        (name, "{" + ",".join(str(category) for category in range(size)) + "}")
        for name, size in zip(FEATURE_COLUMNS, pipeline.preprocessor.category_sizes)
    ] + [(TARGET_COLUMN, "{0,1}")]
    write_arff(output_dir / "raw.arff", "water_raw", attributes, _data_rows(raw["X"], raw["y"]))
    for name, (_, X, y) in partitions.items():
        write_arff(output_dir / f"{name}_raw.arff", f"water_{name}_raw", attributes, _data_rows(X, y))
        transformed = pipeline.preprocessor.transform(X)
        write_arff(output_dir / f"{name}_processed.arff", f"water_{name}_processed",
                   processed_attributes, _data_rows(transformed, y))
    write_arff(output_dir / "roc.arff", "python_nb_roc",
               [("False_Positive_Rate", "numeric"), ("True_Positive_Rate", "numeric")],
               ((point["fpr"], point["tpr"]) for point in metrics["roc_curve"]))
    write_arff(output_dir / "errors.arff", "python_nb_test_errors",
               [("P_Potability_1", "numeric"), ("ph_raw", "numeric"), ("Outcome", "{TN,FP,FN,TP}")],
               error_rows)
    print(f"{run_id}: {len(dev_ids)} dev, {len(test_ids)} test; ARFF: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    export_weka(arguments.run_id, arguments.out)


if __name__ == "__main__":
    main()
