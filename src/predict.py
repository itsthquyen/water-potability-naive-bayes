"""Dự đoán dùng chung cho CLI, nhập tay và CSV; giữ lỗi theo dòng."""

import argparse
import csv
import io
from pathlib import Path

from src.config import FEATURE_COLUMNS, MISSING_WARNING_COUNT
from src.data_validation import read_csv_records, validate_records


def predict_records(records, pipeline, run_id=""):
    """Trả một kết quả cho mỗi bản ghi; dòng lỗi không có nhãn/xác suất.

    records là list dict theo tên cột; pipeline đã fit. Cảnh báo chỉ mô tả
    đầu vào, không phải kết luận an toàn nước. Không đọc/nạp model mỗi dòng.
    """
    validation = validate_records(records, training=False)
    results = []
    for row_id, record in enumerate(records):
        results.append({
            "input_row_id": row_id, "sample_id": record.get("sample_id", ""),
            "predicted_label": None, "probability_0": None,
            "probability_1": None, "status": "error", "error": "",
            "warnings": [], "run_id": run_id,
        })
    for item in validation["errors"]:
        results[item["input_row_id"]]["error"] = item["error"]
    for item in validation["warnings"]:
        results[item["input_row_id"]]["warnings"].append(item["warning"])
    state = pipeline.preprocessor.to_state()
    for row_id, values in zip(validation["row_ids"], validation["X"]):
        result = results[row_id]
        missing = [FEATURE_COLUMNS[index] for index, value in enumerate(values) if value is None]
        if missing:
            result["warnings"].append("Đã điền thiếu từ train: " + ", ".join(missing))
        if len(missing) >= MISSING_WARNING_COUNT:
            result["warnings"].append("Thiếu từ 3/9 chỉ tiêu; kết quả có giới hạn lớn.")
        outside = []
        for index, value in enumerate(values):
            if value is not None and (value < state["raw_min"][index] or value > state["raw_max"][index]):
                outside.append(FEATURE_COLUMNS[index])
        if outside:
            result["warnings"].append("Ngoài min/max train: " + ", ".join(outside))
        try:
            probability = pipeline.predict_proba([values])[0]
            result["predicted_label"] = 1 if probability[1] > probability[0] else 0
            result["probability_0"], result["probability_1"] = probability
            result["status"] = "warning" if result["warnings"] else "ok"
        except (ValueError, OverflowError, ArithmeticError) as error:
            result["error"] = "Không thể dự đoán do lỗi số học: " + str(error)
    return results


def predict_csv(path_or_text, pipeline, *, from_text=False, run_id=""):
    """Kiểm tra cấu trúc toàn file trước khi dự đoán từng bản ghi."""
    records = read_csv_records(path_or_text, from_text=from_text, training=False)
    return predict_records(records, pipeline, run_id)


def results_to_csv(results):
    """Xuất CSV UTF-8; None để trống, danh sách cảnh báo thành chuỗi."""
    output = io.StringIO(newline="")
    columns = ["input_row_id", "sample_id", "predicted_label", "probability_0",
               "probability_1", "status", "error", "warnings", "run_id"]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for result in results:
        row = dict(result)
        row["warnings"] = "; ".join(row["warnings"])
        writer.writerow(row)
    return output.getvalue()


def main():
    """CLI nhận đường dẫn model, CSV đầu vào và CSV đầu ra."""
    from src.pipeline import load_pipeline
    parser = argparse.ArgumentParser(description="Dự đoán nhãn dữ liệu nước bằng GNB tự cài")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        protected = (arguments.input, arguments.model, Path(str(arguments.model) + ".sha256"))
        if any(arguments.output.resolve() == path.resolve() for path in protected):
            raise ValueError("Đầu ra không được ghi đè dữ liệu đầu vào hoặc model/checksum.")
        if arguments.output.exists():
            raise ValueError("Đầu ra đã tồn tại; chọn đường dẫn mới, không ghi đè.")
        pipeline = load_pipeline(arguments.model)
        results = predict_csv(arguments.input, pipeline, run_id=pipeline.run_id)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8", newline="") as output:
            output.write(results_to_csv(results))
        print(f"Đã xuất {len(results)} dòng: {arguments.output}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
