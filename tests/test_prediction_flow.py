"""Dữ liệu tổng hợp chỉ kiểm chứng luồng dự đoán, không phải kết quả thật."""

import csv
import io

import pytest

from src.config import FEATURE_COLUMNS
from src.pipeline import WaterPipeline
from src.predict import predict_records, predict_csv, results_to_csv


def fitted_pipeline():
    """Mẫu hai lớp đối xứng đủ 9 cột, phục vụ kiểm tra API."""
    return WaterPipeline().fit([[float(value)] * 9 for value in (0, 1, 4, 5)], [0, 0, 1, 1])


def record(value, sample_id="001"):
    """Tạo bản ghi cùng cột chuẩn, ID chỉ truy vết."""
    return {**dict.fromkeys(FEATURE_COLUMNS, value), "sample_id": sample_id}


def test_rows_keep_order_errors_missing_and_numeric_overflow():
    pipeline = fitted_pipeline()
    rows = [record(0), record("bad", "002"), record("", "003"), record(1e308, "004"), record(5, "005")]
    rows[0][FEATURE_COLUMNS[0]] = ""
    results = predict_records(rows, pipeline, "synthetic-check")
    assert [item["input_row_id"] for item in results] == list(range(5))
    assert [item["sample_id"] for item in results] == ["001", "002", "003", "004", "005"]
    assert results[0]["status"] == "warning"
    assert results[0]["predicted_label"] == 0
    for index in (1, 2, 3):
        assert results[index]["status"] == "error"
        assert results[index]["predicted_label"] is None
        assert results[index]["probability_1"] is None
    assert results[4]["predicted_label"] == 1
    parsed = list(csv.DictReader(io.StringIO(results_to_csv(results))))
    assert parsed[0]["sample_id"] == "001"
    assert parsed[1]["predicted_label"] == ""


def test_csv_reordered_columns_matches_direct_prediction():
    pipeline = fitted_pipeline()
    text = ",".join(reversed(FEATURE_COLUMNS)) + "\n" + ",".join(["5"] * 9) + "\n"
    result = predict_csv(text, pipeline, from_text=True)[0]
    assert result["predicted_label"] == pipeline.predict([[5.0] * 9])[0]
    assert result["probability_1"] == pytest.approx(pipeline.predict_proba([[5.0] * 9])[0][1])


def test_all_invalid_does_not_call_model(monkeypatch):
    pipeline = fitted_pipeline()
    def forbidden(*args):
        raise AssertionError("Không được gọi model khi mọi dòng đều lỗi")
    monkeypatch.setattr(pipeline, "predict_proba", forbidden)
    assert predict_records([record("bad")], pipeline)[0]["status"] == "error"


@pytest.mark.parametrize("destination", ["input.csv", "model.json", "model.json.sha256", "existing.csv"])
def test_cli_never_overwrites_existing_or_protected_files(tmp_path, monkeypatch, destination):
    import sys
    from src.predict import main
    for name in ("input.csv", "model.json", "model.json.sha256", "existing.csv"):
        (tmp_path / name).write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["predict", "--model", str(tmp_path / "model.json"),
                        "--input", str(tmp_path / "input.csv"), "--output", str(tmp_path / destination)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert (tmp_path / destination).read_text(encoding="utf-8") == "original"
