"""Kiểm thử demo bằng AppTest, không huấn luyện hay thay validation thật."""

import importlib.util
import hashlib
import json
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.config import FEATURE_COLUMNS, PROJECT_ROOT
from src.pipeline import WaterPipeline, load_pipeline, save_pipeline
from src.predict import predict_csv, predict_records, results_to_csv


APP_PATH = PROJECT_ROOT / "app" / "app.py"


def load_app_module():
    """Nạp hàm giao diện để kiểm thử xác minh file mà không chạy main."""
    specification = importlib.util.spec_from_file_location("water_app_tests", APP_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_missing_model_has_warning_no_prediction_or_training(tmp_path):
    """Model không có phải báo đúng, không gọi predictor hoặc fit."""
    missing_path = tmp_path / "missing_pipeline.json"
    with patch("src.pipeline.load_pipeline", side_effect=AssertionError("Không nạp model thiếu")), \
         patch("src.predict.predict_records", side_effect=AssertionError("Không được dự đoán")), \
         patch("src.predict.predict_csv", side_effect=AssertionError("Không được dự đoán")):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.sidebar.text_input[0].set_value(str(missing_path)).run()
    assert not app.exception
    assert any("Chưa có model" in error.value for error in app.error)
    assert any("không thay thế kiểm nghiệm" in warning.value for warning in app.warning)
    assert not app.button
    assert not app.dataframe


def test_shared_prediction_contract_available():
    """Không bỏ qua test khi src đang thiếu; hợp đồng thật phải import được."""
    assert len(FEATURE_COLUMNS) == 9
    for function in (load_pipeline, predict_csv, predict_records, results_to_csv):
        assert callable(function)


def test_loaded_model_uses_shared_validation_and_does_not_train(tmp_path):
    """Model tổng hợp cho test UI; fit chỉ ở fixture trước khi mở app."""
    pipeline = WaterPipeline().fit(
        [[float(value)] * len(FEATURE_COLUMNS) for value in (1, 2, 8, 9)],
        [0, 0, 1, 1],
    )
    model_path = tmp_path / "pipeline.json"
    save_pipeline(pipeline, model_path, run_id="synthetic-ui-test")
    with patch.object(WaterPipeline, "fit", side_effect=AssertionError("App không được train")):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.sidebar.text_input[0].set_value(str(model_path)).run()
        assert not app.exception
        assert len([item for item in app.text_input if (item.key or "").startswith("feature_")]) == 9
        next(button for button in app.button if button.label == "Dự đoán mẫu").click().run()
        assert not app.exception
        assert app.error
        for feature in FEATURE_COLUMNS:
            app.text_input(key=f"feature_{feature}").set_value("2")
        next(button for button in app.button if button.label == "Dự đoán mẫu").click().run()
        assert not app.exception
        assert not app.error
        assert any("theo nhãn dữ liệu" in item.value for item in app.success)
        assert app.dataframe
        assert any("Chưa có kết quả thực nghiệm" in item.value for item in app.info)
        app.sidebar.text_input[0].set_value(str(tmp_path / "missing.json")).run()
        assert not app.exception
        assert not app.dataframe


def test_verified_files_require_run_model_and_content_checksums(tmp_path, monkeypatch):
    """Fixture tổng hợp chỉ kiểm tra tính toàn vẹn, không là số liệu thực nghiệm."""
    module = load_app_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    run_id = "synthetic-test"
    metrics_path = tmp_path / "results" / "metrics" / run_id / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    content = json.dumps({"run_id": run_id, "purpose": "integrity-test"}).encode()
    metrics_path.write_bytes(content)
    manifest_path = tmp_path / "results" / "runs" / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {"run_id": run_id, "model_sha256": "a" * 64, "files": {metrics_path.relative_to(tmp_path).as_posix(): hashlib.sha256(content).hexdigest()}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module.verified_run_files(run_id, "a" * 64) == {metrics_path: content}
    # Không đọc file ngoài run, kể cả khi manifest ghi checksum của file đó.
    manifest["files"]["../outside.json"] = "c" * 64
    manifest["files"]["results/metrics/another-run/metrics.json"] = "d" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module.verified_run_files(run_id, "a" * 64) == {metrics_path: content}
    with pytest.raises(ValueError, match="checksum model"):
        module.verified_run_files(run_id, "b" * 64)
    metrics_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        module.verified_run_files(run_id, "a" * 64)


@pytest.mark.parametrize("run_id", ["../outside", "a/b", "..", "", "/absolute"])
def test_untrusted_run_id_rejected_before_reading(run_id):
    """Run ID không được trở thành đường dẫn tùy ý."""
    with pytest.raises(ValueError, match="Run ID"):
        load_app_module().verified_run_files(run_id, "a" * 64)
