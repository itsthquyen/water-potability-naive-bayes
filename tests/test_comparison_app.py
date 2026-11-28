"""Luồng dự đoán và xử lý dòng lỗi của web so sánh."""

import importlib.util
from pathlib import Path

from src.config import FEATURE_COLUMNS
from src.six_experiments import ALGORITHMS, ComparisonPipeline


def test_web_csv_rows_keep_ids_and_errors_for_all_six_models():
    spec = importlib.util.spec_from_file_location("comparison_web_test", Path(__file__).resolve().parents[1] / "app/comparison_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    X = [[float(i)] * 9 for i in range(12)]
    models = {algorithm: ComparisonPipeline(algorithm).fit(X, [0] * 6 + [1] * 6) for algorithm in ALGORITHMS}
    good = dict.fromkeys(FEATURE_COLUMNS, "2")
    good["sample_id"] = "sample-a"
    bad = dict(good, ph="invalid", sample_id="sample-b")
    results = app.predict([good, bad], models)
    assert len(results) == 12
    assert all(row["sample_id"] == "sample-a" and row["label"] in (0, 1) and not row["error"] for row in results[:6])
    assert all(row["sample_id"] == "sample-b" and row["label"] is None and row["error"] for row in results[6:])
    assert results[5]["score_type"] == "biên quyết định"


def test_editing_manual_input_and_reset_discard_old_predictions():
    """Kết quả phải biến mất khi đầu vào thay đổi hoặc người dùng xóa ô nhập."""
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[1] / "app/comparison_app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == "Dự đoán và so sánh").click().run(timeout=30)
    assert len(app.session_state["six_manual"]) == 6
    app.text_input[0].set_value("8.1").run(timeout=30)
    assert "six_manual" not in app.session_state
    next(button for button in app.button if button.label == "Dự đoán và so sánh").click().run(timeout=30)
    assert len(app.session_state["six_manual"]) == 6
    next(button for button in app.button if button.label == "Xóa ô nhập").click().run(timeout=30)
    assert all(field.value == "" for field in app.text_input)
    assert "six_manual" not in app.session_state
    next(button for button in app.button if button.label == "Dự đoán và so sánh").click().run(timeout=30)
    assert all(row["label"] is None and row["error"] for row in app.session_state["six_manual"])
    assert not app.exception


def test_comparison_app_ui_rendering_and_prediction():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app/comparison_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert any("không thay thế kiểm nghiệm" in warning.value for warning in app.warning)

    predict_btn = next((b for b in app.button if "Dự đoán và so sánh" in b.label), None)
    assert predict_btn is not None
    predict_btn.click().run()
    assert not app.exception
    assert any("Kết quả phân loại" in header.value for header in app.subheader)
