"""Luồng dự đoán và xử lý dòng lỗi của web so sánh."""

import importlib.util
from pathlib import Path
import csv
import io

from src.config import FEATURE_COLUMNS
from src.six_experiments import ALGORITHMS, ComparisonPipeline


def load_web_module():
    spec = importlib.util.spec_from_file_location("comparison_label_test", Path(__file__).resolve().parents[1] / "app/comparison_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_label_comparison_joins_by_row_and_keeps_failed_predictions():
    app = load_web_module()
    records = [{"sample_id": "001", "Potability": "0"},
               {"sample_id": "002", "Potability": "1"}]
    predictions = [{"row_id": 1, "algorithm": "Naive Bayes", "label": 0},
                   {"row_id": 0, "algorithm": "Naive Bayes", "label": 1},
                   {"row_id": 0, "algorithm": "C4.5", "label": None}]
    rows, skipped = app.label_comparison(records, predictions)
    assert skipped == 0
    assert [(r["sample_id"], r["actual"], r["naive_bayes"]) for r in rows] == [("001", 0, 1), ("002", 1, 0)]
    assert rows[0]["c45"] is None


def test_label_comparison_skips_unavailable_or_invalid_truth():
    rows, skipped = load_web_module().label_comparison([
        {"Potability": ""}, {}, {"Potability": "2"}, {"Potability": "text"}, {"Potability": "1"}], [])
    assert skipped == 4 and len(rows) == 1
    assert rows[0]["actual"] == 1 and rows[0]["naive_bayes"] is None


def test_truth_labels_and_sample_ids_do_not_change_predictions():
    app = load_web_module()
    X = [[float(i)] * 9 for i in range(12)]
    models = {algorithm: ComparisonPipeline(algorithm).fit(X, [0] * 6 + [1] * 6) for algorithm in ALGORITHMS}
    row = dict.fromkeys(FEATURE_COLUMNS, "2")
    first = app.predict([dict(row, Potability="0", sample_id="a")], models)
    second = app.predict([dict(row, Potability="1", sample_id="b")], models)
    assert [(r["label"], r["score_1"], r["error"]) for r in first] == [(r["label"], r["score_1"], r["error"]) for r in second]


def test_download_csv_uses_readable_classification_and_keeps_error_rows():
    app = load_web_module()
    results = [{"row_id": i, "sample_id": f"00{i}", "algorithm": "Naive Bayes", "label": label,
                "score_1": score, "score_type": "xác suất ước lượng", "error": error}
               for i, (label, score, error) in enumerate(((0, 0.2, ""), (1, 0.8, ""), (None, None, "Dữ liệu lỗi")))]
    rows = list(csv.DictReader(io.StringIO(app.prediction_csv(results).decode("utf-8-sig"))))
    assert [row["classification"] for row in rows] == ["KHÔNG UỐNG ĐƯỢC", "UỐNG ĐƯỢC", "KHÔNG XÁC ĐỊNH"]
    assert [row["sample_id"] for row in rows] == ["000", "001", "002"]
    assert rows[2]["error"] == "Dữ liệu lỗi" and rows[1]["potable_score"] == "0.8"


def test_sidebar_navigation_preserves_manual_values():
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / "app/comparison_app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    app.text_input[0].set_value("8.125").run(timeout=30)
    app.radio[0].set_value("Dự đoán CSV").run(timeout=30)
    app.radio[0].set_value("Dự đoán một mẫu").run(timeout=30)
    assert app.text_input[0].value == "8.125"
    assert not app.exception


def test_sidebar_navigation_opens_all_pages():
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / "app/comparison_app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    for label, heading in (("Dự đoán CSV", "Dự đoán nhiều mẫu từ CSV"),
                           ("Kết quả thực nghiệm", "So sánh sáu thuật toán"),
                           ("Dữ liệu gốc", "Dữ liệu trước tiền xử lý"),
                           ("Thông tin 9 chỉ tiêu", "Thông tin chín thuộc tính đầu vào"),
                           ("Dự đoán một mẫu", "Thông số mẫu nước")):
        app.radio[0].set_value(label).run(timeout=30)
        assert not app.exception
        assert any(header.value == heading for header in app.subheader)


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
