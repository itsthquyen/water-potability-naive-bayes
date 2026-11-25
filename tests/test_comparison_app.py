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
