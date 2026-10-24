"""Kiểm chứng điều phối end-to-end bằng CSV tổng hợp, không phải kết quả thật."""

import csv
import json

import pytest

from src.config import FEATURE_COLUMNS
from src.evaluate import evaluate_run
from src.pipeline import load_pipeline
from src.train import prepare_data, run_training


def make_dataset(tmp_path):
    """Hai lớp, mỗi lớp 10 quan sát khác nhau đủ phép chia 5-fold."""
    path = tmp_path / "synthetic.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(FEATURE_COLUMNS + ["Potability"])
        for label in (0, 1):
            for index in range(10):
                writer.writerow([label * 20 + index + column / 10 for column in range(9)] + [label])
    return path


def test_end_to_end_freezes_data_and_evaluates_once(tmp_path):
    path = make_dataset(tmp_path)
    prepared = prepare_data(path, tmp_path)
    assert len(prepared["dev"]) == 16
    assert len(prepared["test"]) == 4
    assert set(prepared["dev"]).isdisjoint(prepared["test"])
    config = {"strategy": "mean", "standardize": False, "var_smoothing": 1e-9, "variance_floor": 1e-12}
    manifest = run_training(path, "synthetic-test", tmp_path, configs=[config])
    assert manifest["stage"] == "trained_not_tested"
    assert not (tmp_path / "results/metrics/synthetic-test/test_metrics.json").exists()
    with pytest.raises(ValueError, match="xác nhận"):
        evaluate_run("synthetic-test", path, tmp_path)
    result = evaluate_run("synthetic-test", path, tmp_path, verification_confirmed=True)
    assert result["model"]["accuracy"] == 1.0
    assert result["baseline"]["roc_auc"] == 0.5
    pipeline = load_pipeline(tmp_path / "artifacts/synthetic-test/pipeline.json")
    assert pipeline.run_id == "synthetic-test"
    updated = json.loads((tmp_path / "results/runs/synthetic-test.json").read_text(encoding="utf-8"))
    assert updated["stage"] == "evaluated"
    assert "results/metrics/synthetic-test/test_metrics.json" in updated["files"]
    with pytest.raises(ValueError, match="đã đánh giá|đánh giá"):
        evaluate_run("synthetic-test", path, tmp_path, verification_confirmed=True)
    with pytest.raises(ValueError, match="tồn tại"):
        run_training(path, "synthetic-test", tmp_path, configs=[config])


def test_changed_data_or_split_stops_comparison(tmp_path):
    path = make_dataset(tmp_path)
    prepare_data(path, tmp_path)
    with pytest.raises(ValueError, match="Checksum"):
        prepare_data(path, tmp_path, seed=43)
    with path.open("a", encoding="utf-8") as output:
        output.write(",".join(["99"] * 9 + ["0"]) + "\n")
    with pytest.raises(ValueError, match="Checksum"):
        prepare_data(path, tmp_path)


def test_source_checksum_mismatch_stops_before_split_write(tmp_path):
    path = make_dataset(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/source.json").write_text(json.dumps({"csv_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="nguồn"):
        prepare_data(path, tmp_path)
    assert not (tmp_path / "data/splits").exists()


def test_stale_figures_prevent_training_overwrite(tmp_path):
    path = make_dataset(tmp_path)
    figures = tmp_path / "results/figures/stale"
    figures.mkdir(parents=True)
    with pytest.raises(ValueError, match="tồn tại"):
        run_training(path, "stale", tmp_path)


def test_tampered_split_is_not_silently_repaired(tmp_path):
    path = make_dataset(tmp_path)
    prepare_data(path, tmp_path)
    split = tmp_path / "data/splits/test.txt"
    split.write_text("999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum"):
        prepare_data(path, tmp_path)
    assert split.read_text(encoding="utf-8") == "999\n"


@pytest.mark.parametrize("run_id", ["../escape", "", "a/b", "x" * 81])
def test_invalid_run_id(tmp_path, run_id):
    from src.train import validate_run_id
    with pytest.raises(ValueError):
        validate_run_id(run_id)
