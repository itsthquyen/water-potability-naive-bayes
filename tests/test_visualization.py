"""Smoke test Agg và thống kê tính tay trên mẫu tổng hợp, không phải dataset."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest

from src.config import FEATURE_COLUMNS
from src.visualization import create_eda_figures, create_evaluation_figures


def test_eda_figures_and_hand_statistics(tmp_path, monkeypatch):
    def forbid_statistics(*args, **kwargs):
        raise AssertionError("Không được gọi tiện ích tính thống kê của matplotlib")

    monkeypatch.setattr(Axes, "hist", forbid_statistics)
    monkeypatch.setattr(Axes, "boxplot", forbid_statistics)
    rows = [[value] * len(FEATURE_COLUMNS) for value in [1, 2, 3, 4]]
    rows[0][1] = None
    rows[1][2] = None
    rows[2][2] = float("nan")
    for row in rows:
        row[3] = 7
        row[4] = None
    destination = tmp_path / "chosen" / "eda"
    paths = create_eda_figures(rows, [0, 0, 1, 1], destination)
    assert len(paths) == 13
    assert len(list(destination.glob("*.png"))) == 13
    for path in paths:
        assert isinstance(path, Path)
        assert path.parent == destination
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads((destination / "eda_manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"] == [path.name for path in paths]
    stats = manifest["statistics"]
    assert stats["class_counts"] == {"0": 2, "1": 2}
    assert stats["features"]["ph"]["mean"] == 2.5
    assert stats["features"]["ph"]["quartiles"] == {"q1": 1.75, "q2": 2.5, "q3": 3.25}
    assert stats["features"]["Hardness"]["missing"] == 1
    assert stats["histograms"]["ph"]["edges"] == pytest.approx(
        [1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4])
    assert stats["histograms"]["ph"]["per_class"] == {
        "0": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "1": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]}
    assert stats["per_class"]["0"]["ph"]["median"] == 1.5
    assert stats["pearson"]["values"][0][1] == pytest.approx(1)
    assert stats["pearson"]["pair_counts"][0][1] == 3
    assert stats["pearson"]["values"][0][3] is None
    assert stats["pearson"]["pair_counts"][0][4] == 0
    assert plt.get_fignums() == []


def test_evaluation_draws_supplied_points_and_preserves_stats(tmp_path, monkeypatch):
    supplied = {"confusion_matrix": [[3, 1], [2, 4]], "roc_auc": 0.75,
                "roc_curve": [{"threshold": None, "fpr": 0, "tpr": 0},
                              {"threshold": 0.5, "fpr": 0.5, "tpr": 1},
                              {"threshold": 0.1, "fpr": 1, "tpr": 1}]}
    plotted = []
    original_plot = Axes.plot

    def capture_plot(axis, x, y, *args, **kwargs):
        plotted.append((list(x), list(y)))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    paths = create_evaluation_figures(supplied, tmp_path)
    assert [path.name for path in paths] == ["confusion_matrix.png", "roc.png"]
    assert plotted[0] == ([0, 0.5, 1], [0, 1, 1])
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"] == supplied
    assert all(path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("points", [None, []])
def test_no_roc_points_produces_only_confusion(tmp_path, points):
    paths = create_evaluation_figures({"confusion_matrix": [[2, 0], [0, 0]],
                                      "roc_auc": None, "roc_curve": points}, tmp_path)
    assert [path.name for path in paths] == ["confusion_matrix.png"]
    assert not (tmp_path / "roc.png").exists()


@pytest.mark.parametrize("rows,labels", [([], []), ([[1] * 9], []),
    ([[1] * 8], [0]), ([[1] * 9], [2]), ([[1] * 9], [True]),
    ([[float("inf")] * 9], [0])])
def test_invalid_eda_does_not_create_output(tmp_path, rows, labels):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        create_eda_figures(rows, labels, destination)
    assert not destination.exists()


@pytest.mark.parametrize("metrics", [
    {"confusion_matrix": [[1]]}, {"confusion_matrix": [[1, -1], [0, 1]]},
    {"confusion_matrix": [[True, 0], [0, 1]]},
    {"confusion_matrix": [[1, 0], [0, 1]], "roc_curve": [{"fpr": 2, "tpr": 1}]},
    {"confusion_matrix": [[1, 0], [0, 1]], "roc_curve": [
        {"fpr": 1, "tpr": 1}, {"fpr": 0, "tpr": 1}]},
    {"confusion_matrix": [[1, 0], [0, 1]], "roc_auc": float("nan")}])
def test_invalid_evaluation_does_not_create_output(tmp_path, metrics):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        create_evaluation_figures(metrics, destination)
    assert not destination.exists()
