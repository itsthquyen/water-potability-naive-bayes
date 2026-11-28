"""Biểu đồ chỉ dùng số liệu tự tính đã lưu và EDA development."""

from pathlib import Path
import json
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.data_processing.data_validation import read_csv_records, validate_records
from src.eda import summarize_features, pearson_correlation, histogram_counts
from src.artifacts import sha256_file, write_json
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    arguments = parser.parse_args()
    folder = ROOT / "results/comparison" / arguments.run_id
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    clean = validate_records(read_csv_records(RAW_DATA_PATH, training=True), training=True)
    X = [clean["X"][i] for i in summary["development_ids"]]
    y = [clean["y"][i] for i in summary["development_ids"]]
    stats = summarize_features(X, FEATURE_COLUMNS)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = [len(y) - sum(y), sum(y)]
    axes[0].bar(["Không uống được (0)", "Uống được (1)"], counts, color=["#285c89", "#ac5536"])
    axes[0].set_title("Phân bố lớp trên development")
    for i, value in enumerate(counts):
        axes[0].text(i, value + 15, str(value), ha="center")
    axes[1].barh(["pH", "Sulfate", "THM"], [stats[c]["missing"] for c in ("ph", "Sulfate", "Trihalomethanes")], color="#285c89")
    axes[1].set_title("Số ô thiếu trước xử lý")
    save(fig, "class_and_missing.png")

    fig, axes = plt.subplots(3, 3, figsize=(10, 7))
    for j, (name, axis) in enumerate(zip(FEATURE_COLUMNS, axes.flat)):
        # Tự đếm histogram; matplotlib chỉ vẽ các số đếm đã tính.
        observed = [row[j] for row in X if row[j] is not None]
        histogram = histogram_counts(observed, bins=18)
        edges = histogram['edges']
        axis.bar(edges[:-1], histogram['counts'], width=[b-a for a,b in zip(edges, edges[1:])],
                 align='edge', color="#396b92", edgecolor="white")
        axis.set_title(name)
    save(fig, "histograms.png")

    correlations = []
    for j in range(9):
        correlations.append([pearson_correlation([row[j] for row in X], [row[k] for row in X])[0] for k in range(9)])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(9), FEATURE_COLUMNS, rotation=70, ha="right")
    ax.set_yticks(range(9), FEATURE_COLUMNS)
    fig.colorbar(im, ax=ax)
    save(fig, "correlation.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = ["Naive Bayes", "ID3", "C4.5", "KNN", "Logistic", "SVM"]
    for i, algorithm in enumerate(ALGORITHMS):
        metric = summary["models"][algorithm]["test"]
        ax.bar(i - .18, 100 * metric["accuracy"], .34, color="#2a638c", label="Accuracy" if i == 0 else None)
        ax.bar(i + .18, 100 * metric["balanced_accuracy"], .34, color="#ae643e", label="Balanced Accuracy" if i == 0 else None)
    ax.set_xticks(range(6), labels)
    ax.set_ylim(0, 80)
    ax.set_ylabel("%")
    ax.legend(loc="upper right")
    save(fig, "comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for algorithm, axis in zip(ALGORITHMS, axes.flat):
        matrix = summary["models"][algorithm]["test"]["confusion_matrix"]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=400)
        for j in range(2):
            for k in range(2):
                axis.text(k, j, str(matrix[j][k]), ha="center", va="center", color="white" if matrix[j][k] > 220 else "black", fontsize=14)
        axis.set_xticks([0, 1]); axis.set_yticks([0, 1])
        axis.set_xlabel("Dự đoán"); axis.set_ylabel("Thật")
        axis.set_title(NAMES[algorithm])
    save(fig, "confusions.png")

    fig, axis = plt.subplots(figsize=(8, 6))
    for algorithm in ALGORITHMS:
        metric = summary["models"][algorithm]["test"]
        axis.plot([p["fpr"] for p in metric["roc_curve"]], [p["tpr"] for p in metric["roc_curve"]],
                  label=f"{NAMES[algorithm]} ({metric['roc_auc']:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_xlabel("False Positive Rate"); axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    save(fig, "roc_comparison.png")
    (folder / "eda.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files'] = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob('*')
                         if p.is_file() and p.name != 'manifest.json'}
    write_json(folder / 'manifest.json', manifest)
    print("Saved six figures and development EDA; checksums updated.")


if __name__ == "__main__":
    main()
