"""Thực nghiệm sáu thuật toán tự cài, cùng split và năm fold đã công bố."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .algorithms.c45 import C45
from .algorithms.id3 import ID3
from .algorithms.knn import KNearestNeighbors
from .algorithms.logistic import LogisticRegression
from .algorithms.svm import LinearSVM
from .config import FEATURE_COLUMNS, PROJECT_ROOT, RAW_DATA_PATH
from .data_processing.discretization import Discretizer
from .metrics import evaluate_predictions, summarize_scores
from .algorithms.naive_bayes import NaiveBayes
from .artifacts import sha256_file, write_json, validate_run_id, runtime_metadata
from .data_processing.preprocessing import prepare_data, NumericPreprocessor, _validate_matrix

RUN_ID = "20261128-c45-full-v1"
ALGORITHMS = ("naive_bayes", "id3", "c45", "knn", "logistic", "svm")
NAMES = {"naive_bayes": "Naive Bayes", "id3": "ID3", "c45": "C4.5",
         "knn": "KNN", "logistic": "Logistic Regression", "svm": "SVM tuyến tính"}


class ComparisonPipeline:
    """Một cấu hình cố định mỗi thuật toán; fit mới toàn bộ ở từng fold."""

    def __init__(self, algorithm):
        if algorithm not in ALGORITHMS:
            raise ValueError("Thuật toán không được hỗ trợ.")
        self.algorithm = algorithm

    def fit(self, X, y):
        if self.algorithm == "c45":
            self.processor = None  # C4.5 tự xử lý số liên tục và ô thiếu.
            self.model = C45().fit(X, y)
            return self
        nominal = self.algorithm in ("naive_bayes", "id3")
        processor = Discretizer() if nominal else NumericPreprocessor()
        processor.fit(X)
        transformed = processor.transform(X)
        if self.algorithm == "naive_bayes":
            model = NaiveBayes().fit(transformed, y, processor.category_sizes)
        else:
            builders = {"id3": ID3,
                        "knn": KNearestNeighbors, "logistic": LogisticRegression, "svm": LinearSVM}
            model = builders[self.algorithm]().fit(transformed, y)
        self.processor, self.model = processor, model
        return self

    def predict_with_scores(self, X):
        transformed = self.processor.transform(X) if self.processor is not None else X
        if self.algorithm == "naive_bayes":
            labels, probabilities = self.model.predict_with_proba(transformed)
            return labels, [values[1] for values in probabilities]
        return self.model.predict_with_scores(transformed)

    def to_state(self):
        processor_state = (None if self.processor is None else self.processor.to_state()
                           if isinstance(self.processor, Discretizer) else self.processor.state)
        return {"algorithm": self.algorithm, "feature_columns": FEATURE_COLUMNS,
                "preprocessor": processor_state, "model": self.model.to_state()}

    @classmethod
    def from_state(cls, state):
        if state.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Sai thứ tự chín thuộc tính trong mô hình.")
        instance = cls(state["algorithm"])
        params = state["model"]
        if instance.algorithm == "c45":
            instance.processor = None
            instance.model = C45.from_state(params)
        elif instance.algorithm in ("naive_bayes", "id3"):
            instance.processor = Discretizer.from_state(state["preprocessor"])
            if instance.algorithm == "naive_bayes":
                instance.model = NaiveBayes.from_state(params)
            else:
                instance.model = ID3()
                instance.model.root = params["root"]
                instance.model.n_features = params["n_features"]
        else:
            instance.processor = NumericPreprocessor()
            instance.processor.state = state["preprocessor"]
            if instance.algorithm == "knn":
                instance.model = KNearestNeighbors(params["k"]).fit(params["rows"], params["labels"])
            else:
                builder = LogisticRegression if instance.algorithm == "logistic" else LinearSVM
                config = {k: v for k, v in params.items() if k not in ("weights", "bias")}
                instance.model = builder(**config)
                instance.model.weights = params["weights"]
                instance.model.bias = params["bias"]
        return instance


def export_arff(path, X, y, nominal=False):
    lines = ["@relation water_potability", ""]
    for name in FEATURE_COLUMNS:
        lines.append(f"@attribute {name} " + ("{0,1,2}" if nominal else "numeric"))
    lines.extend(("@attribute Potability {0,1}", "", "@data"))
    for row, label in zip(X, y):
        values = ["?" if value is None else str(int(value)) if nominal else repr(value) for value in row]
        lines.append(",".join(values + [str(label)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(run_id=RUN_ID):
    validate_run_id(run_id)
    folder = PROJECT_ROOT / "results" / "comparison" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    data = prepare_data(RAW_DATA_PATH)
    X, y, ids = data["clean"]["X"], data["clean"]["y"], data["clean"]["row_ids"]
    dev, test = data["dev"], data["test"]
    X_dev, y_dev = [X[i] for i in dev], [y[i] for i in dev]
    X_test, y_test = [X[i] for i in test], [y[i] for i in test]
    summary = {"run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
               "raw_sha256": sha256_file(RAW_DATA_PATH), "seed": 42,
               "environment": runtime_metadata(),
               "development_ids": [ids[i] for i in dev], "test_ids": [ids[i] for i in test],
               "protocol": "Fixed configurations; 5-fold CV on development; historical holdout reused, not external validation.",
               "models": {}}
    baseline_label = int(sum(y_dev) > len(y_dev) / 2)
    summary["baseline"] = evaluate_predictions(y_test, [baseline_label] * len(test), [sum(y_dev) / len(dev)] * len(test))
    for algorithm in ALGORITHMS:
        folds = []
        for fold_id, (train, validation) in enumerate(data["folds"]):
            model = ComparisonPipeline(algorithm)
            start = perf_counter()
            model.fit([X_dev[i] for i in train], [y_dev[i] for i in train])
            fit_seconds = perf_counter() - start
            labels, scores = model.predict_with_scores([X_dev[i] for i in validation])
            metric = evaluate_predictions([y_dev[i] for i in validation], labels, scores)
            metric.pop("roc_curve")
            metric["fit_seconds"] = fit_seconds
            folds.append(metric)
            if algorithm == 'c45':
                prediction_path = folder / f'c45_fold_{fold_id}_predictions.csv'
                with prediction_path.open('w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow(('row_id', 'actual', 'predicted', 'score_1'))
                    writer.writerows((ids[dev[i]], y_dev[i], label, score)
                                     for i, label, score in zip(validation, labels, scores))
        pipeline = ComparisonPipeline(algorithm)
        start = perf_counter()
        pipeline.fit(X_dev, y_dev)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        labels, scores = pipeline.predict_with_scores(X_test)
        predict_seconds = perf_counter() - start
        metrics = evaluate_predictions(y_test, labels, scores)
        state = pipeline.to_state()
        reloaded = ComparisonPipeline.from_state(state)
        assert reloaded.predict_with_scores(X_test[:8]) == (labels[:8], scores[:8])
        write_json(folder / "models" / f"{algorithm}.json", state)
        with (folder / f"{algorithm}_predictions.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("row_id", "actual", "predicted", "score_1"))
            writer.writerows((ids[index], actual, label, score) for index, actual, label, score in zip(test, y_test, labels, scores))
        summary["models"][algorithm] = {"name": NAMES[algorithm], "folds": folds,
            "configuration": (pipeline.model.config if algorithm == "c45" else
                              {"fixed_configuration": "see algorithm constructor defaults"}),
            "preprocessing": ("raw numeric with native fractional missing-value handling" if algorithm == "c45"
                              else "train-only median and discretization" if algorithm in ("naive_bayes", "id3")
                              else "train-only median and z-score"),
            "cv": {field: summarize_scores([row[field] for row in folds]) for field in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")},
            "test": metrics, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
        print(algorithm, "CV BA", summary["models"][algorithm]["cv"]["balanced_accuracy"],
              "TEST", metrics["confusion_matrix"], metrics["accuracy"], flush=True)
        if algorithm == "c45":
            summary["models"][algorithm]["pruning"] = pipeline.model.pruning
    nominal_processor = Discretizer().fit(X_dev)
    numeric_processor = NumericPreprocessor().fit(X_dev)
    arff = folder / "weka"
    export_arff(arff / "raw.arff", X, y)
    export_arff(arff / "development_raw.arff", X_dev, y_dev)
    export_arff(arff / "test_raw.arff", X_test, y_test)
    for fold_id, (train_indices, validation_indices) in enumerate(data['folds']):
        for name, indices in (('train', train_indices), ('validation', validation_indices)):
            export_arff(arff / 'cv' / f'fold_{fold_id}_{name}.arff',
                        [X_dev[i] for i in indices], [y_dev[i] for i in indices])
    for part, values, targets in (("development", X_dev, y_dev), ("test", X_test, y_test)):
        export_arff(arff / f"{part}_nominal.arff", nominal_processor.transform(values), targets, True)
        export_arff(arff / f"{part}_numeric.arff", numeric_processor.transform(values), targets)
    write_json(folder / "preprocessing.json", {"nominal": nominal_processor.to_state(), "numeric": numeric_processor.state})
    write_json(folder / "summary.json", summary)
    files = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob("*") if p.is_file()}
    source_paths = [p for p in (PROJECT_ROOT / "src").rglob("*.py")]
    write_json(folder / "manifest.json", {"run_id": run_id, "raw_sha256": summary["raw_sha256"], "files": files,
               "source_hashes": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in source_paths}})
    print(folder, flush=True)
    return folder


def load_models(folder):
    """Nạp duy nhất các mô hình đúng hash trong manifest của lần chạy."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    models = {}
    for algorithm in ALGORITHMS:
        relative = f"models/{algorithm}.json"
        if sha256_file(folder / relative) != manifest["files"][relative]:
            raise ValueError(f"Checksum mô hình sai: {algorithm}")
        models[algorithm] = ComparisonPipeline.from_state(json.loads((folder / relative).read_text(encoding="utf-8")))
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    run_experiment(parser.parse_args().run_id)
