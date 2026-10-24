"""Đánh giá một lần pipeline đã chốt, không fit trên tập test."""

import argparse
import csv
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import PROJECT_ROOT, RAW_DATA_PATH, TEST_FRACTION, N_FOLDS
from src.train import prepare_data, validate_run_id


def _confined_path(root, relative):
    """Chỉ nhận đường dẫn tương đối nằm trong root sau khi giải symlink."""
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError("Đường dẫn manifest phải nằm trong output root.")
    destination = (root / path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Đường dẫn manifest thoát output root.")
    return destination


def _durable_json(path, value):
    """Flush/fsync JSON tạm rồi replace; không để manifest/checkpoint nửa chừng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


@contextmanager
def _evaluation_lock(path):
    """Khóa OS không chờ; tự nhả cả khi process chết, giữ file khóa để tránh race."""
    with path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Run đang được đánh giá bởi tiến trình khác (exclusive lock).") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def evaluate_run(run_id, csv_path=RAW_DATA_PATH, output_root=PROJECT_ROOT, *, verification_confirmed=False):
    """Mở test chỉ khi đã xác nhận kiểm chứng; kiểm tra liên kết checksum.

    Không suy luận lần hai cùng run_id. Cho phép render/publish lại từ checkpoint
    đã xác minh, không dùng sai số test để điều chỉnh tham số hoặc seed.
    """
    if verification_confirmed is not True:
        raise ValueError("Cần xác nhận kiểm chứng tự cài đã đạt trước khi mở test.")
    run_id = validate_run_id(run_id)
    output_root = Path(output_root).resolve()
    run_path = _confined_path(output_root, f"results/runs/{run_id}.json")
    with _evaluation_lock(run_path.with_suffix(".evaluate.lock")):
        return _evaluate_locked(run_id, csv_path, output_root, run_path)


def _evaluate_locked(run_id, csv_path, output_root, run_path):
    """Xác minh trước test; retry chỉ render checkpoint, không suy luận lần hai."""
    from src.pipeline import sha256_file
    from src.visualization import create_evaluation_figures

    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id:
        raise ValueError("run_id trong manifest không khớp.")
    stage = manifest["stage"]
    if stage not in ("trained_not_tested", "evaluation_started", "evaluation_checkpointed"):
        raise ValueError("Run đã đánh giá hoặc không ở trạng thái chờ test; không ghi đè.")
    # Không ký lại các CV/EDA/model cũ: mọi hash phải khớp trước khi mở test.
    for relative, checksum in manifest["files"].items():
        if sha256_file(_confined_path(output_root, relative)) != checksum:
            raise ValueError(f"Checksum file manifest không khớp: {relative}")
    checkpoint_path = _confined_path(output_root, f"results/runs/{run_id}.evaluation_checkpoint.json")
    if stage == "evaluation_started":
        raise ValueError("Evaluation interrupted: inference đã bắt đầu nhưng chưa có checkpoint được xác nhận; không tự infer lại.")
    if stage == "trained_not_tested":
        if checkpoint_path.exists():
            raise ValueError("Checkpoint tồn tại ngoài trạng thái hợp lệ; không tự infer lại.")
        for filename in ("test_metrics.json", "baseline_test_metrics.json", "test_predictions.csv", "error_analysis.json"):
            if _confined_path(output_root, f"results/metrics/{run_id}/{filename}").exists():
                raise ValueError("Output đánh giá đã tồn tại; không tự infer lại.")
        checkpoint = _infer_checkpoint(run_id, csv_path, output_root, run_path, manifest)
        _durable_json(checkpoint_path, checkpoint)
        manifest["evaluation_checkpoint_sha256"] = sha256_file(checkpoint_path)
        manifest["stage"] = "evaluation_checkpointed"
        _durable_json(run_path, manifest)
    else:
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != manifest["evaluation_checkpoint_sha256"]:
            raise ValueError("Evaluation interrupted: checkpoint thiếu hoặc checksum không khớp; không tự infer lại.")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint["run_id"] != run_id:
        raise ValueError("run_id checkpoint không khớp.")
    with TemporaryDirectory(prefix=f".{run_id}-evaluation-", dir=run_path.parent) as temporary:
        staging = Path(temporary)
        for filename, content in checkpoint["outputs"].items():
            path = _confined_path(staging.resolve(), f"results/metrics/{run_id}/{filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        create_evaluation_figures(checkpoint["model"], staging / "results" / "figures" / run_id)
        additions = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative = path.relative_to(staging).as_posix()
                destination = _confined_path(output_root, relative)
                if relative in manifest["files"]:
                    raise ValueError("Không ghi đè output cũ đã có trong manifest.")
                if destination.exists():
                    expected = manifest.get("evaluation_pending_files", {}).get(relative)
                    if expected is None or sha256_file(destination) != expected:
                        raise ValueError("Output đã tồn tại ngoài giao dịch hoặc bị thay đổi.")
                additions[relative] = sha256_file(path)
        manifest["evaluation_pending_files"] = additions
        _durable_json(run_path, manifest)
        for relative in additions:
            destination = _confined_path(output_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = staging / relative
            with source.open("r+b") as staged_file:
                os.fsync(staged_file.fileno())
            os.replace(source, destination)
    manifest["files"].update(additions)
    manifest.pop("evaluation_pending_files", None)
    manifest["stage"] = "evaluated"
    manifest["verification_confirmed"] = True
    manifest["acceptance_status"] = (
        "development_evaluation_not_final_signed_submission" if manifest["environment"]["dirty"] else
        "unverified_source_evaluation" if checkpoint.get("source_verified") is not True else "evaluated_candidate")
    _durable_json(run_path, manifest)
    return {"run_id": run_id, "model": checkpoint["model"], "baseline": checkpoint["baseline"]}


def _infer_checkpoint(run_id, csv_path, output_root, run_path, manifest):
    """Mở test tối đa một lần; checkpoint gồm metrics, dự đoán và phân tích lỗi."""
    from src.metrics import evaluate_predictions
    from src.pipeline import load_pipeline, sha256_file

    data_manifest_path = _confined_path(output_root, "data/manifest.json")
    if sha256_file(data_manifest_path) != manifest["data_manifest_sha256"]:
        raise ValueError("Manifest phép chia không khớp lần huấn luyện.")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("test_fraction") != TEST_FRACTION or data_manifest.get("n_folds") != N_FOLDS:
        raise ValueError("Cấu hình test_fraction/n_folds khác manifest đã huấn luyện; dừng trước test.")
    seed = data_manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed manifest phải là số nguyên không âm.")
    model_path = _confined_path(output_root, manifest.get("model_path", f"artifacts/{run_id}/pipeline.json"))
    pipeline = load_pipeline(model_path, expected_sha256=manifest["model_sha256"])
    if pipeline.run_id != run_id:
        raise ValueError("run_id trong model không khớp manifest.")
    if sha256_file(csv_path) != manifest["csv_sha256"]:
        raise ValueError("CSV không khớp checksum của lần huấn luyện.")
    data = prepare_data(csv_path, output_root, seed=seed)
    if sha256_file(data["manifest_path"]) != manifest["data_manifest_sha256"]:
        raise ValueError("Manifest phép chia không khớp lần huấn luyện.")
    X = [data["clean"]["X"][index] for index in data["test"]]
    y = [data["clean"]["y"][index] for index in data["test"]]
    ids = [data["clean"]["row_ids"][index] for index in data["test"]]
    manifest["stage"] = "evaluation_started"
    _durable_json(run_path, manifest)
    probabilities = pipeline.predict_proba(X)
    scores = [row[1] for row in probabilities]
    predictions = [1 if row[1] > row[0] else 0 for row in probabilities]
    metrics = evaluate_predictions(y, predictions, scores)
    prior = manifest["baseline_prior_1"]
    baseline_label = 1 if prior > 0.5 else 0
    baseline_metrics = evaluate_predictions(y, [baseline_label] * len(y), [prior] * len(y))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["row_id", "actual", "predicted", "probability_0", "probability_1", "error_type", "run_id"])
    errors = []
    for row_id, actual, prediction, probability, values in zip(ids, y, predictions, probabilities, X):
        error_type = "FP" if actual == 0 and prediction == 1 else "FN" if actual == 1 and prediction == 0 else ""
        writer.writerow([row_id, actual, prediction, *probability, error_type, run_id])
        if error_type:
            errors.append({"row_id": row_id, "actual": actual, "predicted": prediction,
                           "error_type": error_type, "probability_1": probability[1],
                           "missing_count": sum(value is None for value in values), "features": values})
    matrix = metrics["confusion_matrix"]
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    error_analysis = {
        "run_id": run_id, "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "false_positive_rate": false_positive / (false_positive + true_negative) if false_positive + true_negative else None,
        "false_discovery_rate": false_positive / (false_positive + true_positive) if false_positive + true_positive else None,
        "examples": (errors[:5] + [item for item in errors if item["error_type"] == "FN"][:5]),
        "note": "Chỉ phân tích kết quả cuối, không dùng để chọn lại cấu hình."}
    outputs = {"test_predictions.csv": output.getvalue()}
    for filename, value in (("test_metrics.json", metrics), ("baseline_test_metrics.json", baseline_metrics),
                            ("error_analysis.json", error_analysis)):
        outputs[filename] = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return {"run_id": run_id, "model": metrics, "baseline": baseline_metrics, "outputs": outputs,
            "source_verified": pipeline.metadata.get("source_verified") is True}


def main():
    """CLI yêu cầu xác nhận rõ, không tự train hoặc mở test ngầm."""
    parser = argparse.ArgumentParser(description="Đánh giá pipeline đã chốt trên test giữ riêng")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--verification-confirmed", action="store_true")
    arguments = parser.parse_args()
    try:
        result = evaluate_run(arguments.run_id, arguments.data, verification_confirmed=arguments.verification_confirmed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
