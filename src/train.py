"""CLI thực nghiệm tự cài: audit, chia tập, CV và fit cuối, chưa mở test."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.config import (PROJECT_ROOT, RAW_DATA_PATH, SEED, N_FOLDS, TEST_FRACTION,
                        DEFAULT_VAR_SMOOTHING, VARIANCE_FLOOR, FEATURE_COLUMNS)


def write_json(path, value):
    """Ghi JSON UTF-8 hữu hạn, tạo thư mục cha; chỉ dùng cho output thực tế."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment_metadata():
    """Ghi runtime, phiên bản thư viện và commit/dirty thực tế, không tự commit."""
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "streamlit", "pytest", "nbformat", "nbclient"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = subprocess.run(["rtk", "git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)
    status = subprocess.run(["rtk", "git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)
    return {"python": platform.python_version(), "packages": versions,
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}


def validate_run_id(run_id):
    """Chặn ID rỗng/đường dẫn; mỗi lần chạy có thư mục riêng không ghi đè."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id chỉ gồm chữ/số, gạch dưới/gạch ngang, dài tối đa 80.")
    return run_id


def prepare_data(csv_path, output_root=PROJECT_ROOT, seed=SEED):
    """Audit CSV, loại trùng cùng nhãn, chặn mâu thuẫn và lưu split checksum.

    Không fit/EDA trên test; phép chia dùng ID trước lọc. Nếu manifest đã có,
    checksum khác sẽ dừng thay vì âm thầm thay phép chia chuẩn.
    """
    from src.data_validation import read_csv_records, validate_records, clean_training_data
    from src.preprocessing import stratified_holdout, stratified_folds
    from src.pipeline import sha256_file
    csv_path, output_root = Path(csv_path), Path(output_root)
    validated = validate_records(read_csv_records(csv_path, training=True), training=True)
    if validated["errors"]:
        raise ValueError("Dữ liệu huấn luyện có dòng lỗi: " + json.dumps(validated["errors"][:10], ensure_ascii=False))
    clean = clean_training_data(validated["X"], validated["y"], validated["row_ids"])
    dev, test = stratified_holdout(clean["y"], test_fraction=TEST_FRACTION, seed=seed)
    dev_y = [clean["y"][index] for index in dev]
    folds = stratified_folds(dev_y, n_folds=N_FOLDS, seed=seed)
    split_values = {"development": [clean["row_ids"][index] for index in dev],
                    "test": [clean["row_ids"][index] for index in test]}
    for fold_id, (train, validation) in enumerate(folds):
        split_values[f"fold_{fold_id}_train"] = [clean["row_ids"][dev[index]] for index in train]
        split_values[f"fold_{fold_id}_validation"] = [clean["row_ids"][dev[index]] for index in validation]
    contents = {name: "".join(f"{row_id}\n" for row_id in ids).encode("utf-8") for name, ids in split_values.items()}
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    manifest_path = output_root / "data" / "manifest.json"
    csv_checksum = sha256_file(csv_path)
    source_path = output_root / "data" / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {"source_status": "Chưa xác minh nguồn; không coi dữ liệu này là thực nghiệm chính thức."}
    if source_path.exists() and source.get("csv_sha256") != csv_checksum:
        raise ValueError("Checksum CSV không khớp nguồn source.json.")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["csv_sha256"] != csv_checksum or old["split_checksums"] != checksums:
            raise ValueError("Checksum dữ liệu/phép chia khác manifest chuẩn; dừng để kiểm tra.")
        for name, checksum in checksums.items():
            split_path = output_root / "data" / "splits" / f"{name}.txt"
            if not split_path.is_file() or sha256_file(split_path) != checksum:
                raise ValueError("Checksum file phép chia khác manifest; không tự ghi đè.")
    split_directory = output_root / "data" / "splits"
    split_directory.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        for name, content in contents.items():
            (split_directory / f"{name}.txt").write_bytes(content)
    manifest = {"csv_sha256": csv_checksum, "source": source,
                "raw_count": len(validated["row_ids"]), "clean_count": len(clean["y"]),
                "audit": validated["audit_summary"], "cleaning": clean["audit_summary"],
                "removed_row_ids": clean["removed_row_ids"], "cleaning_version": 1,
                "cleaning_rule": "keep_first_exact_duplicate_reject_conflicting_labels",
                "seed": seed, "test_fraction": TEST_FRACTION, "n_folds": N_FOLDS,
                "rounding": f"floor(class_count * {TEST_FRACTION} + 0.5)",
                "split_checksums": checksums, "environment": environment_metadata()}
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return {"clean": clean, "dev": dev, "test": test, "folds": folds,
            "manifest": manifest, "manifest_path": manifest_path}


def run_training(csv_path=RAW_DATA_PATH, run_id=None, output_root=PROJECT_ROOT, *, configs=None, seed=SEED):
    """Chạy CV và lưu pipeline cuối trên dev; không tính điểm test trong hàm này.

    configs=None là lưới 28 cấu hình. Đối số configs phục vụ tests nhỏ có nguồn
    ghi rõ; không thay thế thực nghiệm chính. Output mới, không ghi đè run cũ.
    """
    from src.experiments import cross_validate, baseline_cross_validate, search_configurations
    from src.pipeline import WaterPipeline, save_pipeline, sha256_file
    from src.visualization import create_eda_figures
    output_root = Path(output_root)
    run_id = validate_run_id(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-E4"))
    run_path = output_root / "results" / "runs" / f"{run_id}.json"
    model_path = output_root / "artifacts" / run_id / "pipeline.json"
    metrics_directory = output_root / "results" / "metrics" / run_id
    figures = output_root / "results" / "figures" / run_id
    if any(path.exists() for path in (run_path, model_path.parent, metrics_directory, figures)):
        raise ValueError("run_id đã tồn tại; chọn ID mới, không ghi đè.")
    # Giữ ID độc quyền trước mọi thực nghiệm; lần lỗi vẫn phải dùng ID mới.
    model_path.parent.mkdir(parents=True, exist_ok=False)
    data = prepare_data(csv_path, output_root, seed)
    X = [data["clean"]["X"][index] for index in data["dev"]]
    y = [data["clean"]["y"][index] for index in data["dev"]]
    baseline = baseline_cross_validate(y, data["folds"])
    fixed_results = {}
    for experiment, strategy, standardize in (("E1", "mean", False), ("E2", "median", False), ("E3", "median", True)):
        config = {"strategy": strategy, "standardize": standardize,
                  "var_smoothing": DEFAULT_VAR_SMOOTHING, "variance_floor": VARIANCE_FLOOR}
        fixed_results[experiment] = cross_validate(X, y, data["folds"], config)
    search = search_configurations(X, y, data["folds"], configs=configs)
    selected = search["best"]["config"]
    pipeline = WaterPipeline(**selected).fit(X, y)
    environment = environment_metadata()
    metadata = {"csv_sha256": data["manifest"]["csv_sha256"],
                "data_manifest_sha256": sha256_file(data["manifest_path"]),
                "environment": environment, "selection": "balanced_accuracy_5fold_tolerance_0.005",
                "feature_columns": FEATURE_COLUMNS,
                 "source_verified": "csv_sha256" in data["manifest"]["source"],
                 "status": "development_run_uncommitted" if environment["dirty"] else
                           "unverified_source_run" if "csv_sha256" not in data["manifest"]["source"] else "candidate_run"}
    model_checksum = save_pipeline(pipeline, model_path, run_id, metadata)
    write_json(metrics_directory / "cv_results.json", {"E0": baseline, **fixed_results, "E4": search})
    create_eda_figures(X, y, figures)
    files = {}
    for folder in (metrics_directory, figures, model_path.parent):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                files[path.relative_to(output_root).as_posix()] = sha256_file(path)
    prior = sum(1 for label in y if label == 1) / len(y)
    manifest = {"run_id": run_id, "stage": "trained_not_tested", "model_sha256": model_checksum,
                "model_path": model_path.relative_to(output_root).as_posix(), "files": files,
                "csv_sha256": metadata["csv_sha256"],
                "data_manifest_sha256": metadata["data_manifest_sha256"],
                "selected_config": selected, "baseline_prior_1": prior,
                "environment": environment, "verification": "Automated tests must pass before final evaluation; not a final signed acceptance.",
                "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(run_path, manifest)
    return manifest


def main():
    """Huấn luyện chạy bằng python -m src.train; test có lệnh tách riêng."""
    parser = argparse.ArgumentParser(description="CV và fit GNB tự cài, chưa mở test")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--run-id")
    parser.add_argument("--prepare-only", action="store_true", help="Chỉ audit/chia tập, chưa fit")
    arguments = parser.parse_args()
    try:
        if arguments.prepare_only:
            data = prepare_data(arguments.data)
            print(json.dumps(data["manifest"], ensure_ascii=False, indent=2))
        else:
            print(json.dumps(run_training(arguments.data, arguments.run_id), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
