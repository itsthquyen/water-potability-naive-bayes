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

from src.data_processing.preprocessing import prepare_data
from src.artifacts import validate_run_id

from src.config import (PROJECT_ROOT, RAW_DATA_PATH, SEED, N_FOLDS, TEST_FRACTION,
                        FEATURE_COLUMNS)


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








def run_training(csv_path=RAW_DATA_PATH, run_id=None, output_root=PROJECT_ROOT, *, seed=SEED):
    """Chạy CV và lưu pipeline cuối trên dev; không tính điểm test trong hàm này.

    Cấu hình cố định trước CV; không chọn số khoảng theo test.
    Output mới, không ghi đè run cũ.
    """
    from src.experiments import cross_validate, baseline_cross_validate
    from src.pipeline import WaterPipeline, save_pipeline, sha256_file
    from src.visualization import create_eda_figures
    output_root = Path(output_root)
    run_id = validate_run_id(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-NB"))
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
    cv = cross_validate(X, y, data["folds"])
    configuration = cv["config"]
    pipeline = WaterPipeline().fit(X, y)
    environment = environment_metadata()
    metadata = {"csv_sha256": data["manifest"]["csv_sha256"],
                "data_manifest_sha256": sha256_file(data["manifest_path"]),
                "environment": environment, "configuration": configuration,
                "feature_columns": FEATURE_COLUMNS,
                 "source_verified": "csv_sha256" in data["manifest"]["source"],
                 "status": "development_run_uncommitted" if environment["dirty"] else
                           "unverified_source_run" if "csv_sha256" not in data["manifest"]["source"] else "candidate_run"}
    model_checksum = save_pipeline(pipeline, model_path, run_id, metadata)
    write_json(metrics_directory / "cv_results.json", {"baseline": baseline, "naive_bayes": cv})
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
                "configuration": configuration, "baseline_prior_1": prior,
                "environment": environment, "verification": "Automated tests must pass before final evaluation; not a final signed acceptance.",
                "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(run_path, manifest)
    return manifest


def main():
    """Huấn luyện chạy bằng python -m src.train; test có lệnh tách riêng."""
    parser = argparse.ArgumentParser(description="CV và fit Naive Bayes đếm tần suất, chưa mở test")
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
