"""Xuất hình Weka từ một run đã đánh giá; liên kết hình và ARFF bằng SHA-256."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.config import DEMO_RUN_ID
from src.export_weka import export_weka
from src.pipeline import sha256_file
from src.train import validate_run_id

FIGURES = (
    "01-data-and-class", "02-development-missing", "03-development-processed",
    "04-test-partition", "05-roc", "06-errors", "07-development-class", "08-ph-processed",
)


def render_weka(run_id, weka_home, java, javac):
    """Dựng các panel thật từ ARFF; chỉ công bố manifest sau khi đủ hình."""
    validate_run_id(run_id)
    jar = weka_home / "weka.jar"
    if not jar.is_file():
        raise ValueError("Thư mục Weka không có weka.jar.")
    if java is None:
        packaged = sorted(weka_home.glob("jre/**/bin/java.exe"))
        java = packaged[0] if packaged else shutil.which("java")
    if not java or not javac:
        raise ValueError("Cần Java và javac; có thể cung cấp đường dẫn bằng --java/--javac.")
    run = json.loads((ROOT / "results" / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
    folder = Path(__file__).resolve().parent
    evidence = folder / "evidence"
    evidence.mkdir(exist_ok=True)
    classes = ROOT / "build" / "weka"
    classes.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="weka-", dir=evidence.parent) as temp:
        staging = Path(temp)
        export_weka(run_id, staging)
        subprocess.run(["rtk", "proxy", str(javac), "-encoding", "UTF-8", "-cp", str(jar),
                        "-d", str(classes), str(folder / "WekaEvidence.java")], check=True)
        subprocess.run(["rtk", "proxy", str(java), "--enable-native-access=ALL-UNNAMED",
                        "--add-opens=java.base/java.lang=ALL-UNNAMED", "-Dweka.packageManager.loadPackages=false",
                        "-cp", str(classes) + ";" + str(jar), "WekaEvidence", str(staging)], check=True)
        pngs = [staging / f"{name}.png" for name in FIGURES]
        if any(not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n" for path in pngs):
            raise ValueError("Weka không xuất đủ ảnh PNG.")
        assets = sorted(staging.glob("*.arff")) + pngs
        manifest = {
            "run_id": run_id, "model_sha256": run["model_sha256"], "csv_sha256": run["csv_sha256"],
            "files": {path.name: sha256_file(path) for path in assets},
        }
        marker = staging / "evidence_manifest.json"
        marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in assets:
            path.replace(evidence / path.name)
        marker.replace(evidence / marker.name)
    print(f"{run_id}: {len(FIGURES)} hình Weka, ARFF và manifest → {evidence}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEMO_RUN_ID)
    parser.add_argument("--weka-home", type=Path, default=Path("C:/Program Files/Weka-3-8-7"))
    parser.add_argument("--java", type=Path)
    parser.add_argument("--javac", default=shutil.which("javac"))
    args = parser.parse_args()
    try:
        render_weka(args.run_id, args.weka_home, args.java, args.javac)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Lỗi dựng hình Weka: {error}\n")


if __name__ == "__main__":
    main()
