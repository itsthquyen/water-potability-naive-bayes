"""Biên dịch và chạy sáu bộ phân lớp Weka trên phép chia của Python."""

import argparse
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import evaluate_predictions
from src.artifacts import sha256_file
from src.six_experiments import ALGORITHMS, RUN_ID, write_json


def execute(command):
    """Dùng RTK nếu máy có; mã nộp vẫn chạy được trên máy chỉ cài Java/Python."""
    prefix = ["rtk", "proxy"] if shutil.which("rtk") else []
    subprocess.run(prefix + command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--weka-home", type=Path, default=Path("C:/Program Files/Weka-3-8-7"))
    args = parser.parse_args()
    from src.artifacts import validate_run_id
    validate_run_id(args.run_id)
    folder = ROOT / "results/comparison" / args.run_id
    if not (folder / "summary.json").is_file():
        parser.error("Cần chạy src.six_experiments trước khi đối chiếu Weka.")
    build = ROOT / "build/six"
    build.mkdir(parents=True, exist_ok=True)
    jar = build / "simpleEducationalLearningSchemes.jar"
    package_url = "https://prdownloads.sourceforge.net/weka/simpleEducationalLearningSchemes1.0.1.zip?download"
    if not jar.is_file():
        with urlopen(Request(package_url, headers={"User-Agent": "water-potability-coursework"}), timeout=60) as response:
            data = response.read()
        with ZipFile(BytesIO(data)) as archive:
            name = next(name for name in archive.namelist() if name.endswith("simpleEducationalLearningSchemes.jar"))
            jar.write_bytes(archive.read(name))
    java_candidates = sorted(args.weka_home.glob("jre/**/bin/java.exe"))
    java = str(java_candidates[0]) if java_candidates else shutil.which("java")
    javac = shutil.which("javac")
    weka_jar = args.weka_home / "weka.jar"
    if not java or not javac or not weka_jar.is_file():
        parser.error("Cần Weka, Java và javac trên máy.")
    classpath = os.pathsep.join((str(build), str(jar), str(weka_jar)))
    execute([javac, "-encoding", "UTF-8", "-cp", classpath, "-d", str(build),
             str(ROOT / "weka/DirectFrequencyNB.java"), str(ROOT / "weka/SixModelEvidence.java")])
    execute([java, "--enable-native-access=ALL-UNNAMED", "--add-opens=java.base/java.lang=ALL-UNNAMED",
                    "-Dweka.packageManager.loadPackages=false", "-cp", classpath,
                    "SixModelEvidence", str(folder / "weka")])
    python_summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    summary = {}
    for algorithm in ALGORITHMS:
        with (folder / "weka" / f"{algorithm}_weka_predictions.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(python_summary['test_ids']):
            raise ValueError("Số mẫu test Weka không đúng phép chia đã chốt.")
        missing = [row for row in rows if not row["predicted"]]
        matrix = [[0, 0, 0], [0, 0, 0]]
        for row in rows:
            predicted = int(row["predicted"]) if row["predicted"] else 2
            matrix[int(row["actual"])][predicted] += 1
        result = {"accuracy": sum(row["actual"] == row["predicted"] for row in rows) / len(rows),
                  "coverage": (len(rows) - len(missing)) / len(rows), "unclassified": len(missing),
                  "matrix_with_unclassified": matrix}
        if not missing:
            result["metrics"] = evaluate_predictions([int(row["actual"]) for row in rows],
                [int(row["predicted"]) for row in rows], [float(row["score_1"]) for row in rows])
        summary[algorithm] = result
        if algorithm == 'c45':
            with (folder / 'c45_predictions.csv').open(encoding='utf-8', newline='') as file:
                own = list(csv.DictReader(file))
            result['python_agreement'] = {
                'compared': len(rows),
                'different_labels': sum(a['predicted'] != b['predicted'] for a, b in zip(own, rows)),
                'max_score_difference': max(abs(float(a['score_1']) - float(b['score_1'])) for a, b in zip(own, rows)),
            }
            fold_metrics = []
            fold_agreement = []
            for fold_id in range(5):
                with (folder / 'weka' / f'c45_fold_{fold_id}_weka_predictions.csv').open(encoding='utf-8', newline='') as file:
                    fold_rows = list(csv.DictReader(file))
                fold_metrics.append(evaluate_predictions([int(r['actual']) for r in fold_rows],
                    [int(r['predicted']) for r in fold_rows], [float(r['score_1']) for r in fold_rows]))
                with (folder / f'c45_fold_{fold_id}_predictions.csv').open(encoding='utf-8', newline='') as file:
                    python_fold = list(csv.DictReader(file))
                if len(python_fold) != len(fold_rows):
                    raise ValueError('Số dòng validation Python/Weka khác nhau.')
                fold_agreement.append({'fold': fold_id,
                    'compared': len(fold_rows),
                    'different_labels': sum(a['predicted'] != b['predicted'] for a, b in zip(python_fold, fold_rows)),
                    'max_score_difference': max(abs(float(a['score_1']) - float(b['score_1'])) for a, b in zip(python_fold, fold_rows))})
            result['folds'] = fold_metrics
            result['fold_agreement'] = fold_agreement
    write_json(folder / "weka_summary.json", summary)
    write_json(folder / "weka/source.json", {"weka_version": "3.8.7", "weka_jar_sha256": sha256_file(weka_jar),
        "educational_package_url": package_url, "educational_package_jar_sha256": sha256_file(jar)})
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob("*")
                         if p.is_file() and p.name != "manifest.json"}
    write_json(folder / "manifest.json", manifest)
    print("Weka summary and checksums updated.")


if __name__ == "__main__":
    main()
