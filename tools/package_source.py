"""Tạo thư mục nộp chỉ chứa source theo whitelist, không quét/copy toàn repo."""

from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'src/__init__.py', 'src/config.py', 'src/data_validation.py',
    'src/preprocessing.py', 'src/discretization.py', 'src/metrics.py',
    'src/artifacts.py', 'src/six_experiments.py', 'src/eda.py', 'src/plots.py',
    'src/download_data.py', 'src/ui_theme.py',
    'src/algorithms/__init__.py', 'src/algorithms/common.py',
    'src/algorithms/naive_bayes.py', 'src/algorithms/id3.py', 'src/algorithms/c45.py',
    'src/algorithms/knn.py', 'src/algorithms/logistic.py', 'src/algorithms/svm.py',
    'app/comparison_app.py', 'app/assets/comparison.css',
    'weka/DirectFrequencyNB.java', 'weka/SixModelEvidence.java', 'weka/run_six_weka.py',
    'tests/test_naive_bayes.py', 'tests/test_c45.py', 'tests/test_comparison_models.py',
    'tests/test_preprocessing.py', 'tests/test_data_validation.py', 'tests/test_metrics.py',
    'tests/test_eda.py', 'tests/test_comparison_app.py',
    'pytest.ini',
]


def main():
    destination = ROOT / 'output/source/BaiTapLon_Nhom11_Source'
    mapping = {relative: ROOT / relative for relative in FILES}
    mapping['README.md'] = ROOT / 'DOCS/README_NOP_SOURCE.md'
    expected = set(mapping) | {'requirements.txt'}
    # Không xóa file lạ hoặc công việc người dùng thêm vào thư mục đã bàn giao.
    if destination.exists():
        unexpected = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} - expected
        if unexpected:
            raise ValueError('Thư mục đích có file ngoài whitelist: '+repr(sorted(unexpected)))
    destination.mkdir(parents=True, exist_ok=True)
    for relative, source in mapping.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
    requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    wanted = ('pandas==', 'matplotlib==', 'streamlit==', 'pytest==')
    (destination / 'requirements.txt').write_text('\n'.join(line for line in requirements if line.startswith(wanted))+'\n', encoding='utf-8')
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    assert actual == expected
    assert all(Path(n).suffix in ('.py', '.java', '.css', '.md', '.txt', '.ini') for n in actual)
    manifest = {'kind': 'source-only', 'files': {n: sha256((destination/n).read_bytes()).hexdigest() for n in sorted(actual)}}
    manifest_path = destination.parent / (destination.name+'.manifest.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    archive_path = destination.with_suffix('.zip')
    with ZipFile(archive_path, 'w', ZIP_DEFLATED) as archive:
        for relative in sorted(actual):
            archive.write(destination/relative, arcname=destination.name+'/'+relative)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(actual)
        for relative, expected_hash in manifest['files'].items():
            assert sha256(archive.read(destination.name+'/'+relative)).hexdigest() == expected_hash
    print(json.dumps({'folder': str(destination), 'zip': str(archive_path), 'files': len(actual),
                      'reports_data_models_cache': 'excluded'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
