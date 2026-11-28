"""Đọc/ghi kết quả và checksum; không chứa thuật toán học máy."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re


def write_json(path, value):
    """Ghi JSON UTF-8, từ chối NaN/Infinity để kết quả có thể kiểm chứng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256_file(path):
    """Tính SHA-256 trên byte tệp; không sửa dữ liệu nguồn."""
    if not isinstance(path, (str, Path)) or not str(path).strip() or Path(path).is_dir():
        raise ValueError('Đường dẫn phải trỏ tới một tệp.')
    digest = sha256()
    try:
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(65536), b''):
                digest.update(block)
    except OSError as error:
        raise ValueError(f'Không đọc được tệp: {path}') from error
    return digest.hexdigest()


def validate_run_id(run_id):
    """Chặn ID rỗng, đường dẫn và ký tự lạ; tránh ghi đè lần chạy khác."""
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('run_id chỉ gồm chữ/số, gạch dưới/gạch ngang, dài tối đa 80.')
    return run_id


def runtime_metadata():
    """Thông tin thực thi tối thiểu, chạy được cả khi máy không có Git/RTK."""
    return {'python': platform.python_version(), 'platform': platform.platform()}
