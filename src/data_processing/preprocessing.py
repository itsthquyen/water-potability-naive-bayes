"""Kiểm tra ma trận số và tự chia dữ liệu phân tầng."""

import math
import hashlib
import json
from pathlib import Path
from numbers import Integral, Real
import random
from math import sqrt, isfinite

from ..config import N_FOLDS, SEED, TEST_FRACTION, PROJECT_ROOT
from ..artifacts import write_json, runtime_metadata


def _finite_number(value, name):
    """Đổi số thực hữu hạn thành float; từ chối bool, chuỗi và số tràn."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} phải là số thực, không phải bool.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} nằm ngoài phạm vi số thực.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} phải hữu hạn.")
    return number


def _validate_matrix(X, n_features=None):
    """Sao chép X (n mẫu, p cột) thành list; None/NaN là thiếu, còn lại hữu hạn.

    Ma trận rỗng, hàng rỗng/không đều, chuỗi, bool và vô hạn gây ValueError.
    Thứ tự cột và thứ tự hàng được giữ nguyên; không sửa đầu vào.
    """
    if isinstance(X, (str, bytes, dict)):
        raise ValueError("X phải là ma trận hai chiều.")
    try:
        rows = list(X)
    except TypeError as error:
        raise ValueError("X phải là ma trận hai chiều.") from error
    if not rows:
        raise ValueError("X không được rỗng.")
    matrix = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, dict)):
            raise ValueError(f"Hàng {row_index} phải là dãy số.")
        try:
            values = list(row)
        except TypeError as error:
            raise ValueError(f"Hàng {row_index} phải là dãy số.") from error
        if n_features is None:
            n_features = len(values)
        if not values or len(values) != n_features:
            raise ValueError(f"Hàng {row_index} có số cột không hợp lệ.")
        clean_row = []
        for column_index, value in enumerate(values):
            if value is None:
                clean_row.append(None)
            elif isinstance(value, Real) and not isinstance(value, bool) and value != value:
                clean_row.append(None)
            else:
                clean_row.append(_finite_number(value, f"Ô [{row_index}, {column_index}]"))
        matrix.append(clean_row)
    return matrix


def _group_labels(y, seed):
    """Kiểm tra y một chiều chỉ số nguyên 0/1, đủ hai lớp; shuffle cục bộ.

    Seed phải là số nguyên không âm, không phải bool. Trả hai list chỉ số;
    không sửa y hay trạng thái random toàn cục. Đầu vào sai gây ValueError.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed phải là số nguyên không âm.")
    if isinstance(y, (str, bytes, dict)):
        raise ValueError("y phải là dãy nhãn 0/1.")
    try:
        labels = list(y)
    except TypeError as error:
        raise ValueError("y phải là dãy nhãn 0/1.") from error
    groups = [[], []]
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
            raise ValueError("Nhãn phải là số nguyên 0 hoặc 1, không phải bool.")
        groups[int(label)].append(index)
    if not groups[0] or not groups[1]:
        raise ValueError("y phải có đủ hai lớp 0 và 1.")
    generator = random.Random(int(seed))
    for group in groups:
        generator.shuffle(group)
    return groups


def stratified_holdout(y, test_fraction=TEST_FRACTION, seed=SEED):
    """Trả (dev_indices, test_indices) phân tầng theo y, chỉ số bắt đầu từ 0.

    Số test mỗi lớp = floor(n_lớp * test_fraction + 0.5), không tự ép/clamp.
    Tỷ lệ phải hữu hạn trong (0, 1); mỗi lớp phải có ở cả dev và test, nếu
    không thì ValueError. Không bảo đảm dev đủ 5 fold: bên huấn luyện kiểm tra.
    Caller phải xử lý trùng/nhóm trước khi gọi. Kết quả sắp theo chỉ số gốc.
    """
    fraction = _finite_number(test_fraction, "test_fraction")
    if not 0 < fraction < 1:
        raise ValueError("test_fraction phải nằm trong (0, 1).")
    groups = _group_labels(y, seed)
    dev_indices, test_indices = [], []
    for group in groups:
        test_count = math.floor(len(group) * fraction + 0.5)
        if test_count < 1 or test_count >= len(group):
            raise ValueError("Mỗi lớp phải có ít nhất một mẫu ở dev và test.")
        test_indices.extend(group[:test_count])
        dev_indices.extend(group[test_count:])
    return sorted(dev_indices), sorted(test_indices)


def stratified_folds(y, n_folds=N_FOLDS, seed=SEED):
    """Trả list n_folds cặp (train_indices, val_indices), chỉ số gốc tăng dần.

    Nhóm và shuffle từng lớp rồi phân phối vòng tròn vào các fold. Mỗi mẫu
    validation đúng một lần; chênh lệch số mẫu cùng lớp giữa fold tối đa 1.
    n_folds là nguyên >=2 (không bool), mỗi lớp ít nhất n_folds mẫu để mọi
    train/validation đủ hai lớp; vi phạm gây ValueError. Xử lý trùng trước gọi.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds phải là số nguyên từ 2 trở lên.")
    groups = _group_labels(y, seed)
    for group in groups:
        if len(group) < n_folds:
            raise ValueError("Mỗi lớp phải có ít nhất n_folds mẫu.")
    validation_groups = [[] for _ in range(n_folds)]
    for group in groups:
        for position, index in enumerate(group):
            validation_groups[position % n_folds].append(index)
    all_indices = sorted(groups[0] + groups[1])
    folds = []
    for validation in validation_groups:
        validation_set = set(validation)
        train = [index for index in all_indices if index not in validation_set]
        folds.append((train, sorted(validation)))
    return folds


def _midpoint(left, right):
    """Cộng trước khi chia để tránh làm tròn hai lần ở vùng số cực nhỏ."""
    total = left + right
    if isfinite(total):
        return total / 2
    return left / 2 + right / 2


class NumericPreprocessor:
    """Học trung vị, mean và std ddof=0 trên train sau điền thiếu."""

    def fit(self, X):
        rows = _validate_matrix(X)
        medians, means, scales = [], [], []
        for column in zip(*rows):
            observed = sorted(value for value in column if value is not None)
            if not observed:
                raise ValueError("Cột toàn thiếu trong tập học.")
            size = len(observed)
            median = _midpoint(observed[(size - 1) // 2], observed[size // 2])
            filled = [median if value is None else value for value in column]
            mean = sum(value / len(filled) for value in filled)
            variance = sum((value - mean) ** 2 / len(filled) for value in filled)
            scale = sqrt(variance) if variance > 0 else 1.0
            if not isfinite(mean) or not isfinite(scale):
                raise ValueError("Thống kê chuẩn hóa vượt miền số hữu hạn.")
            medians.append(median)
            means.append(mean)
            scales.append(scale)
        self.state = {"medians": medians, "means": means, "scales": scales}
        return self

    def transform(self, X):
        if not hasattr(self, "state"):
            raise ValueError("Cần fit tiền xử lý trước.")
        state = self.state
        rows = _validate_matrix(X, len(state["medians"]))
        transformed = []
        for row in rows:
            values = [(state["medians"][j] if value is None else value) for j, value in enumerate(row)]
            normalized = [(value - state["means"][j]) / state["scales"][j] for j, value in enumerate(values)]
            if not all(isfinite(value) for value in normalized):
                raise ValueError("Dữ liệu chuẩn hóa không hữu hạn.")
            transformed.append(normalized)
        return transformed


def prepare_data(csv_path, output_root=PROJECT_ROOT, seed=SEED):
    """Audit CSV, loại trùng cùng nhãn, chặn mâu thuẫn và lưu split checksum.

    Không fit/EDA trên test; phép chia dùng ID trước lọc. Nếu manifest đã có,
    checksum khác sẽ dừng thay vì âm thầm thay phép chia chuẩn.
    """
    from src.data_processing.data_validation import read_csv_records, validate_records, clean_training_data
    from src.artifacts import sha256_file
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
                "split_checksums": checksums, "environment": runtime_metadata()}
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return {"clean": clean, "dev": dev, "test": test, "folds": folds,
            "manifest": manifest, "manifest_path": manifest_path}
