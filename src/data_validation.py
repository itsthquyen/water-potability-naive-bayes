"""Đọc CSV nguyên trạng, kiểm tra từng mẫu và loại trùng trước chia dữ liệu."""

import csv
import io
import math
from numbers import Real
from pathlib import Path

from .config import FEATURE_COLUMNS, TARGET_COLUMN, MISSING_WARNING_COUNT


def _validate_columns(columns, training):
    """Kiểm tra tên cột chính xác; lỗi cấu trúc phải chặn cả đầu vào."""
    required = list(FEATURE_COLUMNS)
    if training:
        required.append(TARGET_COLUMN)
    allowed = required if training else required + ["sample_id", TARGET_COLUMN]
    if len(columns) != len(set(columns)):
        raise ValueError("Header có tên cột trùng.")
    missing = [name for name in required if name not in columns]
    extra = [name for name in columns if name not in allowed]
    if missing or extra:
        raise ValueError(f"Cột thiếu: {missing}; cột thừa: {extra}.")


def read_csv_records(path_or_text, *, from_text=False, training=False):
    """Đọc đường dẫn hoặc chuỗi CSV thành list dict có giá trị chuỗi nguyên trạng.

    CSV phân tách bằng dấu phẩy, UTF-8; chấp nhận đúng một BOM ở đầu file.
    Header phân biệt hoa/thường, không strip; thứ tự cột tùy ý. Training chỉ
    nhận 9 đặc trưng và Potability; prediction cho thêm sample_id/Potability.
    File rỗng, chỉ header, header sai, hàng trắng, thiếu/thừa ô hoặc CSV hỏng
    đều gây ValueError toàn file. Không suy diễn kiểu số hoặc ô thiếu ở đây.
    """
    try:
        if from_text:
            if not isinstance(path_or_text, str):
                raise ValueError("Nội dung CSV phải là chuỗi UTF-8 đã giải mã.")
            text = path_or_text
        else:
            text = Path(path_or_text).read_text(encoding="utf-8")
        if text.startswith("\ufeff"):
            text = text[1:]
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError("File CSV rỗng hoặc không có header.")
        _validate_columns(header, training)
        records = []
        for input_row_id, cells in enumerate(reader):
            if len(cells) != len(header):
                raise ValueError(f"Bản ghi {input_row_id}: số ô không khớp header.")
            records.append(dict(zip(header, cells)))
        if not records:
            raise ValueError("CSV không có bản ghi dữ liệu.")
        return records
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Không đọc được CSV: {error}") from error


def _parse_feature(value, column):
    """Đổi ô số hữu hạn sang float; chỉ None, chuỗi rỗng/NaN là thiếu."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in ("", "NaN"):
            return None
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{column}: phải là số hữu hạn hoặc ô thiếu.")
    try:
        numeric_value = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{column}: không phải số hữu hạn.") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{column}: không phải số hữu hạn.")
    return numeric_value


def _parse_label(value):
    """Nhãn bắt buộc là số 0/1 hoặc chuỗi chính xác 0/1 sau strip."""
    if isinstance(value, str):
        value = value.strip()
        if value not in ("0", "1"):
            raise ValueError("Potability: nhãn bắt buộc là 0 hoặc 1.")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, Real) or value not in (0, 1):
        raise ValueError("Potability: nhãn bắt buộc là 0 hoặc 1.")
    return int(value)


def _audit_rows(X, y, row_ids):
    """Tự đếm trên các hàng hợp lệ; nhóm trùng theo 9 ô gốc, không theo ID."""
    missing_counts = {name: 0 for name in FEATURE_COLUMNS}
    class_counts = {0: 0, 1: 0}
    groups = {}
    for index, features in enumerate(X):
        for column_index, name in enumerate(FEATURE_COLUMNS):
            if features[column_index] is None:
                missing_counts[name] += 1
        if y is not None:
            class_counts[y[index]] += 1
        key = tuple(features)
        groups.setdefault(key, []).append(row_ids[index])
    return {
        "missing_counts": missing_counts,
        "class_counts": class_counts if y is not None else {},
        "duplicate_groups": [group for group in groups.values() if len(group) > 1],
    }


def validate_records(records, training=False):
    """Kiểm tra list dict mà không sửa đầu vào, giữ ID vị trí 0-based.

    Trả dict X (n_valid, 9) float/None theo FEATURE_COLUMNS, y là list nhãn
    khi training hoặc None khi prediction, row_ids và valid_records nguyên
    giá trị. errors gồm {input_row_id,error}, warnings gồm {input_row_id,warning}.
    Lỗi giá trị chỉ loại hàng đó; thiếu toàn bộ 9 ô cũng là lỗi. Sai kiểu list,
    dict hoặc sai bộ cột gây ValueError toàn đầu vào. Potability prediction bị
    bỏ qua có cảnh báo, không kiểm nhãn và không trở thành đặc trưng. sample_id
    giữ nguyên kể cả chuỗi có số 0 đầu; không dùng làm ID nội bộ hay đặc trưng.
    audit_summary đếm missing/lớp/trùng trên hàng hợp lệ, không dùng hàng lỗi.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records phải là list dict không rỗng.")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Mỗi bản ghi phải là dict.")
        _validate_columns(list(record), training)
    X, row_ids, valid_records, errors, warnings = [], [], [], [], []
    y = [] if training else None
    for input_row_id, record in enumerate(records):
        try:
            features = [_parse_feature(record[name], name) for name in FEATURE_COLUMNS]
            missing_count = sum(value is None for value in features)
            if missing_count == len(FEATURE_COLUMNS):
                raise ValueError("Thiếu toàn bộ 9 chỉ tiêu.")
            label = _parse_label(record[TARGET_COLUMN]) if training else None
        except ValueError as error:
            errors.append({"input_row_id": input_row_id, "error": str(error)})
            continue
        X.append(features)
        row_ids.append(input_row_id)
        valid_records.append(dict(record))
        if training:
            y.append(label)
        if missing_count >= MISSING_WARNING_COUNT:
            warnings.append({"input_row_id": input_row_id,
                             "warning": f"Thiếu {missing_count}/9 chỉ tiêu; cần điền thiếu."})
        if not training and TARGET_COLUMN in record:
            warnings.append({"input_row_id": input_row_id,
                             "warning": "Bỏ qua Potability; không dùng nhãn làm đặc trưng."})
    audit_summary = _audit_rows(X, y, row_ids)
    audit_summary.update(input_count=len(records), valid_count=len(X), invalid_count=len(errors))
    return {"X": X, "y": y, "row_ids": row_ids, "valid_records": valid_records,
            "errors": errors, "warnings": warnings, "audit_summary": audit_summary}


def clean_training_data(X, y, row_ids):
    """Loại trùng đầu vào/nhãn trước split, giữ hàng xuất hiện đầu tiên.

    X gồm n hàng 9 ô số/chuỗi/None, y gồm n nhãn 0/1, row_ids gồm n ID nguyên
    không âm duy nhất. Chuẩn hóa ô thiếu thành None trước tạo khóa tuple 9 ô;
    không điền thiếu và không thay đổi dữ liệu gốc. Cùng khóa khác nhãn gây
    ValueError, không tự chọn nhãn. Kết quả có X/y/row_ids đã lọc,
    removed_row_ids và duplicate_groups (mỗi nhóm gồm cả ID giữ và ID bỏ).
    audit_summary đếm missing/lớp sau lọc, số hàng trước/sau và các nhóm trùng
    trước lọc. Đầu vào rỗng, sai shape, ID hoặc giá trị gây ValueError.
    """
    if not X or len(X) != len(y) or len(X) != len(row_ids):
        raise ValueError("X, y, row_ids phải không rỗng và cùng số hàng.")
    if any(type(row_id) is not int or row_id < 0 for row_id in row_ids):
        raise ValueError("row_ids phải là số nguyên không âm.")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids phải duy nhất.")
    cleaned_X, cleaned_y, kept_ids, removed_ids, groups = [], [], [], [], {}
    for features, label, row_id in zip(X, y, row_ids):
        if len(features) != len(FEATURE_COLUMNS):
            raise ValueError("Mỗi hàng X phải có đúng 9 chỉ tiêu.")
        normalized = [_parse_feature(value, name) for name, value in zip(FEATURE_COLUMNS, features)]
        if all(value is None for value in normalized):
            raise ValueError("Thiếu toàn bộ 9 chỉ tiêu.")
        label = _parse_label(label)
        key = tuple(normalized)
        if key in groups:
            group_label, group_ids = groups[key]
            if group_label != label:
                raise ValueError(f"Cùng đặc trưng nhưng khác nhãn: ID {group_ids[0]} và {row_id}.")
            group_ids.append(row_id)
            removed_ids.append(row_id)
        else:
            groups[key] = (label, [row_id])
            cleaned_X.append(normalized)
            cleaned_y.append(label)
            kept_ids.append(row_id)
    duplicate_groups = [ids for _, ids in groups.values() if len(ids) > 1]
    audit_summary = _audit_rows(cleaned_X, cleaned_y, kept_ids)
    audit_summary.update(input_count=len(X), output_count=len(cleaned_X),
                         removed_count=len(removed_ids), duplicate_groups=duplicate_groups)
    return {"X": cleaned_X, "y": cleaned_y, "row_ids": kept_ids,
            "removed_row_ids": removed_ids, "duplicate_groups": duplicate_groups,
            "audit_summary": audit_summary}
