"""Kiểm tra ma trận số và tự chia dữ liệu phân tầng."""

import math
from numbers import Integral, Real
import random

from .config import N_FOLDS, SEED, TEST_FRACTION


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
