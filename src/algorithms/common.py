"""Kiểm tra đầu vào và entropy dùng chung cho các bộ phân lớp."""

from math import log2
from numbers import Integral
from ..data_processing.preprocessing import _validate_matrix


def _supervised(X, y):
    """Sao chép X hữu hạn không thiếu và y nhị phân, cùng số hàng dương."""
    rows = _validate_matrix(X)
    labels = list(y)
    if len(rows) != len(labels):
        raise ValueError("X và y phải cùng số mẫu.")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v not in (0, 1) for v in labels):
        raise ValueError("Nhãn phải là số nguyên 0 hoặc 1.")
    if any(value is None for row in rows for value in row):
        raise ValueError("Cần điền thiếu trước khi fit bộ phân lớp.")
    return rows, [int(v) for v in labels]


def entropy(counts):
    """H = -sum p log2 p, bỏ số đếm 0; nút rỗng có H = 0."""
    total = sum(counts)
    result = 0.0
    for count in counts:
        if count:
            probability = count / total
            result -= probability * log2(probability)
    return result
