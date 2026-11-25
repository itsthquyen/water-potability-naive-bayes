"""Năm bộ phân lớp đối chiếu tự cài cho Naive Bayes.

ID3 và C4.5 nhận mã danh nghĩa không thiếu. Phần C4.5 triển khai tiêu chí
gain ratio với điều kiện gain tối thiểu bằng gain trung bình trên các phép
chia hợp lệ; cây được giữ chưa cắt tỉa. KNN, Logistic và SVM nhận số đã
điền trung vị và chuẩn hóa theo tập học. Không dùng thư viện học máy.
"""

from copy import deepcopy
from math import exp, isfinite, log2, sqrt
from numbers import Integral
import random

from .discretization import _midpoint
from .preprocessing import _validate_matrix


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
