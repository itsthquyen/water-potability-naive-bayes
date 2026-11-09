"""Rời rạc hóa chỉ tiêu số bằng tối đa ba khoảng học trên phần train."""

from copy import deepcopy
from math import isfinite

from .config import N_BINS
from .preprocessing import _finite_number, _validate_matrix


def _midpoint(left, right):
    """Cộng trước khi chia để tránh làm tròn hai lần ở vùng số cực nhỏ."""
    total = left + right
    if isfinite(total):
        return total / 2
    return left / 2 + right / 2  # Chia trước nếu tổng vượt miền float.


class Discretizer:
    """Học trung vị và ngưỡng từ các ô quan sát trong X; không đọc nhãn."""

    def __init__(self):
        self._state = None

    def fit(self, X):
        """Fit X (n, p) có thể thiếu; lỗi cột toàn thiếu hoặc ma trận sai."""
        rows = _validate_matrix(X)
        state = {"raw_min": [], "raw_max": [], "fill_values": [], "cut_points": []}
        for column in range(len(rows[0])):
            ordered = sorted(row[column] for row in rows if row[column] is not None)
            if not ordered:
                raise ValueError(f"Cột {column} thiếu hoàn toàn trong phần fit.")
            size = len(ordered)
            middle = size // 2
            median = ordered[middle] if size % 2 else _midpoint(ordered[middle - 1], ordered[middle])
            cuts = []
            for boundary in range(1, N_BINS):
                # ceil(n * boundary / N_BINS), tránh tính số thực cho vị trí.
                split = (size * boundary + N_BINS - 1) // N_BINS
                if split >= size or ordered[split - 1] == ordered[split]:
                    continue  # Không chia hai quan sát bằng nhau sang hai khoảng.
                left, right = ordered[split - 1], ordered[split]
                cut = _midpoint(left, right)
                if not left <= cut < right:
                    cut = left  # Hai số kề nhau có thể không có float ở giữa.
                if not cuts or cut > cuts[-1]:
                    cuts.append(cut)
            state["raw_min"].append(ordered[0])
            state["raw_max"].append(ordered[-1])
            state["fill_values"].append(median)
            state["cut_points"].append(cuts)
        self._state = state
        return self

    def _fitted(self):
        if self._state is None:
            raise ValueError("Phải fit trước khi rời rạc hóa.")
        return self._state

    @property
    def category_sizes(self):
        """Trả số khoảng của từng cột; cột hằng chỉ có một khoảng."""
        return [len(cuts) + 1 for cuts in self._fitted()["cut_points"]]

    def transform(self, X):
        """Trả X (n, p) mã khoảng nguyên 0-based; giá trị bằng ngưỡng vào khoảng dưới."""
        state = self._fitted()
        rows = _validate_matrix(X, len(state["cut_points"]))
        transformed = []
        for row in rows:
            categories = []
            for column, value in enumerate(row):
                if value is None:
                    value = state["fill_values"][column]
                category = 0
                for cut in state["cut_points"][column]:
                    if value <= cut:
                        break
                    category += 1
                categories.append(category)
            transformed.append(categories)
        return transformed

    def to_state(self):
        """Trả bản sao tham số đã fit theo thứ tự cột."""
        return deepcopy(self._fitted())

    @classmethod
    def from_state(cls, state):
        """Nạp tham số hữu hạn, đúng chiều; không fit lại hoặc thay ngưỡng."""
        keys = {"raw_min", "raw_max", "fill_values", "cut_points"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("Trạng thái rời rạc hóa sai cấu trúc.")
        cuts = state["cut_points"]
        if not isinstance(cuts, list) or not cuts:
            raise ValueError("Thiếu các cột ngưỡng rời rạc hóa.")
        validated = {}
        for key in keys - {"cut_points"}:
            values = state[key]
            if not isinstance(values, list) or len(values) != len(cuts):
                raise ValueError(f"{key} sai số cột.")
            validated[key] = [_finite_number(value, key) for value in values]
        validated["cut_points"] = []
        for column, boundaries in enumerate(cuts):
            lower, upper = validated["raw_min"][column], validated["raw_max"][column]
            if not lower <= validated["fill_values"][column] <= upper:
                raise ValueError("Trung vị nằm ngoài min/max train.")
            if not isinstance(boundaries, list) or len(boundaries) >= N_BINS:
                raise ValueError("Số ngưỡng không hợp lệ.")
            boundaries = [_finite_number(boundary, "Ngưỡng") for boundary in boundaries]
            if (any(not lower <= boundary < upper for boundary in boundaries)
                    or any(left >= right for left, right in zip(boundaries, boundaries[1:]))):
                raise ValueError("Ngưỡng phải tăng dần và nằm trong [min, max).")
            validated["cut_points"].append(boundaries)
        result = cls()
        result._state = validated
        return result
