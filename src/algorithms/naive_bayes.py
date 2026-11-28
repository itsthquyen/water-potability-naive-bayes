"""Naive Bayes đếm tần suất theo công thức giáo trình cho dữ liệu rời rạc."""

from copy import deepcopy
from numbers import Integral

from ..config import CLASS_LABELS


def _list(values, name):
    """Đọc một dãy, từ chối dữ liệu vô hướng, chuỗi hoặc dict."""
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(f"{name} phải là dãy.")
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(f"{name} phải là dãy.") from error


def _rows(X, category_sizes):
    """Sinh từng hàng mã khoảng nguyên; kiểm tra số cột và miền giá trị."""
    rows = _list(X, "X")
    if not rows:
        raise ValueError("X không được rỗng.")
    for index, row in enumerate(rows):
        values = _list(row, f"Dòng {index}")
        if len(values) != len(category_sizes):
            raise ValueError(f"Dòng {index} sai số cột.")
        for value, size in zip(values, category_sizes):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
                raise ValueError(f"Dòng {index} có mã khoảng ngoài miền đã fit.")
        yield values


class NaiveBayes:
    """P(c) = n_c/n, P(A_j=v | c) = count(c,j,v)/n_c; không cộng giả mẫu."""

    def __init__(self):
        self._state = None

    def fit(self, X, y, category_sizes):
        """Đếm nhãn và từng mức của từng thuộc tính; X đã được rời rạc hóa."""
        sizes = _list(category_sizes, "category_sizes")
        if not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Mỗi cột phải có số khoảng nguyên dương.")
        rows = list(_rows(X, sizes))
        labels = _list(y, "y")
        if len(rows) != len(labels):
            raise ValueError("X và y phải có cùng số mẫu.")
        class_counts = [0, 0]
        feature_counts = [[[0] * size for size in sizes] for _ in CLASS_LABELS]
        for row, label in zip(rows, labels):
            if isinstance(label, bool) or not isinstance(label, Integral) or label not in CLASS_LABELS:
                raise ValueError("Nhãn chỉ nhận số nguyên 0 hoặc 1.")
            class_counts[label] += 1
            for column, value in enumerate(row):
                feature_counts[label][column][value] += 1
        if 0 in class_counts:
            raise ValueError("Phần huấn luyện phải có cả hai lớp.")
        self._set_state({"category_sizes": sizes, "class_counts": class_counts, "feature_counts": feature_counts})
        return self

    def _set_state(self, state):
        """Chuẩn bị mẫu số N × n_c^d từ bảng đếm đã kiểm tra, dùng cho mọi mẫu."""
        class_counts = state["class_counts"]
        total = sum(class_counts)
        feature_count = len(state["category_sizes"])
        denominators = tuple(total * count ** feature_count for count in class_counts)
        self._state = (state, denominators)

    def _fitted(self):
        if self._state is None:
            raise ValueError("Phải fit Naive Bayes trước.")
        return self._state

    def predict_with_proba(self, X):
        """Trả (nhãn n mẫu, xác suất n×2); giữ nguyên tần suất 0, hòa chọn 0.

        Điểm mỗi lớp là prior nhân các xác suất có điều kiện. Tính tử và mẫu
        bằng số nguyên, rồi quy đồng để so sánh chính xác trước khi chia.
        Nếu cả hai điểm bằng 0 thì không có hậu nghiệm theo mô hình.
        """
        state, (denominator_0, denominator_1) = self._fitted()
        labels, probabilities = [], []
        for row_index, row in enumerate(_rows(X, state["category_sizes"])):
            numerators = []
            for label in CLASS_LABELS:
                numerator = state["class_counts"][label]  # Tử số của prior n_c/N.
                for column, value in enumerate(row):
                    numerator *= state["feature_counts"][label][column][value]
                numerators.append(numerator)
            score_0 = numerators[0] * denominator_1
            score_1 = numerators[1] * denominator_0
            score_total = score_0 + score_1
            if score_total == 0:
                raise ValueError(f"Dòng {row_index}: cả hai điểm Naive Bayes bằng 0; không thể gán nhãn.")
            labels.append(1 if score_1 > score_0 else 0)
            probabilities.append([score_0 / score_total, score_1 / score_total])
        return labels, probabilities

    def predict(self, X):
        """Trả nhãn MAP, lớp 0 khi hai điểm dương bằng nhau."""
        return self.predict_with_proba(X)[0]

    def predict_proba(self, X):
        """Trả hai xác suất theo thứ tự lớp [0, 1]."""
        return self.predict_with_proba(X)[1]

    def to_state(self):
        """Trả bản sao bảng đếm theo [lớp][thuộc tính][khoảng]."""
        state, _ = self._fitted()
        return deepcopy(state)

    @classmethod
    def from_state(cls, state):
        """Kiểm tra số chiều, kiểu đếm và tổng đếm từng cột bằng số mẫu lớp."""
        if not isinstance(state, dict) or set(state) != {"category_sizes", "class_counts", "feature_counts"}:
            raise ValueError("Trạng thái Naive Bayes sai cấu trúc.")
        sizes, counts, features = state["category_sizes"], state["class_counts"], state["feature_counts"]
        if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Số khoảng không hợp lệ.")
        if not isinstance(counts, list) or len(counts) != 2 or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("Cần hai số đếm lớp nguyên dương.")
        if not isinstance(features, list) or len(features) != 2:
            raise ValueError("Bảng đếm phải có hai lớp.")
        for class_size, columns in zip(counts, features):
            if not isinstance(columns, list) or len(columns) != len(sizes):
                raise ValueError("Bảng đếm sai số cột.")
            for size, bins in zip(sizes, columns):
                if (not isinstance(bins, list) or len(bins) != size
                        or any(type(value) is not int or value < 0 for value in bins)
                        or sum(bins) != class_size):
                    raise ValueError("Bảng đếm không khớp số khoảng hoặc số mẫu lớp.")
        model = cls()
        model._set_state(deepcopy(state))
        return model
