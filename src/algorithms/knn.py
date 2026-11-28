"""KNN tự tính khoảng cách Euclid và bỏ phiếu k láng giềng."""

from ..data_processing.preprocessing import _validate_matrix
from .common import _supervised


class KNearestNeighbors:
    """KNN Euclid, bỏ căn vì thứ tự không đổi; bỏ phiếu đều, hòa chọn 0."""

    def __init__(self, k=5):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k phải là số nguyên dương.")
        self.k = k

    def fit(self, X, y):
        rows, labels = _supervised(X, y)
        if self.k > len(rows):
            raise ValueError("k không được lớn hơn số mẫu học.")
        self.rows, self.labels = rows, labels
        return self

    def predict_with_scores(self, X):
        if not hasattr(self, "rows"):
            raise ValueError("Cần fit KNN trước.")
        labels, scores = [], []
        for row in _validate_matrix(X, len(self.rows[0])):
            if any(value is None for value in row):
                raise ValueError("Cần điền thiếu trước khi dự đoán KNN.")
            distances = []
            for index, other in enumerate(self.rows):
                distance = sum((value - match) ** 2 for value, match in zip(row, other))
                distances.append((distance, index))
            neighbors = sorted(distances)[:self.k]
            votes = sum(self.labels[index] for _, index in neighbors)
            labels.append(int(2 * votes > self.k))
            scores.append(votes / self.k)
        return labels, scores

    def to_state(self):
        return {"k": self.k, "rows": self.rows, "labels": self.labels}
