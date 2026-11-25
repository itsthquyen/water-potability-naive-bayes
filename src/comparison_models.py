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


def entropy(counts):
    """H = -sum p log2 p, bỏ số đếm 0; nút rỗng có H = 0."""
    total = sum(counts)
    result = 0.0
    for count in counts:
        if count:
            probability = count / total
            result -= probability * log2(probability)
    return result


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


class DecisionTree:
    """ID3 (gain) hoặc lõi C4.5 (gain ratio), thuộc tính danh nghĩa đã mã hóa.

    Mỗi thuộc tính chỉ dùng một lần trên đường đi. Dừng ở nút thuần, hết
    thuộc tính, hoặc không còn phép chia có ít nhất hai nhánh. Lá giữ tỷ lệ
    lớp; mã chưa xuất hiện khi học dùng phân bố của nút đang xét.
    """

    def __init__(self, criterion="gain"):
        if criterion not in ("gain", "gain_ratio"):
            raise ValueError("Tiêu chí phải là gain hoặc gain_ratio.")
        self.criterion = criterion

    def fit(self, X, y):
        rows, labels = _supervised(X, y)
        if any(value < 0 or value != int(value) for row in rows for value in row):
            raise ValueError("Cây danh nghĩa cần mã nguyên không âm.")
        self.n_features = len(rows[0])

        def grow(indices, columns):
            counts = [sum(labels[i] == c for i in indices) for c in (0, 1)]
            node = {"counts": counts}
            if not min(counts) or not columns:
                return node
            candidates = []
            initial = entropy(counts)
            for column in columns:
                groups = {}
                for index in indices:
                    groups.setdefault(int(rows[index][column]), []).append(index)
                if len(groups) < 2:
                    continue
                remainder = 0.0
                for group in groups.values():
                    part = [sum(labels[i] == c for i in group) for c in (0, 1)]
                    remainder += len(group) / len(indices) * entropy(part)
                gain = max(0.0, initial - remainder)
                split_info = entropy([len(group) for group in groups.values()])
                candidates.append((column, gain, gain / split_info, groups))
            if not candidates:
                return node
            if self.criterion == "gain_ratio":
                average = sum(item[1] for item in candidates) / len(candidates)
                candidates = [item for item in candidates if item[1] > 1e-12 and item[1] + 1e-12 >= average]
                if not candidates:
                    return node
                best = max(candidates, key=lambda item: (item[2], -item[0]))
            else:
                best = max(candidates, key=lambda item: (item[1], -item[0]))
            column, gain, ratio, groups = best
            remaining = [value for value in columns if value != column]
            node.update(column=column, gain=gain, gain_ratio=ratio,
                        children={str(value): grow(group, remaining) for value, group in sorted(groups.items())})
            return node

        self.root = grow(list(range(len(rows))), list(range(self.n_features)))
        return self

    def predict_with_scores(self, X):
        if not hasattr(self, "root"):
            raise ValueError("Cần fit cây trước.")
        labels, scores = [], []
        for row in _validate_matrix(X, self.n_features):
            if any(value is None or value < 0 or value != int(value) for value in row):
                raise ValueError("Cây danh nghĩa cần mã nguyên không âm.")
            node = self.root
            while "children" in node:
                child = node["children"].get(str(int(row[node["column"]])))
                if child is None:
                    break
                node = child
            counts = node["counts"]
            scores.append(counts[1] / sum(counts))
            labels.append(int(counts[1] > counts[0]))
        return labels, scores

    def to_state(self):
        return {"criterion": self.criterion, "n_features": self.n_features, "root": deepcopy(self.root)}


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


def sigmoid(value):
    """Sigmoid tính ổn định ở cả hai phía của 0."""
    if value >= 0:
        return 1 / (1 + exp(-value))
    exponential = exp(value)
    return exponential / (1 + exponential)


class LogisticRegression:
    """Tối ưu cross entropy + l2/2 * ||w||² bằng gradient toàn tập."""

    def __init__(self, epochs=400, learning_rate=0.1, l2=0.001):
        if type(epochs) is not int or epochs < 1 or not isfinite(learning_rate) or learning_rate <= 0 or not isfinite(l2) or l2 < 0:
            raise ValueError("Cấu hình Logistic không hợp lệ.")
        self.epochs, self.learning_rate, self.l2 = epochs, learning_rate, l2

    def fit(self, X, y):
        rows, labels = _supervised(X, y)
        weights = [0.0] * len(rows[0])
        bias = 0.0
        for _ in range(self.epochs):
            gradients, bias_gradient = [0.0] * len(weights), 0.0
            for row, label in zip(rows, labels):
                probability = sigmoid(sum(w * x for w, x in zip(weights, row)) + bias)
                error = probability - label
                bias_gradient += error / len(rows)
                for j, value in enumerate(row):
                    gradients[j] += error * value / len(rows)
            weights = [w - self.learning_rate * (g + self.l2 * w) for w, g in zip(weights, gradients)]
            bias -= self.learning_rate * bias_gradient
        if not all(isfinite(value) for value in weights + [bias]):
            raise ValueError("Logistic không hội tụ hữu hạn.")
        self.weights, self.bias = weights, bias
        return self

    def predict_with_scores(self, X):
        if not hasattr(self, "weights"):
            raise ValueError("Cần fit Logistic trước.")
        scores = []
        for row in _validate_matrix(X, len(self.weights)):
            if any(value is None for value in row):
                raise ValueError("Cần điền thiếu trước khi dự đoán Logistic.")
            scores.append(sigmoid(sum(w * x for w, x in zip(self.weights, row)) + self.bias))
        return [int(score > 0.5) for score in scores], scores

    def to_state(self):
        return {"weights": self.weights, "bias": self.bias, "epochs": self.epochs,
                "learning_rate": self.learning_rate, "l2": self.l2}


class LinearSVM:
    """SGD cho l2/2*||w||² + mean max(0,1-y*(w*x+b)), y thuộc {-1,+1}."""

    def __init__(self, epochs=100, learning_rate=0.01, l2=0.01, seed=42):
        if type(epochs) is not int or epochs < 1 or not isfinite(learning_rate) or learning_rate <= 0 or not isfinite(l2) or l2 <= 0 or type(seed) is not int:
            raise ValueError("Cấu hình SVM không hợp lệ.")
        self.epochs, self.learning_rate, self.l2, self.seed = epochs, learning_rate, l2, seed

    def fit(self, X, y):
        rows, labels = _supervised(X, y)
        weights, bias = [0.0] * len(rows[0]), 0.0
        indices = list(range(len(rows)))
        generator = random.Random(self.seed)
        for epoch in range(self.epochs):
            generator.shuffle(indices)
            rate = self.learning_rate / (1 + 0.1 * epoch)
            for index in indices:
                row, sign = rows[index], 2 * labels[index] - 1
                margin = sign * (sum(w * x for w, x in zip(weights, row)) + bias)
                weights = [(1 - rate * self.l2) * value for value in weights]
                if margin < 1:
                    weights = [w + rate * sign * value for w, value in zip(weights, row)]
                    bias += rate * sign
        if not all(isfinite(value) for value in weights + [bias]):
            raise ValueError("SVM không hội tụ hữu hạn.")
        self.weights, self.bias = weights, bias
        return self

    def predict_with_scores(self, X):
        if not hasattr(self, "weights"):
            raise ValueError("Cần fit SVM trước.")
        scores = []
        for row in _validate_matrix(X, len(self.weights)):
            if any(value is None for value in row):
                raise ValueError("Cần điền thiếu trước khi dự đoán SVM.")
            scores.append(sum(w * x for w, x in zip(self.weights, row)) + self.bias)
        return [int(score > 0) for score in scores], scores

    def to_state(self):
        return {"weights": self.weights, "bias": self.bias, "epochs": self.epochs,
                "learning_rate": self.learning_rate, "l2": self.l2, "seed": self.seed}
