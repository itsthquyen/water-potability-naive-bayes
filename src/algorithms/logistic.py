"""Hồi quy Logistic tự tính sigmoid, gradient và cập nhật trọng số."""

from math import exp, isfinite
from ..preprocessing import _validate_matrix
from .common import _supervised


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
