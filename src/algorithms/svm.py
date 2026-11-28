"""SVM tuyến tính tự tối ưu hinge loss bằng SGD."""

from math import isfinite
import random
from ..preprocessing import _validate_matrix
from .common import _supervised


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
