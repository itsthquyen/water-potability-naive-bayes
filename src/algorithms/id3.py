"""ID3 tự xây cây theo Information Gain trên mã thuộc tính danh nghĩa."""

from copy import deepcopy
from ..data_processing.preprocessing import _validate_matrix
from .common import _supervised, entropy


class ID3:
    """ID3, thuộc tính danh nghĩa đã mã hóa.

    Mỗi thuộc tính chỉ dùng một lần trên đường đi. Dừng ở nút thuần, hết
    thuộc tính, hoặc không còn phép chia có ít nhất hai nhánh. Lá giữ tỷ lệ
    lớp; mã chưa xuất hiện khi học dùng phân bố của nút đang xét.
    """

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
        return {"n_features": self.n_features, "root": deepcopy(self.root)}
