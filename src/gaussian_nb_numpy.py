"""Gaussian Naive Bayes tự cài; mọi cột xác suất theo thứ tự lớp [0, 1]."""

from copy import deepcopy
import math
from numbers import Real

from .config import DEFAULT_VAR_SMOOTHING, VARIANCE_FLOOR


def _finite_number(value, name):
    """Đổi số thực hữu hạn sang float; từ chối bool, chuỗi và tràn số."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} phải là số thực, không phải bool.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} bị tràn số.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} phải hữu hạn; phép tính có thể đã tràn số.")
    return result


def _matrix(X, n_features=None):
    """Sao chép ma trận số hữu hạn (n, d); n, d > 0, không chấp nhận hàng lệch."""
    if isinstance(X, (str, bytes, dict)):
        raise ValueError("X phải là ma trận hai chiều.")
    try:
        rows = list(X)
    except TypeError as error:
        raise ValueError("X phải là ma trận hai chiều.") from error
    if not rows:
        raise ValueError("X không được rỗng.")
    validated = []
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes, dict)):
            raise ValueError(f"X dòng {index} phải là dãy số.")
        try:
            values = list(row)
        except TypeError as error:
            raise ValueError(f"X dòng {index} phải là dãy số.") from error
        if n_features is None:
            n_features = len(values)
        if n_features == 0 or len(values) != n_features:
            raise ValueError(f"X dòng {index} sai số đặc trưng.")
        validated.append([
            _finite_number(value, f"X[{index}][{column}]")
            for column, value in enumerate(values)
        ])
    return validated


def _moments(rows):
    """Trả mean và phương sai MLE mỗi cột, chia n (không chia n - 1)."""
    means = []
    variances = []
    for column in range(len(rows[0])):
        if all(row[column] == rows[0][column] for row in rows):
            # Giữ cột hằng thập phân đúng bằng hằng, tránh phương sai giả do cộng.
            means.append(rows[0][column])
            variances.append(0.0)
            continue
        total = 0.0
        for row in rows:
            total = _finite_number(total + row[column], "Tổng khi tính trung bình")
        mean = total / len(rows)
        squared_total = 0.0
        for row in rows:
            difference = _finite_number(row[column] - mean, "Độ lệch")
            squared_total = _finite_number(
                squared_total + difference * difference, "Tổng bình phương độ lệch"
            )
        means.append(mean)
        variances.append(squared_total / len(rows))
    return means, variances


class GaussianNaiveBayes:
    """Mô hình hai lớp 0/1 với số đặc trưng tùy ý, không tự điền dữ liệu thiếu.

    var_smoothing và variance_floor phải dương, hữu hạn. Epsilon bằng
    max(var_smoothing * max(phương sai MLE toàn cục), variance_floor).
    Mọi API báo ValueError nếu dữ liệu sai hoặc phép tính không hữu hạn.
    """

    def __init__(self, var_smoothing=DEFAULT_VAR_SMOOTHING,
                 variance_floor=VARIANCE_FLOOR):
        """Nhận hệ số smoothing không thứ nguyên và sàn trong đơn vị bình phương."""
        self.var_smoothing = _finite_number(var_smoothing, "var_smoothing")
        self.variance_floor = _finite_number(variance_floor, "variance_floor")
        if self.var_smoothing <= 0 or self.variance_floor <= 0:
            raise ValueError("var_smoothing và variance_floor phải dương.")
        self._state = None

    def fit(self, X, y):
        """Học từ X (n, d), y (n,) đủ hai lớp; chỉ thay trạng thái sau khi thành công."""
        # Kiểm tra lại tham số công khai nếu bị thay đổi sau khởi tạo.
        parameters = type(self)(self.var_smoothing, self.variance_floor)
        rows = _matrix(X)
        if isinstance(y, (str, bytes, dict)):
            raise ValueError("y phải là dãy nhãn 0/1.")
        try:
            labels = list(y)
        except TypeError as error:
            raise ValueError("y phải là dãy nhãn 0/1.") from error
        if len(labels) != len(rows):
            raise ValueError("X và y phải có cùng số mẫu.")
        grouped_rows = [[], []]
        for row, label in zip(rows, labels):
            label = _finite_number(label, "Nhãn y")
            if label not in (0, 1):
                raise ValueError("Nhãn y chỉ nhận 0 hoặc 1.")
            grouped_rows[int(label)].append(row)
        counts = [len(group) for group in grouped_rows]
        if min(counts) == 0:
            raise ValueError("Dữ liệu fit phải có đủ hai lớp 0 và 1.")
        _, global_variances = _moments(rows)
        smoothing_product = _finite_number(
            parameters.var_smoothing * max(global_variances), "Tích smoothing"
        )
        epsilon = max(smoothing_product, parameters.variance_floor)
        means, variances_mle, variances_effective = [], [], []
        for group in grouped_rows:
            class_means, class_variances = _moments(group)
            effective = []
            for variance in class_variances:
                effective.append(_finite_number(variance + epsilon, "Phương sai hiệu dụng"))
            means.append(class_means)
            variances_mle.append(class_variances)
            variances_effective.append(effective)
        self._state = {
            "version": 1,
            "classes": [0, 1],
            "n_features": len(rows[0]),
            "class_counts": counts,
            "class_priors": [count / len(rows) for count in counts],
            "means": means,
            "variances_mle": variances_mle,
            "global_variances_mle": global_variances,
            "epsilon": epsilon,
            "variances_effective": variances_effective,
            "var_smoothing": parameters.var_smoothing,
            "variance_floor": parameters.variance_floor,
        }
        return self

    def _require_fitted(self):
        """Trả trạng thái đã học; từ chối thao tác trước fit hoặc from_state."""
        if self._state is None:
            raise ValueError("Mô hình chưa được fit.")
        return self._state

    def _joint_log_scores(self, X):
        """Tính log prior + tổng log Gaussian cho từng dòng, chưa chuẩn hóa."""
        state = self._require_fitted()
        rows = _matrix(X, state["n_features"])
        scores = []
        for row_index, row in enumerate(rows):
            class_scores = []
            for label in (0, 1):
                score = math.log(state["class_priors"][label])
                for column, value in enumerate(row):
                    variance = state["variances_effective"][label][column]
                    difference = _finite_number(
                        value - state["means"][label][column], f"Độ lệch dòng {row_index}"
                    )
                    squared_difference = _finite_number(
                        difference * difference, f"Bình phương dòng {row_index}"
                    )
                    # Tách log(2*pi*v) để không nhân v lớn gây tràn không cần thiết.
                    likelihood = -0.5 * (
                        math.log(2.0 * math.pi) + math.log(variance)
                        + _finite_number(squared_difference / variance,
                                         f"Tỷ số phương sai dòng {row_index}")
                    )
                    score = _finite_number(score + likelihood, f"Điểm log dòng {row_index}")
                class_scores.append(score)
            scores.append(class_scores)
        return scores

    def predict_log_proba(self, X):
        """Trả list (n, 2) log hậu nghiệm chuẩn hóa theo lớp [0, 1]."""
        result = []
        for scores in self._joint_log_scores(X):
            largest = max(scores)
            shifted = [_finite_number(score - largest, "Điểm log đã dịch") for score in scores]
            exponential_sum = 0.0
            for score in shifted:
                exponential_sum += math.exp(score)
            log_total = math.log(exponential_sum)
            result.append([_finite_number(score - log_total, "Log xác suất") for score in shifted])
        return result

    def predict_proba(self, X):
        """Trả list (n, 2) xác suất; exp có thể về 0 khi hậu nghiệm rất nhỏ."""
        return [[math.exp(value) for value in row] for row in self.predict_log_proba(X)]

    def predict(self, X):
        """Trả list nhãn; so sánh điểm log gốc và chọn lớp 0 khi hòa."""
        return [1 if scores[1] > scores[0] else 0 for scores in self._joint_log_scores(X)]

    def to_state(self):
        """Trả bản sao dict có thể ghi JSON, không chia sẻ list với mô hình."""
        return deepcopy(self._require_fitted())

    @classmethod
    def from_state(cls, state):
        """Nạp dict phiên bản 1, kiểm tra shape và quan hệ thống kê trước khi dùng.

        Các hàng means/variances theo lớp [0, 1]. Counts là số nguyên dương;
        prior phải khớp counts. MLE chưa cộng epsilon; effective phải đúng
        MLE + epsilon. Phương sai toàn cục được đối chiếu từ moments hai lớp.
        Không cộng lại epsilon khi nạp; từ chối khóa lạ hoặc thiếu.
        """
        keys = {"version", "classes", "n_features", "class_counts", "class_priors",
                "means", "variances_mle", "global_variances_mle", "epsilon",
                "variances_effective", "var_smoothing", "variance_floor"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("State sai tập khóa.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("State sai phiên bản.")
        if (type(state["classes"]) is not list or len(state["classes"]) != 2
                or any(type(label) is not int for label in state["classes"])
                or state["classes"] != [0, 1]):
            raise ValueError("State phải có thứ tự lớp [0, 1].")
        n_features = state["n_features"]
        if type(n_features) is not int or n_features <= 0:
            raise ValueError("State n_features phải là số nguyên dương.")
        counts = state["class_counts"]
        if (type(counts) is not list or len(counts) != 2
                or any(type(count) is not int or count <= 0 for count in counts)):
            raise ValueError("State class_counts phải có hai số nguyên dương.")
        model = cls(state["var_smoothing"], state["variance_floor"])
        validated = deepcopy(state)
        for key in ("means", "variances_mle", "variances_effective"):
            values = _matrix(state[key], n_features)
            if len(values) != 2:
                raise ValueError(f"State {key} phải có hai hàng.")
            validated[key] = values
        for key, length in (("class_priors", 2), ("global_variances_mle", n_features)):
            if type(state[key]) is not list or len(state[key]) != length:
                raise ValueError(f"State {key} sai kích thước.")
            validated[key] = [_finite_number(value, key) for value in state[key]]
        epsilon = _finite_number(state["epsilon"], "epsilon")
        if epsilon <= 0:
            raise ValueError("State epsilon phải dương.")
        for variance in validated["global_variances_mle"]:
            if variance < 0:
                raise ValueError("State phương sai toàn cục không được âm.")
        expected_epsilon = max(_finite_number(
            model.var_smoothing * max(validated["global_variances_mle"]), "Tích smoothing"
        ), model.variance_floor)
        if epsilon != expected_epsilon:
            raise ValueError("State epsilon không khớp cấu hình và phương sai toàn cục.")
        total_count = sum(counts)
        for label in (0, 1):
            prior = validated["class_priors"][label]
            if not 0 < prior < 1 or prior != counts[label] / total_count:
                raise ValueError("State prior không khớp số mẫu lớp.")
            for column in range(n_features):
                mle = validated["variances_mle"][label][column]
                if counts[label] == 1 and mle != 0:
                    raise ValueError("Lớp một mẫu phải có phương sai MLE bằng 0.")
                effective = validated["variances_effective"][label][column]
                expected_effective = _finite_number(mle + epsilon, "Phương sai hiệu dụng")
                if mle < 0 or effective <= 0 or effective != expected_effective:
                    raise ValueError("State phương sai MLE/hiệu dụng không nhất quán.")
        for column in range(n_features):
            # Đối chiếu định luật phương sai toàn phần. Sai số mean phụ thuộc
            # độ lớn dữ liệu gốc; cần dung sai đó cho dữ liệu có offset rất lớn.
            difference = _finite_number(validated["means"][0][column]
                                        - validated["means"][1][column], "Độ lệch mean state")
            priors = validated["class_priors"]
            reconstructed = _finite_number(
                priors[0] * validated["variances_mle"][0][column]
                + priors[1] * validated["variances_mle"][1][column]
                + (priors[0] * difference) * (priors[1] * difference),
                "Phương sai state"
            )
            actual = validated["global_variances_mle"][column]
            if actual == reconstructed:
                continue
            mean_error = 1e-14 * max(abs(validated["means"][0][column]),
                                    abs(validated["means"][1][column]))
            tolerance = _finite_number(
                1e-12 * max(actual, reconstructed)
                + mean_error * (mean_error + 2 * math.sqrt(max(actual, reconstructed))),
                "Dung sai moments state"
            )
            if abs(actual - reconstructed) > tolerance:
                raise ValueError("State phương sai toàn cục không khớp moments lớp.")
        validated["epsilon"] = epsilon
        validated["var_smoothing"] = model.var_smoothing
        validated["variance_floor"] = model.variance_floor
        model._state = validated
        return model
