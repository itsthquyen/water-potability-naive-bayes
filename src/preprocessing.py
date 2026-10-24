"""Tự điền thiếu, chuẩn hóa và chia phân tầng; không học từ tập đánh giá."""

import math
from numbers import Integral, Real
import random

from .config import N_FOLDS, SEED, TEST_FRACTION


def _finite_number(value, name):
    """Đổi số thực hữu hạn thành float; từ chối bool, chuỗi và số tràn."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} phải là số thực, không phải bool.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} nằm ngoài phạm vi số thực.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} phải hữu hạn.")
    return number


def _validate_matrix(X, n_features=None):
    """Sao chép X (n mẫu, p cột) thành list; None/NaN là thiếu, còn lại hữu hạn.

    Ma trận rỗng, hàng rỗng/không đều, chuỗi, bool và vô hạn gây ValueError.
    Thứ tự cột và thứ tự hàng được giữ nguyên; không sửa đầu vào.
    """
    if isinstance(X, (str, bytes, dict)):
        raise ValueError("X phải là ma trận hai chiều.")
    try:
        rows = list(X)
    except TypeError as error:
        raise ValueError("X phải là ma trận hai chiều.") from error
    if not rows:
        raise ValueError("X không được rỗng.")
    matrix = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, dict)):
            raise ValueError(f"Hàng {row_index} phải là dãy số.")
        try:
            values = list(row)
        except TypeError as error:
            raise ValueError(f"Hàng {row_index} phải là dãy số.") from error
        if n_features is None:
            n_features = len(values)
        if not values or len(values) != n_features:
            raise ValueError(f"Hàng {row_index} có số cột không hợp lệ.")
        clean_row = []
        for column_index, value in enumerate(values):
            if value is None:
                clean_row.append(None)
            elif isinstance(value, Real) and not isinstance(value, bool) and value != value:
                clean_row.append(None)
            else:
                clean_row.append(_finite_number(value, f"Ô [{row_index}, {column_index}]"))
        matrix.append(clean_row)
    return matrix


class Preprocessor:
    """Học thống kê từ X train, áp dụng nguyên trạng cho các tập khác.

    strategy là 'mean' hoặc 'median'; standardize phải là bool. Các vector
    trạng thái theo đúng thứ tự p cột đầu vào, không chứa nhãn hoặc ID.
    """

    def __init__(self, strategy="mean", standardize=False):
        """Khởi tạo chưa fit; cấu hình không hợp lệ gây ValueError."""
        if strategy not in ("mean", "median"):
            raise ValueError("strategy phải là 'mean' hoặc 'median'.")
        if not isinstance(standardize, bool):
            raise ValueError("standardize phải là bool.")
        self.strategy = strategy
        self.standardize = standardize
        self._state = None

    def fit(self, X):
        """Học từ X (n, p), trả self; lỗi cột toàn thiếu hoặc thống kê tràn số.

        Điền thiếu bằng thống kê trên ô quan sát, rồi tính mean và phương sai
        tổng bình phương độ lệch / n (ddof=0). Scale bằng sqrt(variance),
        riêng cột hằng dùng 1. Min/max chỉ lấy ô gốc không thiếu.
        Lần fit lỗi không thay trạng thái đã học trước đó.
        """
        matrix = _validate_matrix(X)
        state = {
            "version": 1, "strategy": self.strategy, "standardize": self.standardize,
            "n_features": len(matrix[0]), "fill_values": [], "means": [],
            "scales": [], "raw_min": [], "raw_max": [],
        }
        for column_index in range(state["n_features"]):
            observed = []
            for row in matrix:
                if row[column_index] is not None:
                    observed.append(row[column_index])
            if not observed:
                raise ValueError(f"Cột {column_index} thiếu hoàn toàn trong phần fit.")
            lower, upper = min(observed), max(observed)
            if lower == upper:
                fill_value = lower
            elif self.strategy == "mean":
                fill_value = sum(observed) / len(observed)
            else:
                ordered = sorted(observed)
                middle = len(ordered) // 2
                if len(ordered) % 2:
                    fill_value = ordered[middle]
                else:
                    fill_value = ordered[middle - 1] / 2 + ordered[middle] / 2
            fill_value = _finite_number(fill_value, "Giá trị điền thiếu")
            filled = []
            for row in matrix:
                value = row[column_index]
                filled.append(fill_value if value is None else value)
            mean = lower if lower == upper else sum(filled) / len(filled)
            mean = _finite_number(mean, "Trung bình")
            squared_deviations = 0.0
            for value in filled:
                difference = value - mean
                squared_deviations += difference * difference
            variance = _finite_number(squared_deviations / len(filled), "Phương sai")
            scale = math.sqrt(variance) if variance > 0 else 1.0
            state["fill_values"].append(fill_value)
            state["means"].append(mean)
            state["scales"].append(scale)
            state["raw_min"].append(lower)
            state["raw_max"].append(upper)
        self._state = state
        return self

    def transform(self, X):
        """Trả list[list[float]] cùng shape X; chỉ dùng thống kê đã fit.

        Nếu bật chuẩn hóa, tính (x đã điền - mean) / scale. Không clip giá trị
        ngoài min/max. Chưa fit, sai số cột, dữ liệu lỗi hoặc tràn gây ValueError.
        """
        if self._state is None:
            raise ValueError("Phải fit trước khi transform.")
        state = self._state
        matrix = _validate_matrix(X, state["n_features"])
        transformed = []
        for row in matrix:
            output_row = []
            for column_index, value in enumerate(row):
                if value is None:
                    value = state["fill_values"][column_index]
                if state["standardize"]:
                    value = (value - state["means"][column_index]) / state["scales"][column_index]
                output_row.append(_finite_number(value, "Kết quả transform"))
            transformed.append(output_row)
        return transformed

    def fit_transform(self, X):
        """Fit đúng một lần rồi trả ma trận train đã biến đổi; lỗi như fit/transform."""
        # Giữ bản sao để X dạng iterator không bị tiêu thụ hai lần.
        matrix = _validate_matrix(X)
        return self.fit(matrix).transform(matrix)

    def to_state(self):
        """Trả dict JSON-compatible độc lập; chưa fit gây ValueError.

        version=1 xác định schema; n_features là p > 0. fill_values là giá trị
        điền, means/scales học sau điền (kể cả khi không chuẩn hóa), raw_min/
        raw_max học trước điền. Mọi vector có p số hữu hạn, scales > 0.
        """
        if self._state is None:
            raise ValueError("Phải fit trước khi lưu trạng thái.")
        return {key: value[:] if isinstance(value, list) else value
                for key, value in self._state.items()}

    @classmethod
    def from_state(cls, state):
        """Nạp bản sao schema v1; thiếu/thừa khóa, kiểu/range sai gây ValueError.

        Không fit lại và không tính lại thống kê từ dữ liệu đánh giá. Kiểm tra
        chiều, hữu hạn, scale dương, min <= fill/mean <= max và cột hằng scale=1.
        """
        keys = {"version", "strategy", "standardize", "n_features", "fill_values",
                "means", "scales", "raw_min", "raw_max"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("Trạng thái không đúng bộ khóa schema v1.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("version phải là số nguyên 1.")
        n_features = state["n_features"]
        if type(n_features) is not int or n_features < 1:
            raise ValueError("n_features phải là số nguyên dương.")
        preprocessor = cls(state["strategy"], state["standardize"])
        clean_state = dict(state)
        for key in ("fill_values", "means", "scales", "raw_min", "raw_max"):
            values = state[key]
            if not isinstance(values, list) or len(values) != n_features:
                raise ValueError(f"{key} phải là list dài n_features.")
            clean_state[key] = [_finite_number(value, key) for value in values]
        for column_index in range(n_features):
            lower = clean_state["raw_min"][column_index]
            upper = clean_state["raw_max"][column_index]
            scale = clean_state["scales"][column_index]
            if lower > upper or scale <= 0:
                raise ValueError("Min/max hoặc scale không hợp lệ.")
            # Cho phép sai số làm tròn khi cộng/chia các số thực hữu hạn.
            tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
            for key in ("fill_values", "means"):
                value = clean_state[key][column_index]
                if value < lower - tolerance or value > upper + tolerance:
                    raise ValueError(f"{key} nằm ngoài min/max train.")
            if lower == upper and scale != 1.0:
                raise ValueError("Cột hằng phải lưu scale=1.")
            if (clean_state["strategy"] == "mean"
                    and abs(clean_state["fill_values"][column_index]
                            - clean_state["means"][column_index]) > tolerance):
                raise ValueError("Giá trị điền mean không khớp trung bình sau điền.")
        preprocessor._state = clean_state
        return preprocessor


def _group_labels(y, seed):
    """Kiểm tra y một chiều chỉ số nguyên 0/1, đủ hai lớp; shuffle cục bộ.

    Seed phải là số nguyên không âm, không phải bool. Trả hai list chỉ số;
    không sửa y hay trạng thái random toàn cục. Đầu vào sai gây ValueError.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed phải là số nguyên không âm.")
    if isinstance(y, (str, bytes, dict)):
        raise ValueError("y phải là dãy nhãn 0/1.")
    try:
        labels = list(y)
    except TypeError as error:
        raise ValueError("y phải là dãy nhãn 0/1.") from error
    groups = [[], []]
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
            raise ValueError("Nhãn phải là số nguyên 0 hoặc 1, không phải bool.")
        groups[int(label)].append(index)
    if not groups[0] or not groups[1]:
        raise ValueError("y phải có đủ hai lớp 0 và 1.")
    generator = random.Random(int(seed))
    for group in groups:
        generator.shuffle(group)
    return groups


def stratified_holdout(y, test_fraction=TEST_FRACTION, seed=SEED):
    """Trả (dev_indices, test_indices) phân tầng theo y, chỉ số bắt đầu từ 0.

    Số test mỗi lớp = floor(n_lớp * test_fraction + 0.5), không tự ép/clamp.
    Tỷ lệ phải hữu hạn trong (0, 1); mỗi lớp phải có ở cả dev và test, nếu
    không thì ValueError. Không bảo đảm dev đủ 5 fold: bên huấn luyện kiểm tra.
    Caller phải xử lý trùng/nhóm trước khi gọi. Kết quả sắp theo chỉ số gốc.
    """
    fraction = _finite_number(test_fraction, "test_fraction")
    if not 0 < fraction < 1:
        raise ValueError("test_fraction phải nằm trong (0, 1).")
    groups = _group_labels(y, seed)
    dev_indices, test_indices = [], []
    for group in groups:
        test_count = math.floor(len(group) * fraction + 0.5)
        if test_count < 1 or test_count >= len(group):
            raise ValueError("Mỗi lớp phải có ít nhất một mẫu ở dev và test.")
        test_indices.extend(group[:test_count])
        dev_indices.extend(group[test_count:])
    return sorted(dev_indices), sorted(test_indices)


def stratified_folds(y, n_folds=N_FOLDS, seed=SEED):
    """Trả list n_folds cặp (train_indices, val_indices), chỉ số gốc tăng dần.

    Nhóm và shuffle từng lớp rồi phân phối vòng tròn vào các fold. Mỗi mẫu
    validation đúng một lần; chênh lệch số mẫu cùng lớp giữa fold tối đa 1.
    n_folds là nguyên >=2 (không bool), mỗi lớp ít nhất n_folds mẫu để mọi
    train/validation đủ hai lớp; vi phạm gây ValueError. Xử lý trùng trước gọi.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds phải là số nguyên từ 2 trở lên.")
    groups = _group_labels(y, seed)
    for group in groups:
        if len(group) < n_folds:
            raise ValueError("Mỗi lớp phải có ít nhất n_folds mẫu.")
    validation_groups = [[] for _ in range(n_folds)]
    for group in groups:
        for position, index in enumerate(group):
            validation_groups[position % n_folds].append(index)
    all_indices = sorted(groups[0] + groups[1])
    folds = []
    for validation in validation_groups:
        validation_set = set(validation)
        train = [index for index in all_indices if index not in validation_set]
        folds.append((train, sorted(validation)))
    return folds
