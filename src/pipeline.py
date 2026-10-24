"""Ghép tiền xử lý và Gaussian NB tự cài; lưu JSON kèm SHA-256."""

import hashlib
import json
import math
from pathlib import Path

from .config import CLASS_LABELS, DEFAULT_VAR_SMOOTHING, FEATURE_COLUMNS, VARIANCE_FLOOR
from .gaussian_nb_numpy import GaussianNaiveBayes
from .preprocessing import Preprocessor


def _json_copy(value):
    """Sao chép cây JSON thuần; từ chối NaN/inf, khóa không chuỗi, kiểu lạ."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("Trạng thái/metadata phải là JSON thuần với số hữu hạn.")


def _artifact_path(path):
    """Chuẩn hóa đường dẫn file; từ chối chuỗi rỗng, sai kiểu hoặc thư mục."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Đường dẫn phải là tên file không rỗng.")
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError("Đường dẫn phải trỏ tới file, không phải thư mục.")
    return file_path


def _checksum_text(value):
    """Kiểm tra SHA-256 dạng 64 ký tự hex, trả chuỗi chữ thường."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SHA-256 phải gồm đúng 64 ký tự hex.")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("SHA-256 chứa ký tự không hợp lệ.")
    return value.lower()


def sha256_file(path):
    """Trả SHA-256 hex của byte file, đọc từng khối; lỗi file/path gây ValueError."""
    file_path = _artifact_path(path)
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Không đọc được file: {file_path}") from error
    return digest.hexdigest()


class WaterPipeline:
    """Pipeline nghiệp vụ đúng 9 cột FEATURE_COLUMNS, xác suất theo [0, 1].

    preprocessor/model công khai để truy xuất trạng thái qua to_state; không
    cần truy cập thuộc tính nội bộ của mô hình. run_id/metadata theo artifact.
    """

    def __init__(self, strategy="mean", standardize=False,
                 var_smoothing=DEFAULT_VAR_SMOOTHING, variance_floor=VARIANCE_FLOOR):
        """Cấu hình như hai bộ tự cài; giá trị không hợp lệ gây ValueError."""
        self.preprocessor = Preprocessor(strategy, standardize)
        self.model = GaussianNaiveBayes(var_smoothing=var_smoothing,
                                       variance_floor=variance_floor)
        self.var_smoothing = var_smoothing
        self.variance_floor = variance_floor
        self.run_id = ""
        self.metadata = {}

    def fit(self, X, y):
        """Fit X (n, 9), y nhị phân; trả self, không đổi state cũ nếu fit lỗi.

        Thống kê tiền xử lý chỉ học trên X của lần gọi này. Sai shape/nhãn,
        cột toàn thiếu hay lỗi số học gây ValueError từ các bộ tự cài.
        """
        preprocessor = Preprocessor(self.preprocessor.strategy, self.preprocessor.standardize)
        transformed = preprocessor.fit_transform(X)
        if preprocessor.to_state()["n_features"] != len(FEATURE_COLUMNS):
            raise ValueError("Pipeline yêu cầu đúng 9 đặc trưng theo FEATURE_COLUMNS.")
        model = GaussianNaiveBayes(var_smoothing=self.var_smoothing,
                                  variance_floor=self.variance_floor)
        model.fit(transformed, y)
        self.preprocessor = preprocessor
        self.model = model
        self.run_id = ""
        self.metadata = {}
        return self

    def predict(self, X):
        """Trả nhãn n mẫu sau transform chung; batch có lỗi bị từ chối toàn bộ."""
        return self.model.predict(self.preprocessor.transform(X))

    def predict_proba(self, X):
        """Trả xác suất shape (n, 2), thứ tự lớp [0, 1], không fit lại."""
        return self.model.predict_proba(self.preprocessor.transform(X))

    def to_state(self, run_id="", metadata=None):
        """Trả state schema_version=1, JSON thuần, bản sao không chia sẻ dữ liệu.

        run_id rỗng và metadata=None giữ thông tin hiện có; metadata phải là
        dict JSON hữu hạn. Chưa fit hoặc cấu trúc liên kết sai gây ValueError.
        """
        if not isinstance(run_id, str):
            raise ValueError("run_id phải là chuỗi.")
        metadata = self.metadata if metadata is None else metadata
        if not isinstance(metadata, dict):
            raise ValueError("metadata phải là dict.")
        state = _json_copy({
            "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS),
            "class_labels": list(CLASS_LABELS), "run_id": run_id or self.run_id,
            "metadata": metadata, "preprocessor": self.preprocessor.to_state(),
            "model": self.model.to_state(),
        })
        self._validate_components(state)
        return state

    @staticmethod
    def _validate_components(state):
        """Kiểm tra liên kết số cột/lớp, còn schema con do từng bộ tự cài kiểm tra."""
        preprocessor = Preprocessor.from_state(state["preprocessor"])
        model = GaussianNaiveBayes.from_state(state["model"])
        if preprocessor.to_state()["n_features"] != len(FEATURE_COLUMNS):
            raise ValueError("Số cột tiền xử lý không khớp FEATURE_COLUMNS.")
        model_state = model.to_state()
        if model_state["n_features"] != len(FEATURE_COLUMNS):
            raise ValueError("Số cột mô hình không khớp tiền xử lý.")
        if model_state["classes"] != CLASS_LABELS:
            raise ValueError("Thứ tự lớp mô hình không khớp [0, 1].")
        preprocessing_state = preprocessor.to_state()
        counts = model_state["class_counts"]
        for column in range(len(FEATURE_COLUMNS)):
            expected_mean = preprocessing_state["means"][column]
            scale = preprocessing_state["scales"][column]
            constant = preprocessing_state["raw_min"][column] == preprocessing_state["raw_max"][column]
            expected_variance = 0.0 if constant else scale * scale
            if preprocessing_state["standardize"]:
                expected_mean = 0.0
                expected_variance = 0.0 if constant else 1.0
            model_mean = sum(model_state["means"][label][column] * counts[label]
                             for label in (0, 1)) / sum(counts)
            mean_tolerance = 1e-9
            if preprocessing_state["standardize"]:
                # Tâm lưu bằng float có thể lệch nửa ULP ở dữ liệu offset lớn.
                mean_tolerance += 2 * math.ulp(preprocessing_state["means"][column]) / scale
                # Scale dựa trên độ lệch quanh tâm float, không phải mean chính xác.
                expected_variance = 0.0 if constant else max(0.0, 1.0 - model_mean * model_mean)
            if not math.isclose(model_mean, expected_mean, rel_tol=1e-8, abs_tol=mean_tolerance):
                raise ValueError("Trung bình mô hình không khớp trạng thái tiền xử lý.")
            if not math.isclose(model_state["global_variances_mle"][column], expected_variance,
                                rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError("Phương sai mô hình không khớp trạng thái tiền xử lý.")
        return preprocessor, model

    @classmethod
    def from_state(cls, state):
        """Nạp state v1 đã kiểm tra; không fit, không cộng lại smoothing.

        Bắt buộc đúng khóa, đúng thứ tự 9 đặc trưng, lớp nguyên [0, 1],
        schema con hợp lệ và số chiều khớp. Mọi lỗi state gây ValueError.
        """
        state = _json_copy(state)
        keys = {"schema_version", "feature_columns", "class_labels", "run_id",
                "metadata", "preprocessor", "model"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("Bộ khóa pipeline không đúng schema v1.")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("schema_version phải là số nguyên 1.")
        if state["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Tên/thứ tự feature_columns không đúng cấu hình.")
        labels = state["class_labels"]
        if not isinstance(labels, list) or labels != CLASS_LABELS or any(type(x) is not int for x in labels):
            raise ValueError("class_labels phải là [0, 1] nguyên theo đúng thứ tự.")
        if not isinstance(state["run_id"], str) or not isinstance(state["metadata"], dict):
            raise ValueError("run_id phải là chuỗi và metadata phải là dict.")
        preprocessor, model = cls._validate_components(state)
        pipeline = cls.__new__(cls)
        pipeline.preprocessor = preprocessor
        pipeline.model = model
        pipeline.var_smoothing = state["model"]["var_smoothing"]
        pipeline.variance_floor = state["model"]["variance_floor"]
        pipeline.run_id = state["run_id"]
        pipeline.metadata = state["metadata"]
        return pipeline


def save_pipeline(pipeline, path, run_id="", metadata=None):
    """Ghi JSON UTF-8 và <path>.sha256; trả SHA-256 của byte JSON đã ghi.

    Không tự tạo thư mục cha. Serialize/validate trước ghi; NaN/inf và state
    sai bị chặn. Lỗi đường dẫn/quyền ghi gây ValueError. Sidecar chứa hex + LF.
    """
    if not isinstance(pipeline, WaterPipeline):
        raise ValueError("pipeline phải là WaterPipeline.")
    file_path = _artifact_path(path)
    state = pipeline.to_state(run_id=run_id, metadata=metadata)
    payload = (json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    checksum = hashlib.sha256(payload).hexdigest()
    try:
        file_path.write_bytes(payload)
        Path(str(file_path) + ".sha256").write_text(checksum + "\n", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Không ghi được pipeline/checksum: {file_path}") from error
    return checksum


def _unique_json_object(pairs):
    """Tạo dict khi đọc JSON, từ chối khóa lặp để không che dữ liệu hỏng."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON có khóa trùng: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    """Chặn NaN/Infinity của parser JSON Python bằng ValueError."""
    raise ValueError(f"Hằng JSON không hợp lệ: {value}")


def load_pipeline(path, expected_sha256=None):
    """Đọc pipeline sau kiểm tra sidecar bắt buộc và checksum kỳ vọng tùy chọn.

    Checksum so trên đúng byte được parse, không đọc model lần hai. JSON lỗi,
    UTF-8 lỗi, thiếu sidecar/file, checksum sai hoặc state sai gây ValueError.
    Checksum bảo vệ tính nhất quán, không phải chữ ký xác thực nguồn model.
    """
    file_path = _artifact_path(path)
    if expected_sha256 is not None:
        expected_sha256 = _checksum_text(expected_sha256)
    try:
        sidecar = Path(str(file_path) + ".sha256").read_text(encoding="utf-8").strip()
        checksum = _checksum_text(sidecar)
        payload = file_path.read_bytes()
    except (OSError, UnicodeError) as error:
        raise ValueError("Không đọc được pipeline hoặc sidecar SHA-256.") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != checksum or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError("Checksum pipeline không khớp.")
    try:
        state = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object,
                           parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Pipeline không phải JSON UTF-8 hợp lệ.") from error
    return WaterPipeline.from_state(state)
