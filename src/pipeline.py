"""Ghép rời rạc hóa và Naive Bayes đếm tần suất; lưu JSON kèm SHA-256."""

import hashlib
import json
import math
from pathlib import Path

from .config import CLASS_LABELS, FEATURE_COLUMNS, MODEL_ALGORITHM
from .data_processing.discretization import Discretizer
from .algorithms.naive_bayes import NaiveBayes
from .data_processing.preprocessing import _validate_matrix


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
    """Chuỗi 9 số đo → điền thiếu/rời rạc hóa → Naive Bayes đếm tần suất."""

    def __init__(self):
        self.preprocessor = Discretizer()
        self.model = NaiveBayes()
        self.run_id = ""
        self.metadata = {}

    def fit(self, X, y):
        """Fit trên X (n, 9), y (n,); lỗi không làm mất trạng thái trước đó."""
        preprocessor = Discretizer()
        # Giữ bản sao hàng và cột nếu đầu vào là iterator.
        rows = _validate_matrix(X, len(FEATURE_COLUMNS))
        preprocessor.fit(rows)
        sizes = preprocessor.category_sizes
        model = NaiveBayes().fit(preprocessor.transform(rows), y, sizes)
        self.preprocessor, self.model = preprocessor, model
        self.run_id, self.metadata = "", {}
        return self

    def predict(self, X):
        """Trả nhãn n mẫu; lỗi dữ liệu hoặc hai điểm bằng 0 được báo rõ."""
        return self.predict_with_proba(X)[0]

    def predict_proba(self, X):
        """Trả hậu nghiệm (n, 2) theo [0, 1]."""
        return self.predict_with_proba(X)[1]

    def predict_with_proba(self, X):
        """Dùng tham số train đã lưu; trả nhãn và xác suất từ cùng lượt suy luận."""
        return self.model.predict_with_proba(self.preprocessor.transform(X))

    def to_state(self, run_id="", metadata=None):
        """Trả JSON schema v2, có định danh thuật toán và thứ tự cột/nhãn."""
        if not isinstance(run_id, str):
            raise ValueError("run_id phải là chuỗi.")
        metadata = self.metadata if metadata is None else metadata
        state = _json_copy({
            "schema_version": 2, "algorithm": MODEL_ALGORITHM,
            "feature_columns": FEATURE_COLUMNS, "class_labels": CLASS_LABELS,
            "run_id": run_id or self.run_id, "metadata": metadata,
            "preprocessor": self.preprocessor.to_state(), "model": self.model.to_state(),
        })
        self.from_state(state)
        return state

    @classmethod
    def from_state(cls, state):
        """Nạp model đếm v2; không chuyển mô hình Gaussian v1 thành bản đếm."""
        state = _json_copy(state)
        if isinstance(state, dict) and type(state.get("schema_version")) is int and state["schema_version"] == 1:
            raise ValueError("Model Gaussian cũ không dùng được; hãy chọn Naive Bayes đếm tần suất.")
        keys = {"schema_version", "algorithm", "feature_columns", "class_labels",
                "run_id", "metadata", "preprocessor", "model"}
        if (not isinstance(state, dict) or set(state) != keys
                or type(state["schema_version"]) is not int or state["schema_version"] != 2
                or state["algorithm"] != MODEL_ALGORITHM):
            raise ValueError("Pipeline phải đúng schema v2 của Naive Bayes đếm tần suất.")
        if state["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Tên/thứ tự cột không khớp FEATURE_COLUMNS.")
        labels = state["class_labels"]
        if labels != CLASS_LABELS or any(type(label) is not int for label in labels):
            raise ValueError("Thứ tự lớp phải là [0, 1].")
        if not isinstance(state["run_id"], str) or not isinstance(state["metadata"], dict):
            raise ValueError("run_id phải là chuỗi và metadata phải là dict.")
        preprocessor = Discretizer.from_state(state["preprocessor"])
        model = NaiveBayes.from_state(state["model"])
        sizes = preprocessor.category_sizes
        if len(sizes) != len(FEATURE_COLUMNS) or state["model"]["category_sizes"] != sizes:
            raise ValueError("Số khoảng/số cột không khớp giữa tiền xử lý và Naive Bayes.")
        pipeline = cls()
        pipeline.preprocessor, pipeline.model = preprocessor, model
        pipeline.run_id, pipeline.metadata = state["run_id"], state["metadata"]
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
