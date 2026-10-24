"""Kiểm thử pipeline trên dữ liệu tổng hợp, không dùng dataset thực."""

import copy
import hashlib
import json

import pytest

from src.config import FEATURE_COLUMNS
from src.gaussian_nb_numpy import GaussianNaiveBayes
from src.pipeline import WaterPipeline, load_pipeline, save_pipeline, sha256_file
from src.preprocessing import Preprocessor


def training_example():
    """Trả 6 mẫu nhân tạo, mỗi lớp 3 mẫu, đúng 9 cột và có ô thiếu."""
    matrix = [[float(value + column) for column in range(9)]
              for value in (1, 2, 3, 8, 9, 10)]
    matrix[1][2] = None
    return matrix, [0, 0, 0, 1, 1, 1]


def test_large_offset_standardized_roundtrip():
    """Offset 1e16 có ULP=2; tâm float lệch vẫn phải lưu/nạp được."""
    rows = [[value] * 9 for value in (1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6)]
    pipeline = WaterPipeline(standardize=True).fit(rows, [0, 0, 1, 1])
    restored = WaterPipeline.from_state(pipeline.to_state())
    assert restored.predict(rows) == [0, 0, 1, 1]
    assert restored.predict_proba(rows) == pipeline.predict_proba(rows)


@pytest.mark.parametrize("standardize", [False, True])
def test_roundtrip_predictions_probabilities_metadata(tmp_path, standardize):
    matrix, labels = training_example()
    pipeline = WaterPipeline("median", standardize)
    assert pipeline.fit(matrix, labels) is pipeline
    samples = [matrix[0], matrix[-1], [None] + matrix[2][1:]]
    expected_labels = pipeline.predict(samples)
    expected_probabilities = pipeline.predict_proba(samples)
    assert expected_labels[:2] == [0, 1]
    file_path = tmp_path / "pipeline.json"
    metadata = {"nguồn": "dữ liệu tổng hợp", "fold": [1, 2], "score": 0.5}
    checksum = save_pipeline(pipeline, file_path, "synthetic-run", metadata)
    assert checksum == sha256_file(file_path)
    assert file_path.with_name("pipeline.json.sha256").read_text().strip() == checksum
    assert "dữ liệu tổng hợp" in file_path.read_text(encoding="utf-8")
    restored = load_pipeline(file_path, expected_sha256=checksum)
    assert restored.run_id == "synthetic-run"
    assert restored.metadata == metadata
    assert restored.predict(samples) == expected_labels
    assert restored.predict_proba(samples) == expected_probabilities
    assert restored.model.to_state() == pipeline.model.to_state()
    assert restored.preprocessor.to_state() == pipeline.preprocessor.to_state()
    metadata["fold"].append(999)
    state = restored.to_state()
    assert state["metadata"]["fold"] == [1, 2]
    state["preprocessor"]["means"][0] = 999
    assert restored.predict_proba(samples) == expected_probabilities
    assert state["feature_columns"] == FEATURE_COLUMNS
    assert state["class_labels"] == [0, 1]


def test_load_does_not_fit_or_add_smoothing(tmp_path, monkeypatch):
    matrix, labels = training_example()
    pipeline = WaterPipeline().fit(matrix, labels)
    checksum = save_pipeline(pipeline, tmp_path / "model.json")
    before = pipeline.model.to_state()

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Không được fit khi load hoặc predict.")

    monkeypatch.setattr(Preprocessor, "fit", forbid_fit)
    monkeypatch.setattr(GaussianNaiveBayes, "fit", forbid_fit)
    restored = load_pipeline(tmp_path / "model.json", checksum)
    assert restored.model.to_state() == before
    assert restored.predict([matrix[0]]) == [0]


def test_invalid_width_unfitted_and_failed_refit():
    with pytest.raises(ValueError):
        WaterPipeline().fit([[1, 2], [3, 4]], [0, 1])
    with pytest.raises(ValueError):
        WaterPipeline().to_state()
    with pytest.raises(ValueError):
        WaterPipeline().predict([[1] * 9])
    matrix, labels = training_example()
    pipeline = WaterPipeline().fit(matrix, labels)
    before = pipeline.to_state()
    with pytest.raises(ValueError):
        pipeline.fit(matrix, [0] * len(labels))
    assert pipeline.to_state() == before
    with pytest.raises(ValueError):
        pipeline.predict([[1] * 8])


@pytest.mark.parametrize("key, value", [
    ("schema_version", True), ("schema_version", 2),
    ("feature_columns", list(reversed(FEATURE_COLUMNS))),
    ("feature_columns", FEATURE_COLUMNS[:-1]),
    ("class_labels", [1, 0]), ("class_labels", [False, True]),
    ("run_id", 123), ("metadata", []),
])
def test_invalid_top_level_schema(key, value):
    matrix, labels = training_example()
    state = WaterPipeline().fit(matrix, labels).to_state()
    state[key] = value
    with pytest.raises(ValueError):
        WaterPipeline.from_state(state)


def test_missing_extra_keys_and_dimensions():
    matrix, labels = training_example()
    state = WaterPipeline().fit(matrix, labels).to_state()
    for invalid in (None, [], {}, {**state, "extra": 1}):
        with pytest.raises(ValueError):
            WaterPipeline.from_state(invalid)
    missing = copy.deepcopy(state)
    del missing["model"]
    with pytest.raises(ValueError):
        WaterPipeline.from_state(missing)
    # State con riêng lẻ hợp lệ nhưng không khớp nghiệp vụ 9 cột.
    bad_preprocessor = copy.deepcopy(state)
    bad_preprocessor["preprocessor"] = Preprocessor().fit([[1], [2]]).to_state()
    with pytest.raises(ValueError):
        WaterPipeline.from_state(bad_preprocessor)
    bad_model = copy.deepcopy(state)
    bad_model["model"] = GaussianNaiveBayes().fit([[1], [2], [8], [9]], [0, 0, 1, 1]).to_state()
    with pytest.raises(ValueError):
        WaterPipeline.from_state(bad_model)
    mismatched = copy.deepcopy(state)
    shifted = [[value + 100 if value is not None else None for value in row] for row in matrix]
    mismatched["preprocessor"] = Preprocessor().fit(shifted).to_state()
    with pytest.raises(ValueError):
        WaterPipeline.from_state(mismatched)


@pytest.mark.parametrize("field, value", [
    ("scales", [0] * 9), ("raw_min", [1000] * 9),
    ("raw_max", [-1000] * 9), ("means", [float("nan")] * 9),
])
def test_reject_invalid_preprocessor_state(field, value):
    matrix, labels = training_example()
    state = WaterPipeline().fit(matrix, labels).to_state()
    state["preprocessor"][field] = value
    with pytest.raises(ValueError):
        WaterPipeline.from_state(state)


@pytest.mark.parametrize("metadata", [
    {"bad": float("nan")}, {"bad": float("inf")}, {"bad": object()},
    {1: "key không chuỗi"}, {"bad": (1, 2)}, [],
])
def test_strict_metadata_rejected_before_write(tmp_path, metadata):
    matrix, labels = training_example()
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        save_pipeline(WaterPipeline().fit(matrix, labels), path, metadata=metadata)
    assert not path.exists()


def write_checked_payload(path, payload):
    """Ghi byte hỏng có checksum đúng để kiểm tra parser/state, không chỉ hash."""
    path.write_bytes(payload)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(payload).hexdigest(), encoding="utf-8")


@pytest.mark.parametrize("payload", [
    b"not json", b"\xff", b'{"schema_version":1,"schema_version":1}',
    b'{"metadata":{"bad":NaN}}', b'{"metadata":{"bad":Infinity}}',
    b'{"metadata":{"bad":1e999}}', b"[]",
])
def test_strict_json_and_schema_with_matching_checksum(tmp_path, payload):
    path = tmp_path / "invalid.json"
    write_checked_payload(path, payload)
    with pytest.raises(ValueError):
        load_pipeline(path)


def test_checksum_required_corruption_and_expected(tmp_path):
    matrix, labels = training_example()
    path = tmp_path / "model.json"
    checksum = save_pipeline(WaterPipeline().fit(matrix, labels), path)
    with pytest.raises(ValueError):
        load_pipeline(path, "0" * 64)
    for invalid in (True, "bad", "g" * 64):
        with pytest.raises(ValueError):
            load_pipeline(path, invalid)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_pipeline(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.unlink()
    with pytest.raises(ValueError):
        load_pipeline(path, checksum)
    sidecar.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline(path)


def test_invalid_paths_and_sha256_known_answer(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    matrix, labels = training_example()
    pipeline = WaterPipeline().fit(matrix, labels)
    for invalid in ("", None, tmp_path, tmp_path / "missing" / "model.json"):
        with pytest.raises(ValueError):
            save_pipeline(pipeline, invalid)
    for invalid in ("", None, tmp_path, tmp_path / "absent.json"):
        with pytest.raises(ValueError):
            load_pipeline(invalid)
        with pytest.raises(ValueError):
            sha256_file(invalid)


def test_written_json_contains_only_finite_values(tmp_path):
    matrix, labels = training_example()
    path = tmp_path / "model.json"
    save_pipeline(WaterPipeline(standardize=True).fit(matrix, labels), path)
    state = json.loads(path.read_text(encoding="utf-8"))
    assert json.dumps(state, allow_nan=False)
