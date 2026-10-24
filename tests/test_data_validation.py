"""Dữ liệu tổng hợp để kiểm chứng CSV/ID/ô thiếu và làm sạch trước split."""

import copy

import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_validation import clean_training_data, read_csv_records, validate_records


def make_record(**overrides):
    """Tạo mẫu tổng hợp đủ chín cột, không phải dữ liệu thực nghiệm."""
    record = {name: str(index + 1) for index, name in enumerate(FEATURE_COLUMNS)}
    record.update(overrides)
    return record


def csv_text(columns, rows):
    return ",".join(columns) + "\n" + "\n".join(",".join(row) for row in rows)


def test_csv_preserves_strings_bom_and_reordered_columns(tmp_path):
    columns = ["sample_id", TARGET_COLUMN] + list(reversed(FEATURE_COLUMNS))
    text = "\ufeff" + csv_text(columns, [["0007", "ignored"] + [str(i) for i in range(9, 0, -1)]])
    path = tmp_path / "input.csv"
    path.write_text(text, encoding="utf-8")
    records = read_csv_records(path)
    assert records == read_csv_records(text, from_text=True)
    assert records[0]["sample_id"] == "0007"
    result = validate_records(records)
    assert result["X"] == [[1., 2., 3., 4., 5., 6., 7., 8., 9.]]
    assert result["row_ids"] == [0]
    assert result["y"] is None
    assert "Potability" in result["warnings"][0]["warning"]


@pytest.mark.parametrize("columns", [FEATURE_COLUMNS[:-1], FEATURE_COLUMNS + ["extra"],
    FEATURE_COLUMNS + ["ph"], ["PH"] + FEATURE_COLUMNS[1:],
    [" ph"] + FEATURE_COLUMNS[1:], ["ph "] + FEATURE_COLUMNS[1:]])
def test_header_rejected_before_value_validation(columns):
    with pytest.raises(ValueError):
        read_csv_records(csv_text(columns, [["1"] * len(columns)]), from_text=True)


@pytest.mark.parametrize("text", ["", "\ufeff", ",".join(FEATURE_COLUMNS),
    ",".join(FEATURE_COLUMNS) + "\n\n",
    csv_text(FEATURE_COLUMNS, [["1"] * 8]),
    csv_text(FEATURE_COLUMNS, [["1"] * 10]),
    ",".join(FEATURE_COLUMNS) + '\n"unterminated'])
def test_empty_and_malformed_csv_are_file_errors(text):
    with pytest.raises(ValueError):
        read_csv_records(text, from_text=True)


def test_bad_utf8_and_missing_path_are_file_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        read_csv_records(path)
    with pytest.raises(ValueError):
        read_csv_records(tmp_path / "missing.csv")


def test_training_header_requires_label_and_forbids_sample_id():
    for columns in (FEATURE_COLUMNS, FEATURE_COLUMNS + [TARGET_COLUMN, "sample_id"]):
        with pytest.raises(ValueError):
            read_csv_records(csv_text(columns, [["1"] * len(columns)]), from_text=True, training=True)
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    records = read_csv_records(csv_text(columns, [["1"] * 10]), from_text=True, training=True)
    assert validate_records(records, training=True)["y"] == [1]


@pytest.mark.parametrize("token", ["NULL", "N/A", "nan", "NAN", "inf", "-inf", "Infinity",
                                  "1e999", "water", True, float("nan"), float("inf")])
def test_invalid_tokens_reject_only_their_row(token):
    result = validate_records([make_record(ph=token), make_record(sample_id="0002")])
    assert result["row_ids"] == [1]
    assert result["errors"][0]["input_row_id"] == 0
    assert result["valid_records"][0]["sample_id"] == "0002"


def test_missing_strip_counts_warnings_and_input_not_mutated():
    record = make_record(ph=" NaN ", Hardness=" ", Solids=None, Chloramines=" 4.5 ", sample_id=" 0001 ")
    before = copy.deepcopy(record)
    result = validate_records([record])
    assert result["X"][0][:4] == [None, None, None, 4.5]
    assert result["valid_records"][0]["sample_id"] == " 0001 "
    assert result["audit_summary"]["missing_counts"]["ph"] == 1
    assert result["warnings"][0]["input_row_id"] == 0
    assert record == before


def test_all_missing_rejected_and_numeric_sample_id_zero_preserved():
    missing = {name: None for name in FEATURE_COLUMNS}
    result = validate_records([missing, make_record(sample_id=0)])
    assert result["row_ids"] == [1]
    assert result["valid_records"][0]["sample_id"] == 0
    assert len(result["errors"]) == 1


@pytest.mark.parametrize("label", [None, "", "NaN", "2", "1.0", "nan", True, -1, 2, .5, float("nan")])
def test_invalid_training_labels(label):
    result = validate_records([make_record(Potability=label)], training=True)
    assert result["X"] == []
    assert result["y"] == []
    assert result["audit_summary"]["invalid_count"] == 1


def test_valid_training_labels_and_audit_duplicate_groups():
    result = validate_records([make_record(Potability=" 0 "), make_record(Potability=1.0)], training=True)
    assert result["y"] == [0, 1]
    assert result["audit_summary"]["class_counts"] == {0: 1, 1: 1}
    assert result["audit_summary"]["duplicate_groups"] == [[0, 1]]


@pytest.mark.parametrize("records", [[], {}, [None], [make_record(extra=1)], [{"ph": 1}]])
def test_direct_api_structure_is_strict(records):
    with pytest.raises(ValueError):
        validate_records(records)


def test_clean_duplicates_normalizes_missing_keeps_first_and_counts():
    X = [[None] + [1] * 8, [" NaN "] + ["1"] * 8, [2] * 9, [""] + [1.] * 8]
    before = copy.deepcopy(X)
    result = clean_training_data(X, [0, 0, 1, 0], [7, 2, 9, 4])
    assert result["X"] == [[None] + [1.] * 8, [2.] * 9]
    assert result["y"] == [0, 1]
    assert result["row_ids"] == [7, 9]
    assert result["removed_row_ids"] == [2, 4]
    assert result["duplicate_groups"] == [[7, 2, 4]]
    assert result["audit_summary"]["missing_counts"]["ph"] == 1
    assert result["audit_summary"]["class_counts"] == {0: 1, 1: 1}
    assert result["audit_summary"]["input_count"] == 4
    assert result["audit_summary"]["output_count"] == 2
    assert result["audit_summary"]["removed_count"] == 2
    assert X == before


def test_conflict_errors_even_after_same_label_duplicate():
    with pytest.raises(ValueError, match="khác nhãn"):
        clean_training_data([[None] + [1] * 8, [""] + [1] * 8, ["NaN"] + [1] * 8],
                            [0, 0, 1], [10, 11, 12])


@pytest.mark.parametrize("X,y,ids", [([], [], []), ([[1] * 9], [], [0]),
    ([[1] * 8], [0], [0]), ([[None] * 9], [0], [0]),
    ([[1] * 9] * 2, [0, 0], [1, 1]), ([[1] * 9], [0], [-1]),
    ([[1] * 9], [0], [True]), ([[1] * 9], [2], [0])])
def test_clean_invalid_inputs(X, y, ids):
    with pytest.raises(ValueError):
        clean_training_data(X, y, ids)
