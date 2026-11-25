"""Kiểm chứng thuật toán tự cài bằng đáp án tính tay."""

import math
import pytest

from src.comparison_models import NumericPreprocessor


def test_numeric_fit_does_not_read_validation():
    processor = NumericPreprocessor().fit([[1, 8], [3, 8], [None, 8]])
    assert processor.state["medians"] == [2, 8]
    assert processor.state["means"] == [2, 8]
    assert processor.state["scales"] == [math.sqrt(2 / 3), 1]
    snapshot = repr(processor.state)
    assert processor.transform([[2, 8]]) == [[0, 0]]
    processor.transform([[1000, 1000]])
    assert repr(processor.state) == snapshot
