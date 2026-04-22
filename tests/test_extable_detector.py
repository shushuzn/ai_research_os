"""Tests for extable/detector.py — TableDetector."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extable.detector import TableDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _fitz_available(monkeypatch):
    """Simulate fitz being importable."""
    monkeypatch.setitem(sys.modules, "fitz", MagicMock())


@pytest.fixture
def _fitz_unavailable(monkeypatch):
    """Simulate fitz/pymupdf NOT being importable."""
    # Block re-import by removing from sys.modules
    monkeypatch.delitem(sys.modules, "fitz", raising=False)
    monkeypatch.delitem(sys.modules, "pymupdf", raising=False)
    # Create detector with _has_fitz=False (instance attribute set in __init__)
    detector = TableDetector()
    detector._has_fitz = False
    return detector


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestTableDetectorInit:
    def test_init_with_fitz_available(self, _fitz_available):
        detector = TableDetector()
        assert detector._has_fitz is True

    def test_init_without_fitz(self, _fitz_unavailable):
        detector = _fitz_unavailable
        assert detector._has_fitz is False


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------

class TestTableDetectorDetect:
    def test_detect_tables_returns_empty_when_no_fitz(self, _fitz_unavailable):
        detector = _fitz_unavailable
        result = detector.detect_tables(0, pdf_path=Path("fake.pdf"))
        assert result == []

    def test_detect_tables_handles_int_page(self, _fitz_available):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__len__.return_value = 1

        with patch("fitz.open") as mock_open:
            mock_open.return_value = mock_doc
            detector = TableDetector()
            result = detector.detect_tables(0, pdf_path=Path("fake.pdf"))
            assert isinstance(result, list)

    def test_detect_tables_skips_non_text_blocks(self, _fitz_available):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {"type": 1, "bbox": [0, 0, 100, 100]},  # image block
            ]
        }
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__len__.return_value = 1

        with patch("fitz.open") as mock_open:
            mock_open.return_value = mock_doc
            detector = TableDetector()
            result = detector.detect_tables(0, pdf_path=Path("fake.pdf"))
            assert result == []


# ---------------------------------------------------------------------------
# _extract_table_from_block (no fitz needed)
# ---------------------------------------------------------------------------

class TestTableDetectorExtractTable:
    def test_extract_table_from_block_empty_lines(self):
        detector = TableDetector()
        block = {"lines": []}
        result = detector._extract_table_from_block(MagicMock(), block)
        assert result == []

    def test_extract_table_from_block_single_row(self):
        detector = TableDetector()
        block = {
            "lines": [
                {
                    "bbox": [0, 100, 200, 120],
                    "spans": [
                        {"bbox": [10, 102, 50, 118], "text": "Model"},
                        {"bbox": [60, 102, 120, 118], "text": "Accuracy"},
                    ]
                }
            ]
        }
        result = detector._extract_table_from_block(MagicMock(), block)
        assert len(result) == 1
        assert result[0] == ["Model", "Accuracy"]

    def test_extract_table_from_block_multiple_rows(self):
        detector = TableDetector()
        block = {
            "lines": [
                {
                    "bbox": [0, 100, 200, 120],
                    "spans": [{"bbox": [10, 102, 50, 118], "text": "Model"}, {"bbox": [60, 102, 120, 118], "text": "Accuracy"}]
                },
                {
                    "bbox": [0, 130, 200, 150],
                    "spans": [{"bbox": [10, 132, 50, 148], "text": "BERT"}, {"bbox": [60, 132, 120, 148], "text": "91.2"}]
                },
            ]
        }
        result = detector._extract_table_from_block(MagicMock(), block)
        assert len(result) == 2
        assert result[0] == ["Model", "Accuracy"]
        assert result[1] == ["BERT", "91.2"]

    def test_extract_table_skips_empty_spans(self):
        detector = TableDetector()
        block = {
            "lines": [
                {
                    "bbox": [0, 100, 200, 120],
                    "spans": [{"bbox": [10, 102, 50, 118], "text": ""}, {"bbox": [60, 102, 120, 118], "text": "Accuracy"}]
                }
            ]
        }
        result = detector._extract_table_from_block(MagicMock(), block)
        assert len(result) == 1
        assert result[0] == ["Accuracy"]


# ---------------------------------------------------------------------------
# _is_experiment_table (no fitz needed)
# ---------------------------------------------------------------------------

class TestTableDetectorIsExperiment:
    def test_is_experiment_table_too_few_rows(self):
        detector = TableDetector()
        result = detector._is_experiment_table([["Header"]])
        assert result is False

    def test_is_experiment_table_with_metrics_and_numbers(self):
        detector = TableDetector()
        table_data = [
            ["Model", "Accuracy", "F1"],
            ["BERT", "91.2", "90.1"],
            ["GPT", "92.0", "91.5"],
            ["CNN", "89.0", "88.0"],
        ]
        result = detector._is_experiment_table(table_data)
        assert result is True

    def test_is_experiment_table_with_datasets_and_numbers(self):
        detector = TableDetector()
        table_data = [
            ["Dataset", "SQuAD", "GLUE", "MNLI", "QNLI", "SST-2"],
            ["Model-A", "85.1", "78.2", "80.3", "82.1", "91.0"],
            ["Model-B", "84.5", "77.9", "79.8", "81.5", "90.5"],
            ["Model-C", "86.2", "79.1", "81.0", "83.2", "92.1"],
            ["Model-D", "83.8", "76.5", "78.9", "80.2", "89.8"],
        ]
        result = detector._is_experiment_table(table_data)
        assert result is True

    def test_is_experiment_table_not_experiment(self):
        detector = TableDetector()
        table_data = [
            ["Name", "Value"],
            ["Alice", "100"],
        ]
        result = detector._is_experiment_table(table_data)
        assert result is False


# ---------------------------------------------------------------------------
# extract_all_tables
# ---------------------------------------------------------------------------

class TestTableDetectorExtractAll:
    def test_extract_all_tables_returns_empty_when_no_fitz(self, _fitz_unavailable):
        detector = _fitz_unavailable
        result = detector.extract_all_tables(Path("fake.pdf"))
        assert result == []

    def test_extract_all_tables_calls_detect_per_page(self, _fitz_available):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.return_value = mock_page

        with patch("fitz.open") as mock_open:
            mock_open.return_value = mock_doc
            with patch.object(TableDetector, "detect_tables", return_value=[]) as mock_detect:
                detector = TableDetector()
                _ = detector.extract_all_tables(Path("fake.pdf"), max_pages=2)
                assert mock_detect.call_count == 2

    def test_extract_all_tables_respects_max_pages(self, _fitz_available):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__len__.return_value = 10
        mock_doc.__getitem__.return_value = mock_page

        with patch("fitz.open") as mock_open:
            mock_open.return_value = mock_doc
            with patch.object(TableDetector, "detect_tables", return_value=[]) as mock_detect:
                detector = TableDetector()
                detector.extract_all_tables(Path("fake.pdf"), max_pages=3)
                assert mock_detect.call_count == 3
