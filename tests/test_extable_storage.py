"""Tests for extable/storage.py — ExperimentDB."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from extable.storage import ExperimentDB


@pytest.fixture
def db(tmp_path):
    """Fresh ExperimentDB backed by a temporary file."""
    db_path = tmp_path / "test_extable.db"
    return ExperimentDB(db_path=str(db_path))


class TestExperimentDBInit:
    def test_init_creates_parent_directory(self, tmp_path):
        db_path = tmp_path / "subdir" / "test.db"
        db = ExperimentDB(db_path=str(db_path))
        assert Path(db_path).parent.exists()

    def test_init_creates_tables(self, db):
        conn = sqlite3.connect(db.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "extable_papers" in table_names
        assert "extable_tables" in table_names
        conn.close()


class TestExperimentDBPaperOps:
    def test_add_paper_inserts_row(self, db):
        db.add_paper("paper-1", "Test Paper")
        conn = sqlite3.connect(db.db_path)
        row = conn.execute(
            "SELECT * FROM extable_papers WHERE paper_uid=?", ("paper-1",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "paper-1"
        assert row[1] == "Test Paper"

    def test_add_paper_idempotent(self, db):
        db.add_paper("paper-1", "First")
        db.add_paper("paper-1", "Second")
        conn = sqlite3.connect(db.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM extable_papers WHERE paper_uid=?", ("paper-1",)
        ).fetchone()[0]
        conn.close()
        assert count == 1


class TestExperimentDBTableOps:
    def test_add_table_returns_id(self, db):
        db.add_paper("paper-1", "Test Paper")
        table_struct = {
            "caption": "Results",
            "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": ["SQuAD"],
            "models": ["BERT"],
            "baselines": {},
            "ours_best": {"value": 91.2, "dataset": "SQuAD", "metric": "accuracy"},
        }
        raw_table = [["Model", "Accuracy"], ["BERT", "91.2"]]
        table_id = db.add_table("paper-1", table_struct, raw_table)
        assert isinstance(table_id, str)
        assert len(table_id) == 36

    def test_add_table_stores_all_fields(self, db):
        db.add_paper("paper-1", "Test Paper")
        table_struct = {
            "caption": "Main Results",
            "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": ["SQuAD", "GLUE"],
            "models": ["BERT", "RoBERTa"],
            "baselines": {"BERT": 90.5},
            "ours_best": {"value": 91.2, "dataset": "SQuAD", "metric": "accuracy"},
        }
        raw_table = [["Model", "Accuracy"], ["BERT", "91.2"]]
        table_id = db.add_table("paper-1", table_struct, raw_table)
        tables = db.get_paper_tables("paper-1")
        assert len(tables) == 1
        t = tables[0]
        assert t["caption"] == "Main Results"
        assert len(t["metrics"]) == 1
        assert t["metrics"][0]["value"] == 91.2
        assert "SQuAD" in t["datasets"]
        assert "BERT" in t["models"]
        assert t["baselines"] == {"BERT": 90.5}
        assert t["ours_best"]["value"] == 91.2

    def test_get_paper_tables_returns_empty_for_unknown_paper(self, db):
        result = db.get_paper_tables("ghost")
        assert result == []


class TestExperimentDBSearch:
    def test_search_tables_by_metric(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "t1", "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["Model", "Acc"], ["BERT", "91.2"]])
        results = db.search_tables(metric="accuracy")
        assert len(results) == 1

    def test_search_tables_by_dataset(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "t1", "metrics": [],
            "datasets": ["SQuAD"], "models": [], "baselines": {}, "ours_best": {},
        }, [["Dataset", "Value"], ["SQuAD", "85"]])
        results = db.search_tables(dataset="squad")
        assert len(results) == 1

    def test_search_tables_by_model(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "t1", "metrics": [],
            "datasets": [], "models": ["BERT"], "baselines": {}, "ours_best": {},
        }, [["Model", "Score"], ["BERT", "91"]])
        results = db.search_tables(model="bert")
        assert len(results) == 1

    def test_search_tables_by_min_value(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "t1",
            "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["Model", "Acc"], ["BERT", "91.2"]])
        results = db.search_tables(min_value=90.0)
        assert len(results) == 1
        results_low = db.search_tables(min_value=95.0)
        assert len(results_low) == 0

    def test_search_tables_combined_filters(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "t1",
            "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": ["SQuAD"], "models": ["BERT"],
            "baselines": {}, "ours_best": {},
        }, [["Model", "Acc"], ["BERT", "91.2"]])
        results = db.search_tables(metric="accuracy", dataset="squad", model="bert")
        assert len(results) == 1

    def test_search_tables_returns_all_when_no_filters(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_paper("p2", "Paper 2")
        db.add_table("p1", {
            "caption": "t1", "metrics": [{"name": "accuracy", "value": 91.0}],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "A"], ["A", "91"]])
        db.add_table("p2", {
            "caption": "t2", "metrics": [{"name": "f1", "value": 88.0}],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "F"], ["B", "88"]])
        results = db.search_tables()
        assert len(results) == 2


class TestExperimentDBExport:
    def test_export_to_csv_all_tables(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_table("p1", {
            "caption": "Results",
            "metrics": [{"name": "accuracy", "value": 91.2}],
            "datasets": ["SQuAD"],
            "models": ["BERT"],
            "baselines": {},
            "ours_best": {"value": 91.2, "dataset": "SQuAD", "metric": "accuracy"},
        }, [["Model", "Acc"], ["BERT", "91.2"]])
        csv = db.export_to_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "paper-uid" not in lines[1]  # not the ghost uid
        assert "p1" in lines[1]

    def test_export_to_csv_specific_paper(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_paper("p2", "Paper 2")
        db.add_table("p1", {
            "caption": "T1", "metrics": [],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "V"], ["A", "1"]])
        db.add_table("p2", {
            "caption": "T2", "metrics": [],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "V"], ["B", "2"]])
        csv = db.export_to_csv(paper_uid="p1")
        lines = csv.strip().split("\n")
        assert len(lines) == 2
        assert "p1" in lines[1]
        assert "p2" not in lines[1]


class TestExperimentDBStats:
    def test_stats_empty_db(self, db):
        s = db.stats()
        assert s["papers"] == 0
        assert s["tables"] == 0

    def test_stats_after_adding_papers_and_tables(self, db):
        db.add_paper("p1", "Paper 1")
        db.add_paper("p2", "Paper 2")
        db.add_table("p1", {
            "caption": "t1", "metrics": [],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "V"], ["A", "1"]])
        db.add_table("p2", {
            "caption": "t2", "metrics": [],
            "datasets": [], "models": [], "baselines": {}, "ours_best": {},
        }, [["M", "V"], ["B", "2"]])
        s = db.stats()
        assert s["papers"] == 2
        assert s["tables"] == 2
