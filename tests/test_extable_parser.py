"""Tests for extable/parser.py — ExperimentTableParser."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from extable.parser import ExperimentTableParser


class TestExperimentTableParserInit:
    def test_init_without_llm_client(self):
        parser = ExperimentTableParser()
        assert parser._llm is None

    def test_init_with_llm_client(self):
        mock_client = MagicMock()
        parser = ExperimentTableParser(llm_client=mock_client)
        assert parser._llm is mock_client


class TestExperimentTableParserRegexParse:
    def test_regex_parse_empty_table(self):
        parser = ExperimentTableParser()
        result = parser._regex_parse([], "title")
        assert result == {"tables": []}

    def test_regex_parse_single_row(self):
        parser = ExperimentTableParser()
        result = parser._regex_parse([["Header"]], "title")
        assert result == {"tables": []}

    def test_regex_parse_with_metric_columns(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Accuracy", "F1"],
            ["BERT", "91.2", "90.1"],
            ["GPT", "92.0", "91.5"],
        ]
        result = parser._regex_parse(table_data, "BERT vs GPT")
        assert "tables" in result
        assert len(result["tables"]) == 1
        t = result["tables"][0]
        assert t["caption"] == "BERT vs GPT"
        assert len(t["metrics"]) == 4
        assert t["models"] == ["BERT", "GPT"]

    def test_regex_parse_identifies_datasets(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Dataset", "Accuracy"],
            ["Model-A", "SQuAD", "85.1"],
            ["Model-B", "SQuAD", "84.5"],
        ]
        result = parser._regex_parse(table_data, "dataset test")
        t = result["tables"][0]
        assert "SQuAD" in t["datasets"]

    def test_regex_parse_identifies_our_method(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Accuracy"],
            ["BERT", "91.2"],
            ["Ours", "93.5"],
        ]
        result = parser._regex_parse(table_data, "our method")
        t = result["tables"][0]
        # Ours should be identified as the best
        assert t["ours_best"]["value"] == 93.5

    def test_regex_parse_extracts_numeric_values(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Score"],
            ["Alpha", "99.5%"],
            ["Beta", "98.3%"],
        ]
        result = parser._regex_parse(table_data, "scores")
        t = result["tables"][0]
        metric_vals = [m["value"] for m in t["metrics"]]
        assert 99.5 in metric_vals
        assert 98.3 in metric_vals

    def test_regex_parse_with_no_header_metrics(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Col1", "Col2"],
            ["A", "85.1", "84.0"],
            ["B", "86.2", "85.1"],
            ["C", "87.0", "86.0"],
            ["D", "88.1", "87.0"],
        ]
        result = parser._regex_parse(table_data, "numeric cols")
        t = result["tables"][0]
        assert len(t["metrics"]) >= 1

    def test_regex_parse_baselines_excludes_our_method(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Accuracy"],
            ["BERT", "91.2"],
            ["Ours", "93.5"],
            ["RoBERTa", "92.0"],
        ]
        result = parser._regex_parse(table_data, "comparison")
        baselines = result["tables"][0]["baselines"]
        assert "Ours" not in baselines


class TestExperimentTableParserParseToStruct:
    def test_parse_table_falls_back_to_regex_when_no_llm(self):
        parser = ExperimentTableParser()
        table_data = [
            ["Model", "Accuracy"],
            ["BERT", "91.2"],
        ]
        result = parser.parse_table_to_struct(table_data, "test")
        assert "tables" in result

    def test_parse_table_uses_regex_on_llm_exception(self):
        mock_client = MagicMock()
        mock_client.parse.side_effect = Exception("LLM error")
        parser = ExperimentTableParser(llm_client=mock_client)
        table_data = [
            ["Model", "Accuracy"],
            ["BERT", "91.2"],
        ]
        result = parser.parse_table_to_struct(table_data, "test")
        assert "tables" in result


class TestExperimentTableParserLLMParse:
    def test_llm_parse_returns_regex_fallback_on_error(self):
        mock_client = MagicMock()
        parser = ExperimentTableParser(llm_client=mock_client)
        table_data = [
            ["Model", "Accuracy"],
            ["BERT", "91.2"],
        ]
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            result = parser._llm_parse(table_data, "test", "qwen3.5-plus")
            assert "tables" in result

    def test_llm_parse_strips_markdown_fence(self):
        mock_client = MagicMock()
        parser = ExperimentTableParser(llm_client=mock_client)
        table_data = [["Model", "Accuracy"], ["BERT", "91.2"]]

        mock_response = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"tables\": []}\n```"
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = parser._llm_parse(table_data, "test", "qwen3.5-plus")
            assert result == {"tables": []}

    def test_llm_parse_strips_json_markdown_fence(self):
        mock_client = MagicMock()
        parser = ExperimentTableParser(llm_client=mock_client)
        table_data = [["Model", "Accuracy"], ["BERT", "91.2"]]

        mock_response = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"tables\": [{\"caption\":\"test\",\"metrics\":[{\"name\":\"accuracy\",\"value\":91.2}],\"datasets\":[],\"models\":[\"BERT\"],\"baselines\":{},\"ours_best\":{\"value\":91.2,\"dataset\":\"\",\"metric\":\"accuracy\"}}]}\n```"
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = parser._llm_parse(table_data, "test", "qwen3.5-plus")
            assert len(result["tables"]) == 1
            assert result["tables"][0]["metrics"][0]["name"] == "accuracy"
