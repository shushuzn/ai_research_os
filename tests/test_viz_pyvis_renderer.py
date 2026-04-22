"""Tests for viz/pyvis_renderer.py."""

import pytest
from unittest.mock import MagicMock, patch


class TestKGVizRendererInit:
    def test_init_with_kg_manager(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        renderer = KGVizRenderer(mock_kg)
        assert renderer.kg is mock_kg

    def test_init_default_creates_kg_manager(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer()
        from kg.manager import KGManager
        assert isinstance(renderer.kg, KGManager)


class TestKGVizRendererBaseNet:
    def test_base_net_creates_network(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer(MagicMock())
        with patch("viz.pyvis_renderer.Network") as MockNet:
            mock_net = MagicMock()
            mock_net.generate_html.return_value = "<html></html>"
            MockNet.return_value = mock_net
            result = renderer._base_net(height="500px", bgcolor="#f0f0f0")
            MockNet.assert_called_once_with(height="500px", bgcolor="#f0f0f0", directed=False, select_menu=True)
            mock_net.set_options.assert_called_once()
            assert result is mock_net


class TestKGVizRendererAddNodes:
    def test_add_nodes_from_graph(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer(MagicMock())
        mock_net = MagicMock()
        graph = {
            "nodes": [
                {"id": "p1", "label": "Paper One", "type": "Paper"},
                {"id": "t1", "label": "ml", "type": "Tag"},
            ],
            "edges": [
                {"id": "e1", "source_id": "p1", "target_id": "t1", "relation_type": "mentions", "weight": 1.0},
            ],
        }
        renderer._add_nodes_from_graph(mock_net, graph)
        assert mock_net.add_node.call_count == 2
        mock_net.add_edge.assert_called_once_with("p1", "t1", title="mentions", width=1.0)

    def test_add_nodes_from_graph_skips_duplicate_edges(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer(MagicMock())
        mock_net = MagicMock()
        graph = {
            "nodes": [{"id": "p1", "label": "P1", "type": "Paper"}],
            "edges": [
                {"id": "e1", "source_id": "p1", "target_id": "p2", "relation_type": "cites", "weight": 1.0},
                {"id": "e1", "source_id": "p1", "target_id": "p2", "relation_type": "cites", "weight": 1.0},
            ],
        }
        renderer._add_nodes_from_graph(mock_net, graph)
        assert mock_net.add_node.call_count == 1
        assert mock_net.add_edge.call_count == 1

    def test_add_nodes_truncates_long_labels(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer(MagicMock())
        mock_net = MagicMock()
        long_label = "A" * 100
        graph = {
            "nodes": [{"id": "p1", "label": long_label, "type": "Paper"}],
            "edges": [],
        }
        renderer._add_nodes_from_graph(mock_net, graph)
        call = mock_net.add_node.call_args
        assert len(call.kwargs["label"]) <= 60  # label truncated


class TestKGVizRendererPaperGraph:
    def test_paper_graph_uses_pyvis_when_available(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        renderer = KGVizRenderer(mock_kg)
        subgraph = {
            "nodes": [{"id": "p1", "label": "P1", "type": "Paper"}],
            "edges": [],
            "center": "p1",
        }
        with patch("viz.pyvis_renderer._HAS_PYVIS", True):
            with patch("viz.pyvis_renderer.Network") as MockNet:
                mock_net = MagicMock()
                mock_net.generate_html.return_value = "<html>paper</html>"
                MockNet.return_value = mock_net
                with patch("kg.queries.KGQueries") as MockKGQueries:
                    mock_q = MagicMock()
                    mock_q.get_paper_subgraph.return_value = subgraph
                    MockKGQueries.return_value = mock_q
                    result = renderer.paper_graph("p1", depth=2)
                    mock_q.get_paper_subgraph.assert_called_once_with("p1", depth=2)
                    assert result == "<html>paper</html>"

    def test_paper_graph_fallback_when_no_pyvis(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        mock_kg.get_node_by_entity.return_value = {"id": "n1", "label": "Paper One", "type": "Paper"}
        mock_kg.find_neighbors.return_value = []
        renderer = KGVizRenderer(mock_kg)
        with patch("viz.pyvis_renderer._HAS_PYVIS", False):
            result = renderer.paper_graph("p1", depth=1)
        assert "Paper One" in result
        assert "not found" not in result

    def test_paper_graph_fallback_not_found(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        mock_kg.get_node_by_entity.return_value = None
        renderer = KGVizRenderer(mock_kg)
        with patch("viz.pyvis_renderer._HAS_PYVIS", False):
            result = renderer.paper_graph("nonexistent", depth=1)
        assert "not found" in result


class TestKGVizRendererTagGraph:
    def test_tag_graph_uses_pyvis_when_available(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        renderer = KGVizRenderer(mock_kg)
        ecosystem = {
            "nodes": [{"id": "p1", "label": "P1", "type": "Paper"}],
            "edges": [],
        }
        with patch("viz.pyvis_renderer._HAS_PYVIS", True):
            with patch("viz.pyvis_renderer.Network") as MockNet:
                mock_net = MagicMock()
                mock_net.generate_html.return_value = "<html>tag</html>"
                MockNet.return_value = mock_net
                with patch("kg.queries.KGQueries") as MockKGQueries:
                    mock_q = MagicMock()
                    mock_q.get_tag_ecosystem.return_value = ecosystem
                    MockKGQueries.return_value = mock_q
                    result = renderer.tag_graph("ml")
                    mock_q.get_tag_ecosystem.assert_called_once_with("ml")
                    assert result == "<html>tag</html>"

    def test_tag_graph_fallback(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        mock_kg.find_papers_by_tag.return_value = [
            {"id": "p1", "label": "Paper 1", "type": "Paper"},
        ]
        renderer = KGVizRenderer(mock_kg)
        with patch("viz.pyvis_renderer._HAS_PYVIS", False):
            result = renderer.tag_graph("ml")
        assert "ml" in result
        assert "Paper 1" in result

    def test_tag_graph_fallback_no_papers(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        mock_kg.find_papers_by_tag.return_value = []
        renderer = KGVizRenderer(mock_kg)
        with patch("viz.pyvis_renderer._HAS_PYVIS", False):
            result = renderer.tag_graph("nonexistent")
        assert "No papers found" in result


class TestKGVizRendererFullGraph:
    def test_full_graph_uses_pyvis_when_available(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        renderer = KGVizRenderer(mock_kg)
        export = {
            "nodes": [{"id": "p1", "label": "P1", "type": "Paper"}],
            "edges": [],
        }
        with patch("viz.pyvis_renderer._HAS_PYVIS", True):
            with patch("viz.pyvis_renderer.Network") as MockNet:
                mock_net = MagicMock()
                mock_net.generate_html.return_value = "<html>full</html>"
                MockNet.return_value = mock_net
                with patch("kg.queries.KGQueries") as MockKGQueries:
                    mock_q = MagicMock()
                    mock_q.export_graph_json.return_value = export
                    MockKGQueries.return_value = mock_q
                    result = renderer.full_graph(max_nodes=100)
                    mock_q.export_graph_json.assert_called_once()
                    assert result == "<html>full</html>"

    def test_full_graph_respects_max_nodes(self):
        from viz.pyvis_renderer import KGVizRenderer
        mock_kg = MagicMock()
        renderer = KGVizRenderer(mock_kg)
        export = {
            "nodes": [{"id": f"p{i}", "label": f"P{i}", "type": "Paper"} for i in range(10)],
            "edges": [],
        }
        with patch("viz.pyvis_renderer._HAS_PYVIS", True):
            with patch("viz.pyvis_renderer.Network") as MockNet:
                mock_net = MagicMock()
                mock_net.generate_html.return_value = "<html>"
                MockNet.return_value = mock_net
                with patch("kg.queries.KGQueries") as MockKGQueries:
                    mock_q = MagicMock()
                    mock_q.export_graph_json.return_value = export
                    MockKGQueries.return_value = mock_q
                    renderer.full_graph(max_nodes=3)
                    assert mock_net.add_node.call_count == 3

    def test_full_graph_fallback_no_pyvis(self):
        from viz.pyvis_renderer import KGVizRenderer
        renderer = KGVizRenderer(MagicMock())
        with patch("viz.pyvis_renderer._HAS_PYVIS", False):
            result = renderer.full_graph()
        assert "PyVis not installed" in result
