"""Tests for viz/d3_renderer.py."""

from unittest.mock import MagicMock


class TestD3ForceGraphInit:
    def test_init_with_kg_manager(self):
        from viz.d3_renderer import D3ForceGraph
        mock_kg = MagicMock()
        renderer = D3ForceGraph(mock_kg)
        assert renderer.kg is mock_kg

    def test_init_default_creates_kg_manager(self):
        from viz.d3_renderer import D3ForceGraph
        renderer = D3ForceGraph()
        from kg.manager import KGManager
        assert isinstance(renderer.kg, KGManager)


class TestD3ForceGraphToJson:
    def _make_kg(self, nodes, edges):
        kg = MagicMock()
        kg.get_all_nodes.return_value = nodes
        kg.find_papers_by_tag.return_value = nodes
        kg.get_edges_by_node.return_value = edges
        return kg

    def test_paper_uids_returns_ego_subgraph(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        _ = {
            "nodes": [{"id": "p1", "label": "Paper 1", "type": "Paper", "entity_id": ""}],
            "edges": [{"id": "e1", "source_id": "p1", "target_id": "p2", "relation_type": "cites", "weight": 1.0}],
        }
        kg.find_papers_by_tag.return_value = []
        result = D3ForceGraph(kg).to_json(paper_uids=["p1"])
        assert "nodes" in result
        assert "links" in result
        kg.find_papers_by_tag.assert_not_called()

    def test_tag_filters_papers(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        papers = [
            {"id": "p1", "label": "Tagged Paper", "type": "Paper", "entity_id": ""},
        ]
        kg.find_papers_by_tag.return_value = papers
        kg.get_edges_by_node.return_value = []
        result = D3ForceGraph(kg).to_json(tag="machine-learning")
        kg.find_papers_by_tag.assert_called_once_with("machine-learning")
        assert len(result["nodes"]) == 1

    def test_tag_no_results(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        kg.find_papers_by_tag.return_value = []
        kg.get_edges_by_node.return_value = []
        result = D3ForceGraph(kg).to_json(tag="nonexistent")
        assert result["nodes"] == []
        assert result["links"] == []

    def test_full_graph_uses_get_all_nodes(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        nodes = [
            {"id": "p1", "label": "Paper 1", "type": "Paper", "entity_id": ""},
        ]
        kg.get_all_nodes.return_value = nodes
        kg.get_edges_by_node.return_value = []
        result = D3ForceGraph(kg).to_json()
        kg.get_all_nodes.assert_called_once()
        assert len(result["nodes"]) == 1

    def test_max_nodes_limits_output(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        kg.find_papers_by_tag.return_value = []
        nodes = [{"id": f"n{i}", "label": f"Node {i}", "type": "Paper", "entity_id": ""} for i in range(10)]
        kg.get_all_nodes.return_value = nodes
        kg.get_edges_by_node.return_value = []
        result = D3ForceGraph(kg).to_json(max_nodes=3)
        assert len(result["nodes"]) <= 3

    def test_node_format(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        nodes = [{"id": "p1", "label": "LongLabel", "type": "Paper", "entity_id": "E1"}]
        kg.get_all_nodes.return_value = nodes
        kg.get_edges_by_node.return_value = []
        result = D3ForceGraph(kg).to_json()
        node = result["nodes"][0]
        assert node["id"] == "p1"
        assert node["label"] == "LongLabel"
        assert node["type"] == "Paper"
        assert node["entity_id"] == "E1"

    def test_link_format(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        nodes = [{"id": "p1", "label": "P1", "type": "Paper", "entity_id": ""},
                 {"id": "p2", "label": "P2", "type": "Paper", "entity_id": ""}]
        kg.get_all_nodes.return_value = nodes
        edges = [{"id": "e1", "source_id": "p1", "target_id": "p2", "relation_type": "cites", "weight": 0.5}]
        kg.get_edges_by_node.return_value = edges
        result = D3ForceGraph(kg).to_json()
        # direction="both" is called for each node; each call returns the same edge
        # so we get 2 link entries (one per call)
        assert len(result["links"]) == 2
        link = result["links"][0]
        assert link["source"] == "p1"
        assert link["target"] == "p2"
        assert link["relation"] == "cites"
        assert link["weight"] == 0.5

    def test_edge_direction_both(self):
        from viz.d3_renderer import D3ForceGraph
        kg = MagicMock()
        kg.get_all_nodes.return_value = [{"id": "p1", "label": "P1", "type": "Paper", "entity_id": ""}]
        kg.get_edges_by_node.return_value = []
        D3ForceGraph(kg).to_json()
        kg.get_edges_by_node.assert_called_with("p1", direction="both")
