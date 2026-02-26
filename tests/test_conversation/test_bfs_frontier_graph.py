import networkx as nx

from generative_agents.conversation_managment import BFSFrontierGraph


class TestBFSFrontierGraphCoreNodes:
    def test_single_node(self) -> None:
        graph: nx.Graph[str] = nx.Graph()
        graph.add_node("A")
        frontier = BFSFrontierGraph(graph, sources=["A"])
        assert frontier.get_core_nodes(0) == ["A"]

    def test_chain_graph(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        assert frontier.get_core_nodes(0) == ["A"]
        assert set(frontier.get_core_nodes(1)) == {"A", "B"}
        assert set(frontier.get_core_nodes(2)) == {"A", "B", "C"}
        assert set(frontier.get_core_nodes(3)) == {"A", "B", "C", "D"}

    def test_star_graph(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("A", "C"), ("A", "D")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        assert frontier.get_core_nodes(0) == ["A"]
        assert set(frontier.get_core_nodes(1)) == {"A", "B", "C", "D"}

    def test_multiple_sources(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("X", "Y")])
        frontier = BFSFrontierGraph(graph, sources=["A", "X"])
        assert set(frontier.get_core_nodes(0)) == {"A", "X"}
        assert set(frontier.get_core_nodes(1)) == {"A", "B", "X", "Y"}
        assert set(frontier.get_core_nodes(2)) == {"A", "B", "C", "X", "Y"}

    def test_disconnected_graph(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("C", "D")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        assert frontier.get_core_nodes(0) == ["A"]
        assert set(frontier.get_core_nodes(1)) == {"A", "B"}

    def test_distance_larger_than_graph(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        result = frontier.get_core_nodes(10)
        assert set(result) == {"A", "B"}


class TestBFSFrontierGraphExtendedGraph:
    def test_single_node(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_node("A")
        frontier = BFSFrontierGraph(graph, sources=["A"])
        extended = frontier.get_frontier_extended_graph(0)
        assert nx.utils.graphs_equal(extended, nx.subgraph(graph, ["A"]))

    def test_chain_graph_distance_0(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        extended = frontier.get_frontier_extended_graph(0)
        assert nx.utils.graphs_equal(extended, nx.subgraph(graph, ["A", "B"]))

    def test_chain_graph_distance_1(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        extended = frontier.get_frontier_extended_graph(1)
        assert nx.utils.graphs_equal(extended, nx.subgraph(graph, ["A", "B", "C"]))

    def test_chain_graph_distance_2(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        frontier = BFSFrontierGraph(graph, sources=["A"])
        extended = frontier.get_frontier_extended_graph(2)
        assert nx.utils.graphs_equal(extended, nx.subgraph(graph, ["A", "B", "C", "D"]))

    def test_multiple_sources(self) -> None:
        graph: nx.Graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("X", "Y")])
        frontier = BFSFrontierGraph(graph, sources=["A", "X"])
        extended = frontier.get_frontier_extended_graph(0)
        assert nx.utils.graphs_equal(extended, nx.subgraph(graph, ["A", "B", "X", "Y"]))
