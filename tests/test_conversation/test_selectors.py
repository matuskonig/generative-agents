from typing import Iterable
from unittest.mock import MagicMock

import networkx as nx
import pytest
from numpy.random import Generator

from generative_agents.agent import LLMConversationAgent
from generative_agents.conversation_managment import (
    ConversationRandomRestrictionAdapter,
    FullParallelConversationSelector,
    InformationSpreadConversationSelector,
    SequentialConversationSelector,
)


def create_mock_agent(name: str) -> LLMConversationAgent:
    agent = MagicMock(spec=LLMConversationAgent)
    agent.data = type("AgentData", (), {"full_name": name})()
    agent.get_agent_introduction_message = MagicMock(
        return_value=lambda: f"Hello, I am {name}"
    )
    return agent


class TestSequentialConversationSelector:
    def test_empty_graph(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        selector = SequentialConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert pairs == []

    def test_single_edge(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        selector = SequentialConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert len(pairs) == 1

    def test_multiple_edges(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        agent_c = create_mock_agent("C")
        graph.add_edges_from([(agent_a, agent_b), (agent_b, agent_c)])
        selector = SequentialConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert len(pairs) == 2
        assert pairs == [[(agent_b, agent_c)], [(agent_b, agent_a)]]

    def test_initial_conversation_only_first_epoch(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        initial = [(agent_a, agent_b)]
        selector = SequentialConversationSelector(
            graph, seed=seeded_rng, initial_conversation=initial
        )
        gen = selector.generate_epoch_pairs()
        pairs = list(gen)
        assert pairs == [initial]

    def test_reset_returns_to_initial_state(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        selector = SequentialConversationSelector(graph, seed=seeded_rng)
        list(selector.generate_epoch_pairs())
        list(selector.generate_epoch_pairs())
        selector.reset()
        first_epoch_after_reset: list[
            tuple[LLMConversationAgent, LLMConversationAgent]
        ] = next(iter(selector.generate_epoch_pairs()))
        assert len(first_epoch_after_reset) == 1


class TestFullParallelConversationSelector:
    def test_empty_graph(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert pairs == [[]]

    def test_single_edge(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert len(pairs[0]) == 1

    def test_three_agents_two_edges(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        agent_c = create_mock_agent("C")
        graph.add_edges_from([(agent_a, agent_b), (agent_b, agent_c)])
        selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        assert len(pairs) == 1

    def test_four_agents_two_pairs(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        agent_c = create_mock_agent("C")
        agent_d = create_mock_agent("D")
        graph.add_edges_from([(agent_a, agent_b), (agent_c, agent_d)])
        selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        # Depends on the generator state
        assert len(pairs[0]) == 2

    def test_each_agent_at_most_one_conversation(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        agent_c = create_mock_agent("C")
        agent_d = create_mock_agent("D")
        graph.add_edges_from(
            [
                (agent_a, agent_b),
                (agent_a, agent_c),
                (agent_b, agent_d),
                (agent_c, agent_d),
            ]
        )
        selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        pairs = list(selector.generate_epoch_pairs())
        paired_agents = set()
        for tick in pairs:
            for pair in tick:
                assert pair[0] not in paired_agents
                assert pair[1] not in paired_agents
                paired_agents.add(pair[0])
                paired_agents.add(pair[1])

    def test_reset(self, seeded_rng: Generator) -> None:
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")

        graph1: nx.Graph[LLMConversationAgent] = nx.Graph()
        graph1.add_edge(agent_a, agent_b)
        selector = FullParallelConversationSelector(graph1, seed=seeded_rng)
        gen = selector.generate_epoch_pairs()
        pairs = list(next(iter(gen)))
        assert len(pairs) == 1

        selector.reset()

        graph2: nx.Graph[LLMConversationAgent] = nx.Graph()
        graph2.add_edge(agent_a, agent_b)
        selector2 = FullParallelConversationSelector(graph2, seed=seeded_rng)
        gen2 = selector2.generate_epoch_pairs()
        pairs2 = list(next(iter(gen2)))
        assert len(pairs2) == 1


class TestInformationSpreadConversationSelector:
    def test_seed_nodes_only_epoch_0(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        agent_c = create_mock_agent("C")
        graph.add_edges_from([(agent_a, agent_b), (agent_b, agent_c)])
        seed_nodes = [agent_a]
        selector = InformationSpreadConversationSelector(
            graph, seed_nodes=seed_nodes, seed=seeded_rng
        )
        gen = selector.generate_epoch_pairs()
        epoch = list(gen)
        assert epoch == [[(agent_a, agent_b)]]


class TestConversationRandomRestrictionAdapter:
    def test_probability_1_returns_all_pairs(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        base_selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        adapter = ConversationRandomRestrictionAdapter(
            base_selector, selection_probability=1.0, seed=seeded_rng
        )
        pairs = list(adapter.generate_epoch_pairs())
        assert len(pairs[0]) == 1

    def test_probability_0_returns_no_pairs(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        base_selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        adapter = ConversationRandomRestrictionAdapter(
            base_selector, selection_probability=0.0, seed=seeded_rng
        )
        pairs = list(adapter.generate_epoch_pairs())
        assert pairs[0] == []

    def test_reset_delegates_to_base(self, seeded_rng: Generator) -> None:
        graph: nx.Graph[LLMConversationAgent] = nx.Graph()
        agent_a = create_mock_agent("A")
        agent_b = create_mock_agent("B")
        graph.add_edge(agent_a, agent_b)
        base_selector = FullParallelConversationSelector(graph, seed=seeded_rng)
        adapter = ConversationRandomRestrictionAdapter(
            base_selector, selection_probability=1.0, seed=seeded_rng
        )
        gen = adapter.generate_epoch_pairs()
        list(next(iter(gen)))
        adapter.reset()
        gen2 = adapter.generate_epoch_pairs()
        list(next(iter(gen2)))
        adapter.reset()

        pairs = list(adapter.generate_epoch_pairs())
        assert len(pairs) == 1
