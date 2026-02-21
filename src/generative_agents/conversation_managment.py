import abc
import asyncio
import logging
from typing import Generator, Iterable

import networkx as nx
import numpy as np

from .agent import Conversation, LLMConversationAgent, Utterance, default_config


class ConversationSelectorABC(abc.ABC):

    @abc.abstractmethod
    def generate_epoch_pairs(
        self,
    ) -> Iterable[list[tuple[LLMConversationAgent, LLMConversationAgent]]]:
        """Each tick of the iterable provides a list of agent pairs, which should have a conversation together.
        Conversations in the single tick are performed concurrently.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the selector to the initialization state."""
        pass


class BFSFrontierGraph[T]:
    def __init__(self, graph: "nx.Graph[T]", sources: Iterable[T]) -> None:
        self._graph = graph
        self._sources = sources
        self._layers = list(
            nx.bfs_layers(graph, sources=sources)
        )  # type: list[list[T]]

    def get_core_nodes(self, distance: int) -> list[T]:
        """Nodes completely within the given distance from the source nodes."""
        return [node for nodes in self._layers[: distance + 1] for node in nodes]

    def get_frontier_extended_graph(self, distance: int) -> "nx.Graph[T]":
        """Returns a graph containing complete subgraph of given distance. The graph is extended to include adjacent edges to the nodes in the specified distance.
        Distance starts from 0, which is the source nodes.
        """
        core_nodes = self.get_core_nodes(distance)
        graph = self._graph.subgraph(core_nodes).copy()

        graph.add_edges_from(self._graph.edges(core_nodes))
        return graph


class GeneralParallelSelectorBase(ConversationSelectorABC, abc.ABC):
    """Base class for creating parallel conversation selector. Algorithm is expecting structure generator implementation,
    which is used to generate the conversation pairs in each epoch. From the structure, nodes are selected randomly.
    For each selected node, a random neighbor is selected to form a conversation pair until there are no more possible choices.
    Each agent is selected at most once in single epoch."""

    def __init__(
        self,
        seed: np.random.Generator | None = None,
    ):
        self._generated_epochs = 0
        self.__seed = seed or np.random.default_rng()

    @abc.abstractmethod
    def _get_generator_structure(self) -> "nx.Graph[LLMConversationAgent]":
        """Structure used as a source for the conversation pairs. Only nodes in the structure are considered for the conversation pairs."""
        pass

    def generate_epoch_pairs(
        self,
    ) -> Generator[list[tuple[LLMConversationAgent, LLMConversationAgent]], None, None]:
        # remove agents from the structure during the progression of the algorithm
        structure = self._get_generator_structure()

        agents: list[LLMConversationAgent] = list(structure.nodes)
        self.__seed.shuffle(agents)

        conversation_pairs: list[tuple[LLMConversationAgent, LLMConversationAgent]] = []

        while len(agents):
            agent1 = agents.pop()
            if not structure.has_node(agent1):
                continue
            neighbor_agents: list[LLMConversationAgent] = list(
                structure.neighbors(agent1)
            )
            if len(neighbor_agents) == 0:
                continue
            agent2: LLMConversationAgent = self.__seed.choice(neighbor_agents)  # type: ignore[arg-type]

            conversation_pairs.append((agent1, agent2))
            structure.remove_node(agent1)
            structure.remove_node(agent2)

        yield conversation_pairs
        self._generated_epochs += 1

    def reset(self) -> None:
        self._generated_epochs = 0


class InformationSpreadConversationSelector(GeneralParallelSelectorBase):
    """Only nodes consistent with possible spread of information are considered for selection.
    It is strcitly following BFS distance from the seed nodes. That means a complete subgraph withing the epoch distance is considered to own measured information.
    The graph is extended to include adjacent edges to the nodes in the specified distance, simulating the frontier of the information spread.
    This measure is to restrict the conversations to only those agents, which are relevant for the information spread.
    """

    def __init__(
        self,
        structure: "nx.Graph[LLMConversationAgent]",
        seed_nodes: Iterable[LLMConversationAgent],
        seed: np.random.Generator | None = None,
    ):
        super().__init__(seed)
        for node in structure.nodes:
            assert isinstance(
                node, LLMConversationAgent
            ), f"Graph node must be LLMConversationAgent, got {type(node)}"

        for node in seed_nodes:
            assert isinstance(
                node, LLMConversationAgent
            ), f"Seed node must be LLMConversationAgent, got {type(node)}"

        self.__bfs_frontier = BFSFrontierGraph(graph=structure, sources=seed_nodes)

    def _get_generator_structure(self) -> "nx.Graph[LLMConversationAgent]":
        return self.__bfs_frontier.get_frontier_extended_graph(self._generated_epochs)


class FullParallelConversationSelector(GeneralParallelSelectorBase):
    """Selects the conversation pairs in a parallel manner. In each epoch, every agent has at most one conversation with a randomly selected neighbor."""

    def __init__(
        self,
        structure: "nx.Graph[LLMConversationAgent]",
        seed: np.random.Generator | None = None,
    ):
        super().__init__(seed)
        for node in structure.nodes:
            assert isinstance(
                node, LLMConversationAgent
            ), f"Graph node must be LLMConversationAgent, got {type(node)}"

        self.__structure = structure

    def _get_generator_structure(self) -> "nx.Graph[LLMConversationAgent]":
        return self.__structure


class ConversationRandomRestrictionAdapter(ConversationSelectorABC):
    """Adapter for any other conversation selector, which randomly restricts the conversations done in parallel.
    This is to adjust the conversations further in case the single epoch consists of too many conversations.
    """

    def __init__(
        self,
        base_selector: ConversationSelectorABC,
        selection_probability: float,
        seed: np.random.Generator | None = None,
    ):
        self.__base_selector = base_selector
        self.__selection_probability = selection_probability
        self.__seed = seed or np.random.default_rng()

    def generate_epoch_pairs(
        self,
    ) -> Generator[list[tuple[LLMConversationAgent, LLMConversationAgent]], None, None]:
        """Selects a random subset of conversations, controled using probability."""
        for pairs_in_parallel in self.__base_selector.generate_epoch_pairs():
            yield [
                pair
                for pair in pairs_in_parallel
                if self.__seed.random() < self.__selection_probability
            ]

    def reset(self) -> None:
        self.__base_selector.reset()


class SequentialConversationSelector(ConversationSelectorABC):
    """Selects the conversation pairs in a sequential manner. In a single epoch, each agent has a conversation with every other adjacent agent once.
    This effectively means every edge is executed once in the epoch.
    The order of the conversations is random, so is the order of the pair of agents in the conversation.
    There is a possibility to provide an initial conversation pairs, which will be used in the first epoch.
    """

    def __init__(
        self,
        structure: "nx.Graph[LLMConversationAgent]",
        seed: np.random.Generator | None = None,
        initial_conversation: list[
            tuple[LLMConversationAgent, LLMConversationAgent]
        ] = [],
    ):
        self.__generated_epochs = 0
        for node in structure.nodes:
            assert isinstance(
                node, LLMConversationAgent
            ), f"Graph node must be LLMConversationAgent, got {type(node)}"

        self.__structure = structure
        self.__initial_conversation = initial_conversation
        self.seed = seed or np.random.default_rng()

    def generate_epoch_pairs(
        self,
    ) -> Generator[list[tuple[LLMConversationAgent, LLMConversationAgent]], None, None]:
        initialization = (
            self.__initial_conversation if self.__generated_epochs == 0 else []
        )
        initialization_set = {(first, second) for first, second in initialization} | {
            (second, first) for first, second in initialization
        }

        conversation_pairs: list[tuple[LLMConversationAgent, LLMConversationAgent]] = (
            list(self.__structure.edges)
        )
        self.seed.shuffle(conversation_pairs)
        conversation_pairs = [
            (first, second) if self.seed.random() < 0.5 else (second, first)
            for (first, second) in conversation_pairs
        ]

        total_pairs = initialization + [
            pair for pair in conversation_pairs if pair not in initialization_set
        ]
        for pair in total_pairs:
            yield [pair]

        self.__generated_epochs += 1

    def reset(self) -> None:
        self.__generated_epochs = 0


INITIAL_GREETING_ACTION = "initial greeting"


class ConversationManager:
    def __init__(
        self,
        conversation_selector: ConversationSelectorABC,
        max_conversation_utterances: int,
        logger: logging.Logger | None = None,
    ):
        self.max_conversation_utterances = max_conversation_utterances
        self.conversation_selector = conversation_selector
        self._logger = logger

    async def __generate_conversation(
        self, agent1: LLMConversationAgent, agent2: LLMConversationAgent
    ) -> Conversation:
        conversation: Conversation = []

        for _ in range(self.max_conversation_utterances):
            if len(conversation) == 0:
                message = await agent1.start_conversation(agent2) or ""
                conversation.append(
                    (
                        agent1,
                        Utterance(
                            actions=[INITIAL_GREETING_ACTION],
                            selected_action=INITIAL_GREETING_ACTION,
                            message=message,
                            is_conversation_finished=False,
                        ),
                    )
                )
            else:
                next_utterance = await agent1.generate_next_turn(
                    agent2, list(conversation)
                )
                conversation.append((agent1, next_utterance))
                if next_utterance.is_conversation_finished:
                    break
            agent2, agent1 = agent1, agent2

        return conversation

    async def __process_conversation_pair(
        self, agent1: LLMConversationAgent, agent2: LLMConversationAgent
    ) -> None:
        await asyncio.gather(
            agent1.pre_conversation_hook(agent2),
            agent2.pre_conversation_hook(agent1),
        )
        conversation = await self.__generate_conversation(agent1, agent2)
        if self._logger:
            self._logger.debug(
                f"Conversation between agents {agent1.data.full_name} - {agent2.data.full_name}",
                extra={
                    "conversation": default_config().conversation_to_text(conversation)
                },
            )
        await asyncio.gather(
            agent1.post_conversation_hook(agent2, conversation, logger=self._logger),
            agent2.post_conversation_hook(agent1, conversation, logger=self._logger),
        )

    async def run_simulation_epoch(self) -> None:
        for pairs in self.conversation_selector.generate_epoch_pairs():
            await asyncio.gather(
                *[self.__process_conversation_pair(*pair) for pair in pairs]
            )

    def reset_epochs(self) -> None:
        self.conversation_selector.reset()
