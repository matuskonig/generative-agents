from typing import Iterable
import asyncio
import networkx as nx
import abc
from .agent import LLMAgent, Utterance, Conversation, conversation_to_text
import numpy as np
import logging


class ConversationSelectorABC(abc.ABC):

    @abc.abstractmethod
    def generate_epoch_pairs(self) -> Iterable[list[tuple[LLMAgent, LLMAgent]]]:
        """Each tick of the iterable provides a list of agent pairs, which should have a conversation together.
        The order of the agents in the pair matters, as the first agent is the conversation initializer.
        In addition, conversations in the single tick are performed concurrently.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets the internal state of the selector to the initialization state."""
        pass


class SequentialConversationSelector(ConversationSelectorABC):
    """Selects the conversation pairs in a sequential manner. In a single epoch, each agent has a conversation with every other adjacent agent once.
    The order of the conversations is random, so is the order of the pair of agents in the conversation.
    There is a possibility to provide an initial conversation pairs, which will be used in the first epoch.
    """

    def __init__(
        self,
        structure: nx.Graph,  # must be working with nodes of type LLMAgent
        seed: np.random.Generator | None = None,
        initial_conversation: list[tuple[LLMAgent, LLMAgent]] = [],
    ):
        self.__generated_epochs = 0
        self.__structure = structure
        self.__initial_conversation = initial_conversation
        self.seed = seed or np.random.default_rng()

    def generate_epoch_pairs(self):
        initialization = (
            self.__initial_conversation if self.__generated_epochs == 0 else []
        )
        initialization_set = {(first, second) for first, second in initialization} | {
            (second, first) for first, second in initialization
        }

        conversation_pairs: list[tuple[LLMAgent, LLMAgent]] = list(
            self.__structure.edges
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

    def reset(self):
        self.__generated_epochs = 0


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

    async def __generate_conversation(self, agent1: LLMAgent, agent2: LLMAgent):
        conversation: Conversation = []

        for _ in range(self.max_conversation_utterances):
            if len(conversation) == 0:
                message = await agent1.start_conversation(agent2)
                conversation.append(
                    (agent1, Utterance(message=message, is_ending_conversation=False))
                )
            else:
                next_utterance = await agent1.generate_next_turn(agent2, conversation)
                conversation.append((agent1, next_utterance))
                if next_utterance.is_ending_conversation:
                    break
            agent2, agent1 = agent1, agent2

        return conversation

    async def __process_conversation_pair(self, agent1: LLMAgent, agent2: LLMAgent):
        conversation = await self.__generate_conversation(agent1, agent2)
        if self._logger:
            self._logger.debug(
                f"Conversation between agents {agent1.data.full_name} - {agent2.data.full_name}",
                extra={"conversation": conversation_to_text(conversation)},
            )
        await asyncio.gather(
            agent1.adjust_memory_after_conversation(
                agent2, conversation, logger=self._logger
            ),
            agent2.adjust_memory_after_conversation(
                agent1, conversation, logger=self._logger
            ),
        )

    async def run_simulation_epoch(self):
        self.conversation_selector.reset()
        for pairs in self.conversation_selector.generate_epoch_pairs():
            await asyncio.gather(
                *[self.__process_conversation_pair(*pair) for pair in pairs]
            )
