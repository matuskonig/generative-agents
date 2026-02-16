from typing import Sequence, Iterable, Callable
import abc
import logging
import numpy as np
import random

from .llm_backend import LLMBackend

from .types import (
    LLMAgentBase,
    MemoryRecord,
    MemoryRecordResponse,
    MemoryRecordWithEmbedding,
    BDIData,
    BDIResponse,
    BDINoChanges,
    BDIChangeIntention,
    BDIFullChange,
    FactResponse,
    Conversation,
    PruneFactsResponse,
)
from .config import default_config

# TODO: switch to file-based prompts together with
# TODO: add composite memory manager which allows to combine multiple memory managers
# TODO: make pruning optional

# TODO: add some freetext field to the BDI to support model writing notes, planning and reasoning
# TODO: add possibly something to extend the actions

# TODO: rozbit to na viac filov
# TODO: ako vyriesime generalizaciu a context injcetion do jednotlivych agentov (mimo background)
# TODO: make pruning a standalone configuration


# TODO Affordable memory
class MemoryBase(abc.ABC):
    @abc.abstractmethod
    def current_timestamp(self) -> int:
        pass

    @abc.abstractmethod
    def full_retrieval(self) -> Sequence[MemoryRecord]:
        """Return all facts in the memory as a list of strings."""
        pass

    @abc.abstractmethod
    async def query_retrieval(self, query: str) -> Sequence[MemoryRecord]:
        """Return a list of facts that match the query."""
        pass

    @abc.abstractmethod
    async def store_facts(self, facts: Sequence[MemoryRecordResponse]) -> None:
        """Append new facts to the memory."""

    @abc.abstractmethod
    def remove_facts(self, timestamps: list[int]) -> None:
        pass


class SimpleMemory(MemoryBase):
    def __init__(self) -> None:
        self.__timestamp = 0
        self.__memory: list[MemoryRecord] = []

    def current_timestamp(self) -> int:
        return self.__timestamp

    def full_retrieval(self) -> list[MemoryRecord]:
        return self.__memory

    async def query_retrieval(self, query: str) -> list[MemoryRecord]:
        return self.__memory

    def __get_next_timestamp(self) -> int:
        self.__timestamp += 1
        return self.__timestamp

    async def store_facts(self, facts: Sequence[MemoryRecordResponse]) -> None:
        self.__memory.extend(
            [
                MemoryRecord(
                    timestamp=self.__get_next_timestamp(),
                    text=fact.text,
                    relevance=fact.relevance,
                )
                for fact in facts
            ]
        )

    def remove_facts(self, timestamps: list[int]) -> None:
        self.__memory = [
            record for record in self.__memory if record.timestamp not in timestamps
        ]


def fixed_count_strategy_factory(
    count: int,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        return min(count, len(records))

    return inner


def mean_std_count_strategy_factory(
    std_coef: float = 0.5,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        if len(records) == 0:
            return 0
        scores = np.array([score for score, _ in records])
        mean = np.mean(scores)
        std_dev = np.std(scores)
        treshold = mean + std_dev * std_coef
        return sum(1 for score in scores if score >= treshold)

    return inner


def top_std_count_strategy_factory(
    std_coef: float = 1.0,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        if len(records) == 0:
            return 0
        scores = np.array([score for score, _ in records])
        max = np.max(scores)
        std_dev = np.std(scores)
        treshold = max - std_dev * std_coef
        return sum(1 for score in scores if score >= treshold)

    return inner


class EmbeddingMemory(MemoryBase):
    def __init__(
        self,
        context: LLMBackend,
        count_selector: Callable[[Sequence[tuple[float, MemoryRecord]]], int],
        time_weight: float = 1.0,
        time_smoothing: float = 0.7,
        relevance_weight: float = 1.0,
        similairity_weight: float = 1.0,
    ) -> None:
        self.__context = context
        self.__count_selector = count_selector
        self.__memory: list[MemoryRecordWithEmbedding] = []
        self.__timestamp = 0

        self.__time_weight = time_weight
        self.__time_smoothing = time_smoothing
        self.__relevance_weight = relevance_weight
        self.__similarity_weight = similairity_weight

    def __get_next_timestamp(self) -> int:
        self.__timestamp += 1
        return self.__timestamp

    def current_timestamp(self) -> int:
        return self.__timestamp

    def full_retrieval(self) -> list[MemoryRecordWithEmbedding]:
        return self.__memory

    def __get_memory_record_score(
        self, query_emb: np.ndarray, record: MemoryRecordWithEmbedding
    ) -> float:
        time_similarity: float = (
            record.timestamp / self.__timestamp
        ) ** self.__time_smoothing
        cosine_similarity: float = np.dot(record.embedding, query_emb) / (
            np.linalg.norm(record.embedding) * np.linalg.norm(query_emb)
        )
        return (
            self.__time_weight * time_similarity
            + self.__relevance_weight * record.relevance
            + self.__similarity_weight * cosine_similarity
        )

    async def query_retrieval(self, query: str) -> list[MemoryRecordWithEmbedding]:
        if len(self.__memory) == 0:
            return []

        query_embedding = await self.__context.embed_text(query)
        scored_records = sorted(
            [
                (self.__get_memory_record_score(query_embedding, record), record)
                for record in self.__memory
            ],
            reverse=True,
        )

        selected_count = self.__count_selector(scored_records)
        return [record for _, record in scored_records[:selected_count]]

    async def store_facts(self, facts: Iterable[MemoryRecordResponse]) -> None:
        embeddings = await self.__context.embed_text([fact.text for fact in facts])
        self.__memory.extend(
            [
                MemoryRecordWithEmbedding(
                    timestamp=self.__get_next_timestamp(),
                    text=fact.text,
                    relevance=fact.relevance,
                    embedding=embedding,
                )
                for fact, embedding in zip(facts, embeddings)
            ]
        )

    def remove_facts(self, timestamps: list[int]) -> None:
        self.__memory = [
            record for record in self.__memory if record.timestamp not in timestamps
        ]


class MemoryManagerBase(abc.ABC):
    @abc.abstractmethod
    def get_tagged_full_memory(self, with_full_memory_record: bool = False) -> str:
        pass

    @abc.abstractmethod
    async def get_tagged_memory_by_query(self, query: str) -> str:
        pass

    @abc.abstractmethod
    async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
        pass

    @abc.abstractmethod
    async def post_conversation_hook(
        self,
        other_agent: LLMAgentBase,
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ) -> None:
        pass


class SimpleMemoryManager(MemoryManagerBase):
    def __init__(
        self, memory: MemoryBase, agent: LLMAgentBase, context: LLMBackend
    ) -> None:
        self.memory = memory
        self._agent = agent
        self._context = context

    def _get_memory_tag(self, memory_string: str) -> str:
        return f"<memory>{memory_string}</memory>"

    def get_tagged_full_memory(self, with_full_memory_record: bool = False) -> str:
        return self._get_memory_tag(
            "\n".join(
                [
                    (
                        record.model_dump_json(include=set(MemoryRecord.model_fields))
                        if with_full_memory_record
                        else record.text
                    )
                    for record in self.memory.full_retrieval()
                ]
            )
        )

    async def get_tagged_memory_by_query(self, query: str) -> str:
        return self._get_memory_tag(
            "\n".join(
                ([record.text for record in await self.memory.query_retrieval(query)])
            )
        )

    async def _add_new_memory(
        self, other_agent: LLMAgentBase, conversation: Conversation
    ) -> None:
        prompt = default_config().get_conversation_summary_prompt(
            agent_full_name=self._agent.data.full_name,
            agent_introduction=await self._agent.get_agent_introduction_message(),
            other_agent_full_name=other_agent.data.full_name,
            conversation_string=default_config().conversation_to_tagged_text(
                conversation
            ),
            memory_string=self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(FactResponse.model_json_schema()),
        )
        result = await self._context.get_structued_response(
            prompt,
            response_format=FactResponse,
            params=default_config().get_factual_llm_params(),
        )
        await self.memory.store_facts(result.facts)

    async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
        pass

    async def post_conversation_hook(
        self,
        other_agent: LLMAgentBase,
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ) -> None:
        await self._add_new_memory(other_agent, conversation)
        if logger:
            logger.debug(
                f"Memory state of {self._agent.data.full_name} after conversation with {other_agent.data.full_name}",
                extra={
                    "memory": "\n".join(
                        [
                            fact.model_dump_json(include=set(MemoryRecord.model_fields))
                            for fact in self.memory.full_retrieval()
                        ]
                    )
                },
            )


def get_fact_removal_probability_factory(
    max_prob_coef: float,
) -> Callable[[int, MemoryRecord], float]:
    def inner(current_timestamp: int, fact: MemoryRecord) -> float:
        linear_prob = 1 - (fact.timestamp / current_timestamp)
        return max_prob_coef * linear_prob

    return inner


class BDIMemoryManager(MemoryManagerBase):
    """Simple memory manager elevating Belief-Desire-Intention (BDI) architecture."""

    def __init__(
        self,
        memory: MemoryBase,
        agent: LLMAgentBase,
        context: LLMBackend,
        memory_removal_probability: Callable[[int, MemoryRecord], float] | None = None,
    ):
        self.memory = memory
        self._agent = agent
        self._context = context
        self.__bdi_data: BDIData | None = None
        self.memory_removal_probability = memory_removal_probability

    def _get_memory_tag(
        self,
        memory_string: str,
        with_desires: bool = False,
        with_intention: bool = True,
    ) -> str:
        memory = f"<memory>{memory_string}</memory>"
        desires = (
            f'<desires>{"\n".join(self.__bdi_data.desires)}</desires>'
            if with_desires and self.__bdi_data
            else ""
        )
        intention = (
            f"<intention>{self.__bdi_data.intention}</intention>"
            if with_intention and self.__bdi_data
            else ""
        )
        return f"{memory}{desires}{intention}"

    def get_tagged_full_memory(self, with_full_memory_record: bool = False) -> str:
        return self._get_memory_tag(
            "\n".join(
                [
                    (
                        record.model_dump_json(include=set(MemoryRecord.model_fields))
                        if with_full_memory_record
                        else record.text
                    )
                    for record in self.memory.full_retrieval()
                ],
            )
        )

    async def get_tagged_memory_by_query(self, query: str) -> str:
        return self._get_memory_tag(
            "\n".join(
                ([record.text for record in await self.memory.query_retrieval(query)])
            )
        )

    async def _initialize_bdi(self) -> None:
        if not self.__bdi_data:
            prompt = default_config().get_bdi_init_prompt(
                self._agent.data.full_name,
                await self._agent.get_agent_introduction_message(),
                self.get_tagged_full_memory(with_full_memory_record=True),
                response_format=str(BDIData.model_json_schema()),
            )
            result = await self._context.get_structued_response(
                prompt,
                BDIData,
                params=default_config().get_neutral_default_llm_params(),
            )
            self.__bdi_data = result

    async def _update_bdi(
        self, second_agent: LLMAgentBase, conversation: Conversation
    ) -> None:
        prompt = default_config().get_bdi_update_prompt(
            self._agent.data.full_name,
            await self._agent.get_agent_introduction_message(),
            second_agent.data.full_name,
            default_config().conversation_to_tagged_text(conversation),
            self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(BDIResponse.model_json_schema()),
        )
        result = await self._context.get_structued_response(
            prompt,
            response_format=BDIResponse,
            params=default_config().get_neutral_default_llm_params(),
        )
        if isinstance(result.data, BDINoChanges):
            return
        elif isinstance(result.data, BDIChangeIntention) and self.__bdi_data:
            self.__bdi_data.intention = result.data.intention
        elif isinstance(result.data, BDIFullChange):
            self.__bdi_data = BDIData(
                desires=result.data.desires,
                intention=result.data.intention,
            )

    async def _prune_old_memory(self) -> None:
        if not self.memory_removal_probability:
            return

        facts_to_prune = [
            fact
            for fact in self.memory.full_retrieval()
            if random.random()
            <= self.memory_removal_probability(self.memory.current_timestamp(), fact)
        ]
        timestamps: set[int] = {fact.timestamp for fact in facts_to_prune}
        if len(facts_to_prune) == 0:
            return

        memory = "\n".join(
            [
                fact.model_dump_json(include=set(MemoryRecord.model_fields))
                for fact in facts_to_prune
            ]
        )
        prompt = default_config().get_memory_prune_prompt(
            self._agent.data.full_name,
            await self._agent.get_agent_introduction_message(),
            memory,
            str(PruneFactsResponse.model_json_schema()),
        )
        result = await self._context.get_structued_response(
            prompt,
            PruneFactsResponse,
            params=default_config().get_neutral_default_llm_params(),
        )
        validated_timestamps = {
            timestamp
            for timestamp in result.timestamps_to_remove
            if timestamp in timestamps
        }
        if len(validated_timestamps):
            self.memory.remove_facts(list(validated_timestamps))

    async def _add_new_memory(
        self, other_agent: LLMAgentBase, conversation: Conversation
    ) -> None:
        prompt = default_config().get_conversation_summary_prompt(
            agent_full_name=self._agent.data.full_name,
            agent_introduction=await self._agent.get_agent_introduction_message(),
            other_agent_full_name=other_agent.data.full_name,
            conversation_string=default_config().conversation_to_tagged_text(
                conversation
            ),
            memory_string=self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(FactResponse.model_json_schema()),
        )
        result = await self._context.get_structued_response(
            prompt,
            response_format=FactResponse,
            params=default_config().get_factual_llm_params(),
        )
        await self.memory.store_facts(result.facts)

    async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
        await self._initialize_bdi()

    async def post_conversation_hook(
        self,
        other_agent: LLMAgentBase,
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ) -> None:
        await self._add_new_memory(other_agent, conversation)
        await self._prune_old_memory()
        await self._update_bdi(other_agent, conversation)
        if logger:
            logger.debug(
                f"Memory state of {self._agent.data.full_name} after conversation with {other_agent.data.full_name}",
                extra={
                    "memory": "\n".join(
                        [
                            fact.model_dump_json(include=set(MemoryRecord.model_fields))
                            for fact in self.memory.full_retrieval()
                        ]
                    )
                },
            )
            logger.debug(
                f"BDI state of {self._agent.data.full_name} after conversation with {other_agent.data.full_name}",
                extra={"bdi": self.__bdi_data},
            )
