import abc
import logging
import random
from typing import Callable

from ..config import default_config
from ..llm_backend import LLMBackend
from ..types import Conversation, LLMAgentBase
from .memory_base import MemoryBase
from .models import (
    BDIChangeIntention,
    BDIData,
    BDIFullChange,
    BDINoChanges,
    BDIResponse,
    FactResponse,
    MemoryRecord,
    PruneFactsResponse,
)


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
