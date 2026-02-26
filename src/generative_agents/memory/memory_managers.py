import abc
import logging
from typing import Mapping, Protocol, Sequence, TypeVar

import numpy as np

from ..config import default_config
from ..llm_backend import LLMBackendBase
from ..types import Conversation, LLMAgentBase
from .memory_base import MemoryBase
from .models import (
    BDIChangeIntention,
    BDIData,
    BDIFullChange,
    BDINoChanges,
    BDIResponse,
    BuildInSourceType,
    FactResponse,
    MemoryQueryFilter,
    MemoryRecord,
    MemoryRecordResponse,
    PruneFactsResponse,
)


class MemoryManagerBase(abc.ABC):
    @abc.abstractmethod
    def get_tagged_full_memory(
        self,
        *,
        with_full_memory_record: bool = False,
        query_filter: MemoryQueryFilter | None = None,
    ) -> str:
        pass

    @abc.abstractmethod
    async def get_tagged_memory_by_query(
        self,
        query: str,
        *,
        query_filter: MemoryQueryFilter | None = None,
    ) -> str:
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


BehaviorType = TypeVar("BehaviorType", bound="CompositeBehaviorFactoryBase.Impl")


class CompositeBehaviorFactoryBase(abc.ABC):
    """Factory pattern for composing multiple memory behaviors.

    Uses decorator/strategy pattern to combine different memory management behaviors
    (e.g., conversation memory updating, BDI planning, forgetting) into a single
    CompositeBehaviorMemoryManager. Each behavior implements pre/post conversation hooks.
    This makes the behaviors effectively a plugins.

    The Impl subclass defines the actual behavior, while the factory handles
    instantiation with required dependencies.
    """

    @classmethod
    @abc.abstractmethod
    def get_impl_type(
        cls,
    ) -> "type[CompositeBehaviorFactoryBase.Impl]": ...

    @abc.abstractmethod
    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "CompositeBehaviorFactoryBase.Impl":
        pass

    class Impl(abc.ABC):
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

        @abc.abstractmethod
        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            """Additional data to be included in string representation of the memory."""
            pass


class ConversationMemoryUpdatingBehavior(CompositeBehaviorFactoryBase):
    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "ConversationMemoryUpdatingBehavior.Impl":
        return ConversationMemoryUpdatingBehavior.Impl(memory, owner, agent, context)

    @classmethod
    def get_impl_type(
        cls,
    ) -> type["ConversationMemoryUpdatingBehavior.Impl"]:
        return ConversationMemoryUpdatingBehavior.Impl

    class Impl(CompositeBehaviorFactoryBase.Impl):
        def __init__(
            self,
            memory: MemoryBase,
            owner: MemoryManagerBase,
            agent: LLMAgentBase,
            context: LLMBackendBase,
        ):
            self.memory = memory
            self._owner = owner
            self._agent = agent
            self._context = context

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            pass

        async def _add_new_memory(
            self, other_agent: LLMAgentBase, conversation: Conversation
        ) -> None:
            memory_string = self._owner.get_tagged_full_memory(
                with_full_memory_record=True
            )
            prompt = default_config().get_conversation_summary_prompt(
                agent_full_name=self._agent.data.full_name,
                agent_introduction=await self._agent.get_agent_introduction_message(),
                other_agent_full_name=other_agent.data.full_name,
                conversation_string=default_config().conversation_to_tagged_text(
                    conversation
                ),
                memory_string=memory_string,
                response_format=str(FactResponse.model_json_schema()),
            )
            result = await self._context.get_structued_response(
                prompt,
                response_format=FactResponse,
                params=default_config().get_factual_llm_params(),
            )
            await self.memory.store_facts(
                result.facts,
                source=BuildInSourceType.Conversation(
                    other_agent=other_agent.data.full_name
                ),
            )

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: Conversation,
            logger: logging.Logger | None = None,
        ) -> None:
            await self._add_new_memory(other_agent, conversation)

        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            return None


class UnitaryAgentNoteUpdatingBehavior(CompositeBehaviorFactoryBase):
    """You store only a single note about each other agent, which gets updated after every conversation."""

    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "UnitaryAgentNoteUpdatingBehavior.Impl":
        return UnitaryAgentNoteUpdatingBehavior.Impl(memory, owner, agent, context)

    @classmethod
    def get_impl_type(
        cls,
    ) -> type["UnitaryAgentNoteUpdatingBehavior.Impl"]:
        return UnitaryAgentNoteUpdatingBehavior.Impl

    class Impl(CompositeBehaviorFactoryBase.Impl):
        def __init__(
            self,
            memory: MemoryBase,
            owner: MemoryManagerBase,
            agent: LLMAgentBase,
            context: LLMBackendBase,
        ):
            self.memory = memory
            self._owner = owner
            self._agent = agent
            self._context = context

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            pass

        def _get_existing_notes(self, other_agent_name: str) -> Sequence[MemoryRecord]:
            query_filter = MemoryQueryFilter(
                source_types=[BuildInSourceType.UnitaryAgentNoteKnowledge],
                predicate=lambda record: isinstance(
                    record.source, BuildInSourceType.UnitaryAgentNoteKnowledge
                )
                and record.source.other_agent == other_agent_name,
            )
            return self.memory.full_retrieval(query_filter)

        async def _update_agent_note(
            self, other_agent: LLMAgentBase, conversation: Conversation
        ) -> None:
            other_agent_name = other_agent.data.full_name

            existing_notes_timestamps = [
                note.timestamp for note in self._get_existing_notes(other_agent_name)
            ]

            omit_other_agent_filter = MemoryQueryFilter(
                predicate=lambda record: (
                    record.source.other_agent == other_agent_name
                    if isinstance(
                        record.source, BuildInSourceType.UnitaryAgentNoteKnowledge
                    )
                    else True
                )
            )
            memory_string = self._owner.get_tagged_full_memory(
                with_full_memory_record=False,
                query_filter=omit_other_agent_filter,
            )

            prompt = default_config().get_agent_note_update_prompt(
                agent_full_name=self._agent.data.full_name,
                agent_introduction=await self._agent.get_agent_introduction_message(),
                other_agent_full_name=other_agent_name,
                conversation_string=default_config().conversation_to_tagged_text(
                    conversation
                ),
                memory_string=memory_string,
            )
            result = await self._context.get_text_response(
                prompt,
                params=default_config().get_factual_llm_params(),
            )

            await self.memory.store_facts(
                [MemoryRecordResponse(text=result, relevance=1.0)],
                source=BuildInSourceType.UnitaryAgentNoteKnowledge(
                    other_agent=other_agent_name
                ),
            )
            if existing_notes_timestamps:
                self.memory.remove_facts(existing_notes_timestamps)

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: Conversation,
            logger: logging.Logger | None = None,
        ) -> None:
            await self._update_agent_note(other_agent, conversation)

        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            return None


class BDIPlanningBehavior(CompositeBehaviorFactoryBase):
    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "BDIPlanningBehavior.Impl":
        return BDIPlanningBehavior.Impl(memory, owner, agent, context)

    @classmethod
    def get_impl_type(cls) -> type["BDIPlanningBehavior.Impl"]:
        return BDIPlanningBehavior.Impl

    class Impl(CompositeBehaviorFactoryBase.Impl):
        def __init__(
            self,
            memory: MemoryBase,
            owner: MemoryManagerBase,
            agent: LLMAgentBase,
            context: LLMBackendBase,
        ):
            self.memory = memory
            self._owner = owner
            self._agent = agent
            self._context = context
            self.__bdi_data: BDIData | None = None

        async def _initialize_bdi(self) -> None:
            if self.__bdi_data:
                return
            memory_string = self._owner.get_tagged_full_memory(
                with_full_memory_record=True
            )
            prompt = default_config().get_bdi_init_prompt(
                self._agent.data.full_name,
                await self._agent.get_agent_introduction_message(),
                memory_string,
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
            memory_string = self._owner.get_tagged_full_memory(
                with_full_memory_record=True
            )
            prompt = default_config().get_bdi_update_prompt(
                self._agent.data.full_name,
                await self._agent.get_agent_introduction_message(),
                second_agent.data.full_name,
                default_config().conversation_to_tagged_text(conversation),
                memory_string,
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
                self.__bdi_data.notes = result.data.notes
                self.__bdi_data.intention = result.data.intention
            elif isinstance(result.data, BDIFullChange):
                self.__bdi_data = BDIData(
                    notes=result.data.notes,
                    desires=result.data.desires,
                    intention=result.data.intention,
                )

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            await self._initialize_bdi()

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: Conversation,
            logger: logging.Logger | None = None,
        ) -> None:
            await self._update_bdi(other_agent, conversation)

        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            if not self.__bdi_data:
                return None
            return {
                "your_current_desires": "\n".join(self.__bdi_data.desires),
                "your_current_intention": self.__bdi_data.intention,
            }


class RecordRemovalProbSelector(Protocol):
    def __call__(
        self, current_timestamp: int, target_memory_record: MemoryRecord
    ) -> float: ...

#TODO: maybe some exponential or something like that ?
def get_record_removal_linear_probability(
    max_prob_coef: float,
) -> "RecordRemovalProbSelector":
    """Creates a linear probability function for memory removal.

    Probability increases linearly from 0 (at current timestamp) to max_prob_coef
    (at timestamp 0). Older memories have higher removal probability.

    Formula: probability = max_prob_coef * (1 - record_timestamp / current_timestamp)

    Args:
        max_prob_coef: Maximum probability coefficient (typically 0.0-1.0)
    """

    def inner(current_timestamp: int, target_memory_record: MemoryRecord) -> float:
        linear_prob = 1 - (target_memory_record.timestamp / current_timestamp)
        return max_prob_coef * linear_prob

    return inner


class MemoryForgettingBehavior(CompositeBehaviorFactoryBase):
    """Probabilistic memory forgetting behavior.

    Implements forgetting by probabilistically selecting memories for pruning based on
    their age (timestamp). Older memories have higher probability of removal. The actual
    removal is validated by LLM to ensure only unimportant memories are deleted.

    The probability function should return higher values for older memories (lower
    timestamps relative to current time). Uses seeded randomness for reproducibility.
    """

    def __init__(
        self,
        get_record_removal_prob: RecordRemovalProbSelector,
        seed: np.random.Generator | None = None,
    ):
        self.get_record_removal_prob = get_record_removal_prob
        self._seed = seed or np.random.default_rng()

    @classmethod
    def get_impl_type(
        cls,
    ) -> type["MemoryForgettingBehavior.Impl"]:
        return MemoryForgettingBehavior.Impl

    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "MemoryForgettingBehavior.Impl":
        return MemoryForgettingBehavior.Impl(
            memory, owner, agent, context, self.get_record_removal_prob, self._seed
        )

    class Impl(CompositeBehaviorFactoryBase.Impl):
        def __init__(
            self,
            memory: MemoryBase,
            owner: MemoryManagerBase,
            agent: LLMAgentBase,
            context: LLMBackendBase,
            get_record_removal_prob: RecordRemovalProbSelector,
            seed: np.random.Generator,
        ):
            self.memory = memory
            self._agent = agent
            self._context = context
            self._get_record_removal_prob = get_record_removal_prob
            self._seed = seed

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            pass

        async def _prune_old_memory(self) -> None:
            records_to_prune = [
                record
                for record in self.memory.full_retrieval(
                    query_filter=MemoryQueryFilter(
                        source_types=[BuildInSourceType.Conversation]
                    )
                )
                if self._seed.random()
                <= self._get_record_removal_prob(
                    self.memory.current_timestamp(), record
                )
            ]
            if len(records_to_prune) == 0:
                return
            timestamps: set[int] = {fact.timestamp for fact in records_to_prune}

            memory = get_memory_string(records_to_prune, with_full_memory_record=True)
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

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: Conversation,
            logger: logging.Logger | None = None,
        ) -> None:
            await self._prune_old_memory()

        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            return None


class ConstantContextBehavior(CompositeBehaviorFactoryBase):
    def __init__(self, instructions: str):
        self._instructions = instructions

    @classmethod
    def get_impl_type(
        cls,
    ) -> type["ConstantContextBehavior.Impl"]:
        return ConstantContextBehavior.Impl

    def instantizate(
        self,
        memory: MemoryBase,
        owner: MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
    ) -> "ConstantContextBehavior.Impl":
        return ConstantContextBehavior.Impl(self._instructions)

    class Impl(CompositeBehaviorFactoryBase.Impl):
        def __init__(self, instructions: str):
            self.instructions = instructions

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            pass

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: Conversation,
            logger: logging.Logger | None = None,
        ) -> None:
            pass

        def get_memory_extension_data(self) -> Mapping[str, str] | None:
            return {"instructions": self.instructions}


def get_memory_string(
    records: Sequence[MemoryRecord], with_full_memory_record: bool = False
) -> str:
    return "\n".join(
        (
            record.model_dump_json(include=set(MemoryRecord.model_fields))
            if with_full_memory_record
            else f"{record.source.tag}: {record.text}"
        )
        for record in records
    )


def construct_tagged_combined_data_string(data: dict[str, str]) -> str:
    return "\n".join(f"<{tag}>{value}</{tag}>" for tag, value in data.items())


class CompositeBehaviorMemoryManager(MemoryManagerBase):
    def __init__(
        self,
        memory: MemoryBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
        behaviors: Sequence[CompositeBehaviorFactoryBase],
    ) -> None:
        self.memory = memory
        self._agent = agent
        self._behaviors = [
            (behavior.instantizate(memory, self, agent, context))
            for behavior in behaviors
        ]

    def _get_tagged_memory_string(
        self,
        records: Sequence[MemoryRecord],
        with_full_memory_record: bool = False,
    ) -> str:
        tagged_data = {
            key: value
            for behavior in self._behaviors
            if (tagged_record := behavior.get_memory_extension_data()) is not None
            for (key, value) in tagged_record.items()
        }
        return construct_tagged_combined_data_string(
            {
                **tagged_data,
                "memory": get_memory_string(records, with_full_memory_record),
            }
        )

    def get_tagged_full_memory(
        self,
        *,
        with_full_memory_record: bool = False,
        query_filter: MemoryQueryFilter | None = None,
    ) -> str:
        records = self.memory.full_retrieval(query_filter)
        return self._get_tagged_memory_string(records, with_full_memory_record)

    async def get_tagged_memory_by_query(
        self,
        query: str,
        *,
        query_filter: MemoryQueryFilter | None = None,
    ) -> str:
        records = await self.memory.query_retrieval(query, query_filter)
        return self._get_tagged_memory_string(records)

    async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
        for behavior in self._behaviors:
            await behavior.pre_conversation_hook(other_agent)

    async def post_conversation_hook(
        self,
        other_agent: LLMAgentBase,
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ) -> None:
        for behavior in self._behaviors:
            await behavior.post_conversation_hook(other_agent, conversation, logger)

    def get_behavior(self, behavior_cls: type[BehaviorType]) -> BehaviorType:
        for behavior in self._behaviors:
            if isinstance(behavior, behavior_cls):
                return behavior
        raise ValueError(
            f"Behavior of type {behavior_cls} not found in memory manager."
        )


# TODO: after wrapping up write the tests and set up pipelines
