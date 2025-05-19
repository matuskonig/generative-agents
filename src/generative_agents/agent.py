from typing import Type, Sequence, Iterable, Callable
from pydantic import BaseModel, Field
import abc
import logging
from .llm_backend import LLMBackend, ResponseFormatType
from .utils import OverridableContextVar
import numpy as np
from numpydantic import NDArray


class DefaultPromptBuilder:
    __SYSTEM_PROMPT = """You are an agent in a society simulation. You will be given a persona you are supposed to act as. Keep the persona in mind when responding to the user. Keep the persona communication style as an ultimate goal."""

    def get_system_prompt(self):
        return self.__SYSTEM_PROMPT

    def get_introduction_prompt(self, agent_data: "AgentModelBase"):
        return f"Introduce yourself as {agent_data.full_name}. Based on their agent characteristics ({agent_data.agent_characteristics}), write a brief introduction that establishes their persona. Invent the persona communication style."

    def conversation_to_text(self, conversation: "Conversation"):
        return "\n".join(
            [
                f"[{agent.data.full_name}]: {utterance.message}"
                for agent, utterance in conversation
            ]
        )

    def conversation_to_tagged_text(self, conversation: "Conversation"):
        return (
            "<conversation>\n"
            + self.conversation_to_text(conversation)
            + "\n</conversation>"
        )

    def memory_prompt(self, memory_content: str):
        return f"I have remembered this things from my past conversations: {memory_content}"

    def start_conversation_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
    ) -> str:
        memory_prompt = self.memory_prompt(memory_content)
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            f"Imagine, I want to start conversation with {second_agent_full_name}. What would be the best way to start?",
            "Use the communication style of the given persona.",
        ]

        return "\n".join([prompt for prompt in prompt_template if prompt])

    def generate_next_turn_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
        conversation: "Conversation",
        response_format: str | None = None,
    ) -> str:
        memory_prompt = self.memory_prompt(memory_content)
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            f"We are currently engaged in a conversation with {second_agent_full_name}. This is the content of the conversation so far:",
            "<conversation>",
            self.conversation_to_text(conversation),
            f"[{agent_full_name}]: [MASK]",
            "</conversation>",
            "What should I say next? Focus on the conversation content and the person I am talking to.",
            "If you get bored, the conversation got repetitive or the topic has been exhausted, switch the topic.",
            "The conversation might be limited to fixed number of utterances.",
            "Use the communication style of the given persona, stick to the conversation style as well. Dont be too formal, keep the conversation natural.",
            "You can end the conversation at any time, just say your goodbyes and set the respective property in the response. Prefer this option if you feel the conversation is not going anywhere.",
            (
                f"Respond in JSON following this schema: {response_format}"
                if response_format
                else None
            ),
        ]

        return "\n".join([prompt for prompt in prompt_template if prompt])

    def ask_agent_prompt(
        self,
        memory_string: str,
        agent_full_name: str,
        agent_introduction: str,
        question: str,
        response_format: str | None = None,
    ) -> str:
        memory_prompt = self.memory_prompt(memory_string)

        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            "Answer a following question based on the provided information:",
            question,
            "Stick to the question, respond according to your memory. Do not append additional information, do not hallucinate.",
            (
                f"Respond in JSON following this schema: {response_format}"
                if response_format
                else None
            ),
        ]

        return "\n".join([prompt for prompt in prompt_template if prompt])

    def get_conversation_summary_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_prompt = self.memory_prompt(memory_string) if memory_string else None
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            f"I have just finished a conversation with {other_agent_full_name}. Summarize the information learned from this conversation.",
            "Select the relevant and new information only. Select the facts in biggest detail possible.",
            "You will have access to those information in the following conversations, so select carefully only the information you can build your future conversations on.",
            "Extract data not present in the memory yet. Focus on the topics, information and news mentioned in the conversation and not on the world knowledge.",
            "Use this memory as a sticky note, remember any important information and thoughts that might be consumed later, you can memorize important topics as well.",
            "Try to keep the number of selected facts restricted, but you are free to compress the information as you wish.",
            "Select the relevance score as a number between 0 and 1, highlighting the most important information. Please mark not important information on the lower scale.",
            conversation_string,
            (
                f"Respond in JSON following this format: {response_format}"
                if response_format
                else None
            ),
        ]
        return "\n".join([prompt for prompt in prompt_template if prompt])


default_builder = OverridableContextVar("prompt_builder", DefaultPromptBuilder())


class AgentModelBase(BaseModel, abc.ABC):
    @property
    @abc.abstractmethod
    def full_name(self) -> str:
        pass

    @property
    def agent_characteristics(self) -> str:
        return self.model_dump_json()


class MemoryRecordResponse(BaseModel):
    text: str
    relevance: float


class MemoryRecord(MemoryRecordResponse):
    timestamp: int


class MemoryRecordWithEmbedding(MemoryRecord):
    embedding: NDArray


class MemoryBase(abc.ABC):
    @abc.abstractmethod
    def full_retrieval(self) -> Sequence[MemoryRecord]:
        """Return all facts in the memory as a list of strings."""
        pass

    @abc.abstractmethod
    async def query_retrieval(self, query: str) -> Sequence[MemoryRecord]:
        """Return a list of facts that match the query."""
        pass

    @abc.abstractmethod
    async def store_facts(self, facts: Sequence[MemoryRecordResponse]):
        """Append new facts to the memory."""

    @abc.abstractmethod
    def remove_facts(self, timestamps: list[int]):
        pass


class SimpleMemory(MemoryBase):
    def __init__(self):
        self.__timestamp = 0
        self.__memory: list[MemoryRecord] = []

    def full_retrieval(self) -> list[MemoryRecord]:
        return self.__memory

    async def query_retrieval(self, query: str) -> list[MemoryRecord]:
        return self.__memory

    def __get_next_timestamp(self):
        self.__timestamp += 1
        return self.__timestamp

    async def store_facts(self, facts: Sequence[MemoryRecordResponse]):
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

    def remove_facts(self, timestamps: list[int]):
        self.__memory = [
            record for record in self.__memory if record.timestamp not in timestamps
        ]


def fixed_count_strategy_factory(count: int):
    def inner(records: Sequence[tuple[float, MemoryRecord]]):
        return min(count, len(records))

    return inner


def mean_std_count_strategy_factory(std_coef: float = 0.5):
    def inner(records: Sequence[tuple[float, MemoryRecord]]):
        if len(records) == 0:
            return 0
        scores = np.array([score for score, _ in records])
        mean = np.mean(scores)
        std_dev = np.std(scores)
        treshold = mean + std_dev * std_coef
        return sum(1 for score in scores if score >= treshold)

    return inner


def top_std_count_strategy_factory(std_coef: float = 1.0):
    def inner(records: Sequence[tuple[float, MemoryRecord]]):
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
        time_weight=1.0,
        time_smoothing=0.7,
        relevance_weight=1.0,
        similairity_weight=1.0,
    ):
        self.__context = context
        self.__count_selector = count_selector
        self.__memory: list[MemoryRecordWithEmbedding] = []
        self.__timestamp = 0

        self.__time_weight = time_weight
        self.__time_smoothing = time_smoothing
        self.__relevance_weight = relevance_weight
        self.__similarity_weight = similairity_weight

    def __get_next_timestamp(self):
        self.__timestamp += 1
        return self.__timestamp

    def full_retrieval(self):
        return self.__memory

    def __get_memory_record_score(
        self, query_emb: np.ndarray, record: MemoryRecordWithEmbedding
    ) -> float:
        time_similarity = (record.timestamp / self.__timestamp) ** self.__time_smoothing
        cosine_similarity = np.dot(record.embedding, query_emb) / (
            np.linalg.norm(record.embedding) * np.linalg.norm(query_emb)
        )
        return (
            self.__time_weight * time_similarity
            + self.__relevance_weight * record.relevance
            + self.__similarity_weight * cosine_similarity
        )

    async def query_retrieval(self, query: str):
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

    async def store_facts(self, facts: Iterable[MemoryRecordResponse]):
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

    def remove_facts(self, timestamps: list[int]):
        self.__memory = [
            record for record in self.__memory if record.timestamp not in timestamps
        ]


class MemoryManagerBase(abc.ABC):
    @abc.abstractmethod
    def get_tagged_full_memory(self, with_full_memory_record=False) -> str:
        pass

    @abc.abstractmethod
    async def get_tagged_memory_by_query(self, query: str) -> str:
        pass

    @abc.abstractmethod
    async def post_conversation_hook(
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
        pass


class SimpleMemoryManager(MemoryManagerBase):
    def __init__(self, memory: MemoryBase, agent: "LLMAgent"):
        self.memory = memory
        self._agent = agent

    def get_tagged_full_memory(self, with_full_memory_record=False) -> str:
        return (
            "<memory>"
            + "\n".join(
                [
                    (
                        record.model_dump_json(include=set(MemoryRecord.model_fields))
                        if with_full_memory_record
                        else record.text
                    )
                    for record in self.memory.full_retrieval()
                ]
            )
            + "</memory>"
        )

    async def get_tagged_memory_by_query(self, query: str) -> str:
        return (
            "<memory>"
            + "\n".join(
                [record.text for record in await self.memory.query_retrieval(query)]
            )
            + "</memory>"
        )

    async def post_conversation_hook(
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
        prompt = default_builder().get_conversation_summary_prompt(
            agent_full_name=self._agent.data.full_name,
            agent_introduction=await self._agent.get_agent_introduction_message(),
            other_agent_full_name=other_agent.data.full_name,
            conversation_string=default_builder().conversation_to_tagged_text(
                conversation
            ),
            memory_string=self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(FactResponse.model_json_schema()),
        )
        result = await self._agent.context.get_structued_response(
            prompt, response_format=FactResponse
        )
        await self.memory.store_facts(result.facts)


class BDIMemoryManager(MemoryManagerBase): ...


# RAG for memory retrieval
# BDI architecture for light conversation planning


class Utterance(BaseModel):
    actions: list[str] = Field(
        description="Exhaustive list of possible actions of the agent in the conversation. For example, you can change topic, continue or end the conversation."
    )
    selected_action: str
    message: str = Field(
        description="The utterance of the agent in the conversation based on the single action from the list."
    )
    is_conversation_finished: bool = Field(
        description="Mark this utterance as the last one in the conversation."
    )


Conversation = list[tuple["LLMAgent", Utterance]]


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]


class LLMAgent:
    def __init__(self, data: AgentModelBase, context: LLMBackend):
        self.data = data
        self.context = context
        # TODO: make this outside configurable
        self.memory_manager = SimpleMemoryManager(
            EmbeddingMemory(context, count_selector=mean_std_count_strategy_factory()),
            self,
        )
        self.__intro_message: str | None = None

    async def get_agent_introduction_message(self):
        if self.__intro_message:
            return self.__intro_message

        prompt = default_builder().get_introduction_prompt(self.data)
        response = await self.context.get_text_response(prompt) or ""
        self.__intro_message = response
        return response

    async def start_conversation(self, second_agent: "LLMAgent"):
        prompt = default_builder().start_conversation_prompt(
            self.memory_manager.get_tagged_full_memory(),
            self.data.full_name,
            await self.get_agent_introduction_message(),
            second_agent.data.full_name,
        )

        response = await self.context.get_text_response(prompt)
        return response

    async def generate_next_turn(
        self, second_agent: "LLMAgent", conversation: Conversation
    ) -> Utterance:
        memory_tag = await self.memory_manager.get_tagged_memory_by_query(
            default_builder().conversation_to_tagged_text(conversation)
        )
        prompt = default_builder().generate_next_turn_prompt(
            memory_tag,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            second_agent.data.full_name,
            conversation,
            response_format=str(Utterance.model_json_schema()),
        )
        return await self.context.get_structued_response(prompt, Utterance)

    async def ask_agent(self, question: str, use_full_memory: bool = True):
        memory = (
            self.memory_manager.get_tagged_full_memory()
            if use_full_memory
            else await self.memory_manager.get_tagged_memory_by_query(question)
        )
        prompt = default_builder().ask_agent_prompt(
            memory,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            question,
        )
        return await self.context.get_text_response(prompt)

    async def ask_agent_structured(
        self,
        question: str,
        response_format: Type[ResponseFormatType],
        use_full_memory: bool = True,
    ):
        memory = (
            self.memory_manager.get_tagged_full_memory()
            if use_full_memory
            else await self.memory_manager.get_tagged_memory_by_query(question)
        )
        prompt = default_builder().ask_agent_prompt(
            memory,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            question,
            response_format=str(response_format.model_json_schema()),
        )
        return await self.context.get_structued_response(
            prompt, response_format=response_format
        )

    async def adjust_memory_after_conversation(
        self,
        other: "LLMAgent",
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ):
        await self.memory_manager.post_conversation_hook(other, conversation)


# TODO: ask questions with followups (continuous questions)
# TODO: logger for new memory model
