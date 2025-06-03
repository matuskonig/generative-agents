from typing import Type, Sequence, Iterable, Callable, Union, Literal
from pydantic import BaseModel, Field
import abc
import logging
from .llm_backend import LLMBackend, ResponseFormatType, create_completion_params
from .utils import OverridableContextVar
import numpy as np
from numpydantic import NDArray
import random


class DefaultConfig:
    __SYSTEM_PROMPT = """You are an agent in a society simulation. Embody the provided persona in all your responses, paying close attention to its distinct communication style."""

    def get_factual_llm_params(self):
        return create_completion_params(temperature=0.3)

    def get_neutral_default_llm_params(self):
        return create_completion_params()

    def get_creative_llm_params(self):
        return create_completion_params(temperature=1.3, frequency_penalty=0.8)

    def get_system_prompt(self):
        return self.__SYSTEM_PROMPT

    def get_introduction_prompt(self, agent_data: "AgentModelBase"):
        return f"Your name is {agent_data.full_name}. Your characteristics are: {agent_data.agent_characteristics}. Craft a brief introduction that establishes your persona, including a unique communication style based on these characteristics."

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
            f"You are about to start a conversation with {second_agent_full_name}. How would you initiate the conversation, keeping your persona and communication style in mind?",
            "Use the communication style of the given persona.",  # Kept for emphasis, can be removed if redundant
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
            f"I am currently engaged in a conversation with {second_agent_full_name}. This is the content of the conversation so far:",
            "<conversation>",
            self.conversation_to_text(conversation),
            f"[{agent_full_name}]: [MASK]",
            "</conversation>",
            f"What is your next response? Consider the ongoing conversation, your persona, and {second_agent_full_name}.",
            "If the conversation becomes repetitive, the topic is exhausted, or you (as your persona) would realistically get bored, gracefully switch the topic.",
            "The conversation might be limited to fixed number of utterances.",
            "Maintain your persona's communication style. Keep the conversation natural and engaging. Address any questions asked.",
            "If the conversation is not progressing, or if it's a natural point to conclude, end the conversation by saying goodbye and setting the appropriate response property.",
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
            "Based on your persona and available information (greeting message and memory), answer the following question:",
            question,
            "Answer concisely and accurately based *only* on the information provided in your greeting and memory. Do not add information not present in your context or invent details.",
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
            f"You have just finished a conversation with {other_agent_full_name}. Summarize the key information you learned from this interaction.",
            "Identify relevant and new facts from the conversation. Capture these facts with as much detail as possible, focusing on information not already in your memory.",
            "This summary will be added to your memory for future conversations. Prioritize information that will be useful for future interactions.",
            "Focus on extracting new topics, specific information, and news shared during the conversation. Avoid including general world knowledge.",
            "Treat this summary as a set of important notes for your future self. Include key topics, insights, and any information that might be valuable later.",
            "Be concise, but ensure all crucial new information is captured. You can compress information where appropriate.",
            "Assign a relevance score (0.0 to 1.0) to each fact. Higher scores indicate greater importance for future recall. Use lower scores for less critical information.",
            conversation_string,
            (
                f"Respond in JSON following this format: {response_format}"
                if response_format
                else None
            ),
        ]
        return "\n".join([prompt for prompt in prompt_template if prompt])

    def get_bdi_init_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_string) if memory_string else None
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            "Firstly, you have select desires from your current beliefs, goals you can consider to achieve in the future conversations. You can select multiple desires.",
            f"Secondly, you have to select intention, the goal you are actively pursuing in the future conversations. The intention should be one of the desires.",
            "You will have access to the intention in the future conversations, however your desires are accessible only now.",
            "Desires and intention should be based on your persona.",
            "You can change your intention only if you think the current intention is considered done, no longer relevant or achievable and only after finishing the conversation.",
            (
                f"Respond in JSON following this format: {response_format}"
                if response_format
                else None
            ),
        ]
        return "\n".join([prompt for prompt in prompt_template if prompt])

    def get_bdi_update_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_string) if memory_string else None
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            memory_prompt,
            f"You have just finished a conversation with {other_agent_full_name}.",
            "You have selected beliefs, desires and intentions previously.",
            "You have a chance to reconsider your desires and intentions from your beliefs based on the conversation.",
            "You can leave the desires and intentions unchanged, or you can pick intention from the desires. You can also select a new set of desires, together with the intention.",
            "Desires are the goals you can consider to achieve in the future conversations. You can select multiple desires.",
            "Intention is the goal you are actively pursuing in the future conversations.",
            "You will have access to the intention in the future conversations, however your desires are accessible only now.",
            "You can change your intention only if you think the current intention is considered done, no longer relevant or achievable and only after finishing the conversation.",
            "Intention should be one of the desires. Desires and intention should be based on your persona.",
            conversation_string,
            (
                f"Respond in JSON following this format: {response_format}"
                if response_format
                else None
            ),
        ]
        return "\n".join([prompt for prompt in prompt_template if prompt])

    def get_memory_prune_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_string) if memory_string else None
        prompt_template = [
            f"I am {agent_full_name}. I have this greeting message:",
            "<greeting>",
            agent_introduction,
            "</greeting>",
            "This is a selected content of your memory.",
            memory_prompt,
            "Respective facts are selected randomly from the whole memory, where the probability of selection is rising for older memories.",
            "You can select memory facts, which you want to remove from your memory. Those memories will no longer be available in any further conversations.",
            "You have your own judgement in regards of which memories to choose, but try to remove unimportant facts and duplicities.",
            "Selection is indicated by passing timestamp of the memory to remove.",
            (
                f"Respond in JSON following this format: {response_format}"
                if response_format
                else None
            ),
        ]
        return "\n".join([prompt for prompt in prompt_template if prompt])


default_config = OverridableContextVar("default_config", DefaultConfig())


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
    async def store_facts(self, facts: Sequence[MemoryRecordResponse]):
        """Append new facts to the memory."""

    @abc.abstractmethod
    def remove_facts(self, timestamps: list[int]):
        pass


class SimpleMemory(MemoryBase):
    def __init__(self):
        self.__timestamp = 0
        self.__memory: list[MemoryRecord] = []

    def current_timestamp(self) -> int:
        return self.__timestamp

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

    def current_timestamp(self) -> int:
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

    @abc.abstractmethod
    async def pre_conversation_hook(self, other_agent: "LLMAgent"):
        pass


class SimpleMemoryManager(MemoryManagerBase):
    def __init__(self, memory: MemoryBase, agent: "LLMAgent"):
        self.memory = memory
        self._agent = agent

    def _get_memory_tag(self, memory_string: str):
        return f"<memory>{memory_string}</memory>"

    def get_tagged_full_memory(self, with_full_memory_record=False) -> str:
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
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
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
        result = await self._agent.context.get_structued_response(
            prompt,
            response_format=FactResponse,
            params=default_config().get_factual_llm_params(),
        )
        await self.memory.store_facts(result.facts)

    async def pre_conversation_hook(self, other_agent: "LLMAgent"):
        pass

    async def post_conversation_hook(
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
        await self._add_new_memory(other_agent, conversation)


class BDIData(BaseModel):
    desires: list[str] = Field(description="Enumeration of plans")
    intention: str = Field(description="Selected plan")


class BDINoChanges(BaseModel):
    tag: Literal["no_change"]


class BDIChangeIntention(BaseModel):
    tag: Literal["change_intention"]
    intention: str


class BDIFullChange(BDIData):
    tag: Literal["full_change"]


class BDIResponse(BaseModel):
    data: Union[BDINoChanges, BDIChangeIntention, BDIFullChange] = Field(
        discriminator="tag"
    )


def get_fact_removal_probability_factory(max_prob_coef: float):
    def inner(current_timestamp: int, fact: MemoryRecord):
        linear_prob = 1 - (fact.timestamp / current_timestamp)
        return max_prob_coef * linear_prob

    return inner


class PruneFactsResponse(BaseModel):
    timestamps_to_remove: list[int]


class BDIMemoryManager(MemoryManagerBase):
    """Simple memory manager elevating Belief-Desire-Intention (BDI) architecture."""

    def __init__(
        self,
        memory: MemoryBase,
        agent: "LLMAgent",
        memory_removal_probability: Callable[[int, MemoryRecord], float] | None = None,
    ):
        self.memory = memory
        self._agent = agent
        self.__bdi_data: BDIData | None = None
        self.memory_removal_probability = memory_removal_probability

    def _get_memory_tag(
        self, memory_string: str, with_desires=False, with_intention: bool = True
    ):
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

    def get_tagged_full_memory(self, with_full_memory_record=False) -> str:
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

    async def _initialize_bdi(self):
        prompt = default_config().get_bdi_init_prompt(
            self._agent.data.full_name,
            await self._agent.get_agent_introduction_message(),
            self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(BDIData.model_json_schema()),
        )
        result = await self._agent.context.get_structued_response(
            prompt, BDIData, params=default_config().get_neutral_default_llm_params()
        )
        self.__bdi_data = result

    async def _update_bdi(self, second_agent: "LLMAgent", conversation: "Conversation"):
        prompt = default_config().get_bdi_update_prompt(
            self._agent.data.full_name,
            await self._agent.get_agent_introduction_message(),
            second_agent.data.full_name,
            default_config().conversation_to_tagged_text(conversation),
            self.get_tagged_full_memory(with_full_memory_record=True),
            response_format=str(BDIResponse.model_json_schema()),
        )
        result = await self._agent.context.get_structued_response(
            prompt,
            response_format=BDIResponse,
            params=default_config().get_neutral_default_llm_params(),
        )
        if isinstance(result.data, BDINoChanges):
            return
        elif isinstance(result.data, BDIChangeIntention) and self.__bdi_data:
            self.__bdi_data.intention = result.data.intention
        elif isinstance(result.data, BDIFullChange):
            self.__bdi_data = result.data

    async def _prune_old_memory(self):
        if not self.memory_removal_probability:
            return

        facts_to_prune = [
            fact
            for fact in self.memory.full_retrieval()
            if random.random()
            <= self.memory_removal_probability(self.memory.current_timestamp(), fact)
        ]
        timestamps = {fact.timestamp for fact in facts_to_prune}
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
        result = await self._agent.context.get_structued_response(
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
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
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
        result = await self._agent.context.get_structued_response(
            prompt,
            response_format=FactResponse,
            params=default_config().get_factual_llm_params(),
        )
        await self.memory.store_facts(result.facts)

    async def pre_conversation_hook(self, other_agent: "LLMAgent"):
        await self._initialize_bdi()

    async def post_conversation_hook(
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
        await self._add_new_memory(other_agent, conversation)
        await self._prune_old_memory()
        await self._update_bdi(other_agent, conversation)


class Utterance(BaseModel):
    actions: list[str] = Field(
        description="Exhaustive list of possible actions of the agent in the conversation. For example, you can change topic, continue or end the conversation."
    )
    selected_action: str
    message: str = Field(
        description="The utterance of the agent in the conversation based on the single action from the list."
    )
    is_conversation_finished: bool = Field(
        description="Mark this utterance as the last one in the conversation.",
        default=False,
    )


Conversation = list[tuple["LLMAgent", Utterance]]


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]


class LLMAgent:
    def __init__(
        self,
        data: AgentModelBase,
        context: LLMBackend,
        create_memory_manager: Callable[["LLMAgent"], MemoryManagerBase],
    ):
        self.data = data
        self.context = context
        self.memory_manager = create_memory_manager(self)

        self.__intro_message: str | None = None

    async def get_agent_introduction_message(self):
        if self.__intro_message:
            return self.__intro_message

        prompt = default_config().get_introduction_prompt(self.data)
        response = (
            await self.context.get_text_response(
                prompt, params=default_config().get_creative_llm_params()
            )
            or ""
        )
        self.__intro_message = response
        return response

    async def start_conversation(self, second_agent: "LLMAgent"):
        introduction_message = await self.get_agent_introduction_message()
        prompt = default_config().start_conversation_prompt(
            self.memory_manager.get_tagged_full_memory(),
            self.data.full_name,
            introduction_message,
            second_agent.data.full_name,
        )
        await self.memory_manager.pre_conversation_hook(second_agent)
        response = await self.context.get_text_response(
            prompt, params=default_config().get_creative_llm_params()
        )
        return response

    async def generate_next_turn(
        self, second_agent: "LLMAgent", conversation: Conversation
    ) -> Utterance:
        memory_tag = await self.memory_manager.get_tagged_memory_by_query(
            default_config().conversation_to_tagged_text(conversation)
        )
        prompt = default_config().generate_next_turn_prompt(
            memory_tag,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            second_agent.data.full_name,
            conversation,
            response_format=str(Utterance.model_json_schema()),
        )
        return await self.context.get_structued_response(
            prompt, Utterance, params=default_config().get_creative_llm_params()
        )

    async def ask_agent(self, question: str, use_full_memory: bool = True):
        memory = (
            self.memory_manager.get_tagged_full_memory()
            if use_full_memory
            else await self.memory_manager.get_tagged_memory_by_query(question)
        )
        prompt = default_config().ask_agent_prompt(
            memory,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            question,
        )
        return await self.context.get_text_response(
            prompt, params=default_config().get_factual_llm_params()
        )

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
        prompt = default_config().ask_agent_prompt(
            memory,
            self.data.full_name,
            await self.get_agent_introduction_message(),
            question,
            response_format=str(response_format.model_json_schema()),
        )
        return await self.context.get_structued_response(
            prompt,
            response_format=response_format,
            params=default_config().get_factual_llm_params(),
        )

    async def adjust_memory_after_conversation(
        self,
        other: "LLMAgent",
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ):
        await self.memory_manager.post_conversation_hook(other, conversation)


# TODO: logger for new memory model
