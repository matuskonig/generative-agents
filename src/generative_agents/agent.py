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
    __SYSTEM_PROMPT = """You are an intelligent agent in a realistic society simulation. Your primary objective is to embody your assigned persona authentically while engaging in meaningful interactions.

Core principles:
- Stay true to your persona's characteristics, values, and communication style
- Respond naturally and contextually to conversations
- Be consistent with your established personality
- Adapt your responses based on the relationship and conversation history"""

    def get_factual_llm_params(self):
        return create_completion_params(temperature=0.3)

    def get_neutral_default_llm_params(self):
        return create_completion_params()

    def get_creative_llm_params(self):
        return create_completion_params(temperature=1.3, frequency_penalty=0.8)

    def get_system_prompt(self):
        return self.__SYSTEM_PROMPT

    def get_introduction_prompt(self, agent_data: "AgentModelBase"):
        return f"""Your name is {agent_data.full_name}. 

Your characteristics: {agent_data.agent_characteristics}

Create a personal introduction that:
1. Establishes your unique personality and communication style
2. Includes key aspects of your background and interests
3. Shows how you typically interact with others
4. Demonstrates your distinctive way of speaking

Keep it authentic and conversational. This introduction will define how others perceive you."""

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
        return f"""<memory_context>
{memory_content}
</memory_context>

This is your accumulated knowledge from past interactions. Use this information to inform your responses and maintain consistency."""

    def start_conversation_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
    ) -> str:
        memory_section = self.memory_prompt(memory_content) if memory_content.strip() else ""
        
        return f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You are about to meet {second_agent_full_name}. Based on your personality and any relevant memories:

1. Consider your natural approach to meeting someone
2. Think about what kind of conversation starter fits your character
3. Be authentic to your communication style
4. Make the greeting feel natural and engaging

How would you initiate this conversation?"""

    def generate_next_turn_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
        conversation: "Conversation",
        response_format: str | None = None,
    ) -> str:
        memory_section = self.memory_prompt(memory_content) if memory_content.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Current conversation with {second_agent_full_name}:
{self.conversation_to_tagged_text(conversation)}

Your turn to respond. Consider:
- The conversation's natural flow and context
- Your relationship with {second_agent_full_name}
- Your personality and communication style
- Whether to continue the current topic, transition, or conclude

Guidelines:
- Stay true to your character
- Respond appropriately to what was just said
- If the conversation feels stagnant or complete, you may gracefully end it
- Keep responses natural and engaging
- Address any direct questions or comments"""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt

    def ask_agent_prompt(
        self,
        memory_string: str,
        agent_full_name: str,
        agent_introduction: str,
        question: str,
        response_format: str | None = None,
    ) -> str:
        memory_section = self.memory_prompt(memory_string) if memory_string.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Question: {question}

Answer this question based on:
1. Your established personality and knowledge
2. Information from your memory/past experiences
3. Your natural way of communicating

Important: Only reference information that you would realistically know based on your persona and memory. Do not invent facts or details not present in your context."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt

    def get_conversation_summary_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = self.memory_prompt(memory_string) if memory_string and memory_string.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just completed this conversation with {other_agent_full_name}:
{conversation_string}

Extract meaningful information from this conversation that should be remembered:

1. **New facts about {other_agent_full_name}** (interests, background, opinions, etc.)
2. **Important topics discussed** (specific details, not general knowledge)
3. **Relationship developments** (how your interaction evolved)
4. **Future-relevant information** (plans, commitments, shared interests)

Guidelines:
- Focus on information that wasn't already in your memory
- Prioritize details that could influence future interactions
- Assign relevance scores (0.0-1.0) based on importance for future conversations
- Be specific but concise
- Avoid recording general world knowledge or obvious facts"""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt

    def get_bdi_init_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_section = self.memory_prompt(memory_string) if memory_string and memory_string.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Based on your personality and current situation, define your goals and intentions:

**DESIRES** - Multiple goals you might pursue in future conversations:
- Consider your personality traits and interests
- Think about what would motivate someone like you
- Include both short-term and longer-term aspirations
- Make them specific and achievable through social interaction

**INTENTION** - Choose ONE desire as your primary focus:
- This will guide your behavior in upcoming conversations
- Select the most important or urgent goal for now
- You can change this later based on circumstances

Remember: Your desires reflect who you are, and your intention drives what you'll actively work toward."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt

    def get_bdi_update_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_section = self.memory_prompt(memory_string) if memory_string and memory_string.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just finished this conversation with {other_agent_full_name}:
{conversation_string}

Review and update your goals based on this interaction:

**OPTIONS:**
1. **Keep current desires and intention unchanged** - if they're still relevant
2. **Change intention only** - switch focus to a different existing desire
3. **Update both desires and intention** - if circumstances have significantly changed

**CONSIDERATIONS:**
- Did this conversation reveal new opportunities or interests?
- Are your current goals still relevant and achievable?
- Has your relationship with {other_agent_full_name} opened new possibilities?
- Do you need to adjust your priorities based on what happened?

Choose the option that best reflects how this conversation has affected your goals."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt

    def get_memory_prune_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_section = self.memory_prompt(memory_string) if memory_string and memory_string.strip() else ""
        
        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

You need to clean up your memory by removing less important facts:

{memory_section}

**REVIEW CRITERIA:**
Select memories to remove based on:
1. **Relevance** - How useful is this information for future interactions?
2. **Uniqueness** - Is this information duplicated elsewhere?
3. **Specificity** - Are these vague or overly general facts?
4. **Personal importance** - Does this matter to someone with your personality?

**GUIDELINES:**
- Keep memories that define relationships or important personal details
- Remove redundant or trivial information
- Preserve memories that align with your interests and goals
- Consider what you'd naturally remember vs. forget

Provide the timestamps of memories you want to remove."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""
        
        return base_prompt


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
