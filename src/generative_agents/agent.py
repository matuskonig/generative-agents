from typing import Type
from pydantic import BaseModel, Field
import abc
import logging
from .llm_backend import LLMBackend, ResponseFormatType
from .utils import OverridableContextVar


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

    async def start_conversation_prompt(
        self, memory_content: str, agent: "LLMAgent", second_agent: "LLMAgent"
    ):
        memory_prompt = self.memory_prompt(memory_content)
        agent.get_agent_introduction_message
        prompt_template = [
            f"I am {agent.data.full_name}. I have this greeting message:",
            await agent.get_agent_introduction_message(),
            memory_prompt,
            f"Imagine, I want to start conversation with {second_agent.data.full_name}. What would be the best way to start?",
            "Use the communication style of the given persona.",
        ]

        return "\n".join([prompt for prompt in prompt_template if prompt])

    async def generate_next_turn_prompt(
        self,
        memory_content: str,
        agent: "LLMAgent",
        second_agent: "LLMAgent",
        conversation: "Conversation",
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_content)
        prompt_template = [
            f"I am {agent.data.full_name}. I have this greeting message:",
            await agent.get_agent_introduction_message(),
            memory_prompt,
            f"We are currently engaged in a conversation with {second_agent.data.full_name}. This is the content of the conversation so far:",
            "<conversation>",
            self.conversation_to_text(conversation),
            f"[{agent.data.full_name}]: [MASK]",
            "</conversation>",
            "What should I say next? Focus on the conversation content and the person I am talking to.",
            "If you get bored, the conversation got repetitive or the topic has been exhausted, switch the topic.",
            "The conversation will be limited to fixed number of utterances.",
            "Use the communication style of the given persona, stick to the conversation style as well. Dont be too formal, keep the conversation natural.",
            "You can end the conversation at any time, just say your goodbyes and set the respective property in the response. Prefer this option if you feel the conversation is not going anywhere.",
            (
                f"Respond in JSON following this schema: {response_format}"
                if response_format
                else None
            ),
        ]

        return "\n".join([prompt for prompt in prompt_template if prompt])

    async def ask_agent_prompt(
        self,
        memory_string: str,
        agent: "LLMAgent",
        question: str,
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_string)

        prompt_template = [
            f"I am {agent.data.full_name}. I have this greeting message:",
            await agent.get_agent_introduction_message(),
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

    async def get_conversation_summary_prompt(
        self,
        agent: "LLMAgent",
        other_agent: "LLMAgent",
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_prompt = self.memory_prompt(memory_string) if memory_string else None
        prompt_template = [
            f"I am {agent.data.full_name}. I have this greeting message:",
            await agent.get_agent_introduction_message(),
            memory_prompt,
            f"I have just finished a conversation with {other_agent.data.full_name}. Summarize the information learned from this conversation.",
            "Select the relevant and new information only. Select the facts in biggest detail possible.",
            "You will have access to those information in the following conversations, so select carefully only the information you can build your future conversations on.",
            "Extract data not present in the memory yet. Focus on the topics, information and news mentioned in the conversation and not on the world knowledge.",
            "Use this memory as a sticky note, remember any important information and thoughts that might be consumed later, you can memorize important topics as well.",
            "Try to keep the number of selected facts restricted, but you are free to compress the information as you wish.",
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


class MemoryBase(abc.ABC):
    @abc.abstractmethod
    async def full_retrieval(self) -> list[str]:
        """Return all facts in the memory as a list of strings."""
        pass

    @abc.abstractmethod
    async def query_retrieval(self, query: str) -> list[str]:
        """Return a list of facts that match the query."""
        pass

    @abc.abstractmethod
    async def store_facts(self, facts: list[str]):
        """Append new facts to the memory."""
        pass

    # TODO fact removal API


class SimpleMemory(MemoryBase):
    def __init__(self):
        self.__memory: list[str] = []

    async def full_retrieval(self) -> list[str]:
        return self.__memory

    async def query_retrieval(self, query: str) -> list[str]:
        return self.__memory

    async def store_facts(self, facts: list[str]):
        self.__memory.extend(facts)


class ExtendedMemory(MemoryBase):
    def __init__(self, context: LLMBackend): ...

    async def full_retrieval(self) -> list[str]: ...
    async def query_retrieval(self, query: str) -> list[str]: ...
    async def store_facts(self, facts: list[str]): ...


class MemoryManagerBase(abc.ABC):
    @abc.abstractmethod
    async def get_tagged_full_memory(self) -> str:
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

    async def get_tagged_full_memory(self) -> str:
        return "<memory>" + "\n".join(await self.memory.full_retrieval()) + "</memory>"

    async def get_tagged_memory_by_query(self, query: str) -> str:
        return (
            "<memory>"
            + "\n".join(await self.memory.query_retrieval(query))
            + "</memory>"
        )

    async def post_conversation_hook(
        self, other_agent: "LLMAgent", conversation: "Conversation"
    ):
        prompt = await default_builder().get_conversation_summary_prompt(
            self._agent,
            other_agent,
            conversation_string=default_builder().conversation_to_tagged_text(
                conversation
            ),
            memory_string=await self.get_tagged_full_memory(),
            response_format=str(FactResponse.model_json_schema()),
        )

        result = await self._agent.context.get_structued_response(
            prompt, response_format=FactResponse
        )
        await self.memory.store_facts([fact.text for fact in result.facts])


class BDIMemoryManager(MemoryManagerBase): ...


# RAG for memory retrieval
# BDI architecture for light conversation planning


class Utterance(BaseModel):
    actions: list[str] = Field(
        description="Exhaustive list of possible actions of the agent in the conversation. For example, you can change topic, continue or end the conversation."
    )
    message: str = Field(
        description="The utterance of the agent in the conversation based on the single action from the list."
    )
    is_conversation_finished: bool = Field(
        description="Mark this utterance as the last one in the conversation."
    )


Conversation = list[tuple["LLMAgent", Utterance]]


class Fact(BaseModel):
    text: str


class FactResponse(BaseModel):
    facts: list[Fact]


# TODO: query-scoped memory access
class LLMAgent:
    def __init__(self, data: AgentModelBase, context: LLMBackend):
        self.data = data
        self.context = context
        self.memory_manager = SimpleMemoryManager(SimpleMemory(), self)
        self.__intro_message: str | None = None

    async def get_agent_introduction_message(self):
        if self.__intro_message:
            return self.__intro_message

        prompt = default_builder().get_introduction_prompt(self.data)
        response = await self.context.get_text_response(prompt) or ""
        self.__intro_message = response
        return response

    async def start_conversation(self, second_agent: "LLMAgent"):
        prompt = await default_builder().start_conversation_prompt(
            await self.memory_manager.get_tagged_full_memory(), self, second_agent
        )

        response = await self.context.get_text_response(prompt)
        return response

    async def generate_next_turn(
        self, second_agent: "LLMAgent", conversation: Conversation
    ) -> Utterance:

        response = await self.context.get_structued_response(
            await default_builder().generate_next_turn_prompt(
                await self.memory_manager.get_tagged_full_memory(),
                self,
                second_agent,
                conversation,
                response_format=str(Utterance.model_json_schema()),
            ),
            Utterance,
        )
        return response

    async def ask_agent(self, question: str):
        prompt = await default_builder().ask_agent_prompt(
            await self.memory_manager.get_tagged_full_memory(), self, question
        )
        return await self.context.get_text_response(prompt)

    async def ask_agent_structured(
        self, question: str, response_format: Type[ResponseFormatType]
    ):

        prompt = await default_builder().ask_agent_prompt(
            await self.memory_manager.get_tagged_full_memory(),
            self,
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
