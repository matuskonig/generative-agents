from typing import Type
from pydantic import BaseModel
import asyncio
import abc
from .llm_backend import LLMBackend, ResponseFormatType
from .async_helpers import cached_async_getter


class AgentModelBase(BaseModel, abc.ABC):
    @property
    @abc.abstractmethod
    def full_name(self) -> str:
        pass

    @property
    def agent_characteristics(self) -> str:
        return self.model_dump_json()


# TODO: create a RAG-like memory and memory abstraction
class SimpleMemory:
    __agent_memory: list[str] = []
    __other_agents_knowledge: dict[str, list[str]] = {}

    def get_agent_memory(self):
        return "\n".join(self.__agent_memory)

    def get_other_agent_knowledge(self, agent: "LLMAgent"):
        memory = self.__other_agents_knowledge.get(agent.data.full_name)
        if memory:
            return "\n".join(memory)
        return ""

    def add_to_memory(self, text: str):
        self.__agent_memory.append(text)

    def add_to_other_agent_knowledge(self, other: "LLMAgent", text: str):
        if other.data.full_name not in self.__other_agents_knowledge:
            self.__other_agents_knowledge[other.data.full_name] = []

        self.__other_agents_knowledge[other.data.full_name].append(text)


class Utterance(BaseModel):
    message: str
    is_ending_conversation: bool


Conversation = list[tuple["LLMAgent", Utterance]]


def conversation_to_text(conversation: Conversation):
    return "\n".join(
        [
            f"{agent.data.full_name}: {utterance.message}"
            for agent, utterance in conversation
        ]
    )


class LLMAgent:
    def __init__(self, data: AgentModelBase, context: LLMBackend):
        self.data = data
        self.context = context
        self.memory = SimpleMemory()

    @cached_async_getter
    async def introduce_yourself(self):
        prompt = f"Generate a short introduction for {self.data.agent_characteristics}. Start with 'I am ...'. You can inject any greeting if it matches character persona."
        return await self.context.get_text_response(prompt)

    async def start_conversation(self, second_agent: "LLMAgent"):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )
        knowledge_prompt = (
            f"I have following information about {second_agent.data.full_name} from out past conversations: \n{knowledge}"
            if (knowledge := self.memory.get_other_agent_knowledge(second_agent))
            else None
        )
        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            knowledge_prompt or "I have not met this person before.",
            f"Imagine, I want to start conversation with {second_agent.data.full_name}. What would be the best way to start?",
        ]
        return await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )

    async def generate_next_turn(
        self, second_agent: "LLMAgent", conversation: Conversation
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )
        knowledge_prompt = (
            f"I have following information about {second_agent.data.full_name} from out past conversations:\n{knowledge}"
            if (knowledge := self.memory.get_other_agent_knowledge(second_agent))
            else None
        )
        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            knowledge_prompt or "I have not met this person before.",
            "We are currently engaged in a conversation. This is the content of the conversation so far:",
            conversation_to_text(conversation),
            "What should I say next?",
        ]
        return await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]), Utterance
        )

    async def ask_agent(self, question: str):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            "Answer a question based on the provided information:",
            question,
        ]
        return await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )

    async def ask_agent_structured(
        self, question: str, response_format: Type[ResponseFormatType]
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            "Answer a question based on the provided information:",
            question,
        ]
        return await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]),
            response_format=response_format,
        )

    async def __summarize_conversation(self, conversation: Conversation):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            "You have just finished a conversation. Summarize the conversation in a few sentences, with the upper limit of 3 sentences. Be brief, select the information only relevant to you. You will have access to those information in the following conversations, so select carefully.",
            conversation_to_text(conversation),
        ]
        return await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )
        pass

    async def __extract_agent_knowledge(
        self, other: "LLMAgent", conversation: Conversation
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n{memory}"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            await self.introduce_yourself(),
            memory_prompt,
            f"You have just finished a conversation. Summarize the facts learned from {other.data.full_name} relevant to you. Be brief, use at most 3 sentences. You will have access to those information in the following conversations, so select carefully.",
            conversation_to_text(conversation),
        ]
        return await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )

    async def adjust_memory_after_conversation(
        self, other: "LLMAgent", conversation: Conversation
    ):
        (memory_update, agent_knowledge_update) = await asyncio.gather(
            self.__summarize_conversation(conversation),
            self.__extract_agent_knowledge(other, conversation),
        )

        self.memory.add_to_memory(memory_update)
        self.memory.add_to_other_agent_knowledge(other, agent_knowledge_update)
