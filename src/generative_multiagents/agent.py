from typing import Type
from pydantic import BaseModel
import asyncio
import abc
import logging
from .llm_backend import LLMBackend, ResponseFormatType
from .async_helpers import cached_async_method


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
    def __init__(self):
        self.__agent_memory: list[str] = []
        self.__other_agents_knowledge: dict[str, list[str]] = {}

    def get_agent_memory(self):
        return "\n".join(self.__agent_memory)

    def get_other_agent_knowledge(self, agent: "LLMAgent"):
        memory = self.__other_agents_knowledge.get(agent.data.full_name)
        if memory:
            return "\n".join(memory)
        return ""

    def add_to_memory(self, facts: list[str]):
        self.__agent_memory.extend(facts)

    def add_to_other_agent_knowledge(self, other: "LLMAgent", text: list[str]):
        if other.data.full_name not in self.__other_agents_knowledge:
            self.__other_agents_knowledge[other.data.full_name] = []

        self.__other_agents_knowledge[other.data.full_name].extend(text)

    def dump_agents_knowledge(self):
        return "\n".join(
            [
                f"[{name}]: {fact}"
                for (name, knowledge) in self.__other_agents_knowledge.items()
                for fact in knowledge
            ]
        )


class Utterance(BaseModel):
    message: str
    is_ending_conversation: bool


Conversation = list[tuple["LLMAgent", Utterance]]


def conversation_to_text(conversation: Conversation):
    return "\n".join(
        [
            f"[{agent.data.full_name}]: {utterance.message}"
            for agent, utterance in conversation
        ]
    )


class Fact(BaseModel):
    text: str


class FactResponse(BaseModel):
    facts: list[Fact]


class LLMAgent:
    def __init__(self, data: AgentModelBase, context: LLMBackend):
        self.data = data
        self.context = context
        self.memory = SimpleMemory()

    @cached_async_method(max_cache_size=1)
    async def get_agent_introduction_message(self):
        prompt = f"Generate a short introduction for {self.data.agent_characteristics}. Start with 'I am ...'. You can inject any greeting if it matches character persona."
        return await self.context.get_text_response(prompt)

    async def start_conversation(self, second_agent: "LLMAgent"):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )
        knowledge_prompt = (
            f"I have following information about {second_agent.data.full_name} from out past conversations: \n<knowledge>\n{knowledge}\n</knowledge>"
            if (knowledge := self.memory.get_other_agent_knowledge(second_agent))
            else None
        )
        prompt_template = [
            f"I am {self.data.full_name}. I have this greeting message:",
            await self.get_agent_introduction_message(),
            memory_prompt,
            knowledge_prompt,
            f"Imagine, I want to start conversation with {second_agent.data.full_name}. What would be the best way to start?",
            (
                "I have met this person before."
                if knowledge_prompt
                else "I have not met this person before."
            ),
            "Do not copy anything mentioned here into the reply.",
        ]
        response = await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )
        return response

    async def generate_next_turn(
        self, second_agent: "LLMAgent", conversation: Conversation
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )
        knowledge_prompt = (
            f"I have following information about {second_agent.data.full_name} from out past conversations:\n<knowledge>\n{knowledge}\n</knowledge>"
            if (knowledge := self.memory.get_other_agent_knowledge(second_agent))
            else None
        )
        prompt_template = [
            f"I am {self.data.full_name}. I have this greeting message:",
            await self.get_agent_introduction_message(),
            memory_prompt,
            knowledge_prompt,
            "We are currently engaged in a conversation. This is the content of the conversation so far:",
            "<conversation>",
            conversation_to_text(conversation),
            f"[{self.data.full_name}]: [FILL IN HERE]",
            "</conversation>",
            "What should I say next? Focus on the conversation content and the person I am talking to. Make sure to keep the conversation going. You can also end the conversation.",
            "Please focus on the latest message from the other person and respond to it.",
            (
                "I have met this person before."
                if knowledge_prompt
                else "I have not met this person before."
            ),
            f"Respond in JSON following this schema: {Utterance.model_json_schema()}",
            "Do not copy anything mentioned here into the reply.",
        ]
        response = await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]), Utterance
        )
        return response

    async def ask_agent(self, question: str):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            f"I am {self.data.full_name}. I have this greeting message:",
            await self.get_agent_introduction_message(),
            memory_prompt,
            "I have also the following notes about every person I had conversation with:",
            self.memory.dump_agents_knowledge(),
            "Answer a following question based on the provided information:",
            question,
        ]

        return await self.context.get_text_response(
            "\n".join([prompt for prompt in prompt_template if prompt])
        )

    async def ask_agent_structured(
        self, question: str, response_format: Type[ResponseFormatType]
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            f"I am {self.data.full_name}. I have this greeting message:",
            await self.get_agent_introduction_message(),
            memory_prompt,
            "Answer a question based on the provided information:",
            question,
            f"Respond in JSON following this schema: {response_format.model_json_schema()}",
        ]
        return await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]),
            response_format=response_format,
        )

    async def __summarize_conversation(self, conversation: Conversation):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )

        prompt_template = [
            f"I am {self.data.full_name}. I have this greeting message:",
            await self.get_agent_introduction_message(),
            memory_prompt,
            f"You have just finished a conversation. Summarize the facts learned from this conversation.",
            "Select the relevant facts only, select up to 3 facts. Select the facts in biggest detail possible, try to capture everything.",
            "You will have access to those information in the following conversations, so select carefully only the information you can build your future conversations on.",
            "Extract data not present in the memory yet",
            f"<conversation>\n{conversation_to_text(conversation)}\n</conversation>",
            f"Respond in JSON following this schema: {FactResponse.model_json_schema()}",
        ]

        result = await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]), FactResponse
        )
        return [fact.text for fact in result.facts]

    async def __extract_agent_knowledge(
        self, other: "LLMAgent", conversation: Conversation
    ):
        memory_prompt = (
            f"I have remembered this things from my past conversations:\n<memory>\n{memory}\n</memory>"
            if (memory := self.memory.get_agent_memory())
            else None
        )
        knowledge_prompt = (
            f"I have following information about {other.data.full_name} from out past conversations: \n<knowledge>\n{knowledge}\n</knowledge>"
            if (knowledge := self.memory.get_other_agent_knowledge(other))
            else None
        )

        prompt_template = [
            await self.get_agent_introduction_message(),
            memory_prompt,
            knowledge_prompt,
            f"You have just finished a conversation. Summarize the facts learned about {other.data.full_name} from this conversation.",
            "Select the relevant facts only, select up to 3 facts. Select the facts in biggest detail possible, try to capture everything.",
            "You will have access to those information in the following conversations, so select carefully only the information you can build your future conversations on.",
            "Extract data not present in the memory yet. ",
            f"<conversation>\n{conversation_to_text(conversation)}\n</conversation>",
            f"Respond in JSON following this schema: {FactResponse.model_json_schema()}",
        ]

        result = await self.context.get_structued_response(
            "\n".join([prompt for prompt in prompt_template if prompt]), FactResponse
        )
        return [fact.text for fact in result.facts]

    async def adjust_memory_after_conversation(
        self,
        other: "LLMAgent",
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ):
        (memory_update, agent_knowledge_update) = await asyncio.gather(
            self.__summarize_conversation(conversation),
            self.__extract_agent_knowledge(other, conversation),
        )
        # TODO> compact memory after some time

        if logger:
            logger.debug(
                f"Memory and knowledge update on {self.data.full_name} about {other.data.full_name}",
                extra={
                    "memory_update": "\n".join(memory_update),
                    "knowledge_update": "\n".join(agent_knowledge_update),
                },
            )
        self.memory.add_to_memory(memory_update)
        self.memory.add_to_other_agent_knowledge(other, agent_knowledge_update)

# TODO: ask questions with followups (continuous questions)