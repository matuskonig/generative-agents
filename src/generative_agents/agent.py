from typing import Type, Callable
import logging

from .llm_backend import LLMBackend, ResponseFormatType
from .types import AgentModelBase, Conversation, Utterance, FactResponse, LLMAgentBase
from .memory import MemoryManagerBase
from .config import default_config


class LLMAgent(LLMAgentBase):
    def __init__(
        self,
        data: AgentModelBase,
        context: LLMBackend,
        create_memory_manager: Callable[["LLMAgent"], MemoryManagerBase],
    ) -> None:
        self._data = data
        self.context = context
        self.memory_manager = create_memory_manager(self)

        self.__intro_message: str | None = None

    @property
    def data(self) -> AgentModelBase:
        return self._data

    async def get_agent_introduction_message(self) -> str:
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

    async def start_conversation(self, second_agent: "LLMAgent") -> str:
        introduction_message = await self.get_agent_introduction_message()
        prompt = default_config().start_conversation_prompt(
            self.memory_manager.get_tagged_full_memory(),
            self.data.full_name,
            introduction_message,
            second_agent.data.full_name,
        )
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
        result = await self.context.get_structued_response(
            prompt,
            response_format=Utterance,
            params=default_config().get_creative_llm_params(),
        )
        return result

    async def ask_agent(self, question: str, use_full_memory: bool = True) -> str:
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
    ) -> ResponseFormatType:
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

    async def post_conversation_hook(
        self,
        other: "LLMAgent",
        conversation: Conversation,
        logger: logging.Logger | None = None,
    ) -> None:
        await self.memory_manager.post_conversation_hook(other, conversation, logger)

    async def pre_conversation_hook(self, other: "LLMAgent") -> None:
        await self.memory_manager.pre_conversation_hook(other)


# TODO: in additon to pruning, implement memory compression ?
# TODO: consider adding what they did in Affordable Generative agents - where for every agent they hold some really short summary
# TODO: how to potentially add support for the environment ?
# TODO: how to make it more generic and supportive of expansion outside our control ?
# TODO: switch to normal prompt template
