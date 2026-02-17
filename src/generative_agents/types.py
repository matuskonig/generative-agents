import abc

from pydantic import BaseModel, Field


class AgentModelBase(BaseModel, abc.ABC):
    """Agent data class holding information/background about the agent"""

    @property
    @abc.abstractmethod
    def full_name(self) -> str:
        pass

    @property
    def agent_characteristics(self) -> str:
        return self.model_dump_json()


class LLMAgentBase:
    @property
    @abc.abstractmethod
    def data(self) -> AgentModelBase:
        pass

    @abc.abstractmethod
    async def get_agent_introduction_message(self) -> str:
        pass


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


Conversation = list[tuple[LLMAgentBase, Utterance]]
