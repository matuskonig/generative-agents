from typing import Union, Literal
from pydantic import BaseModel, Field
import abc
from numpydantic import NDArray


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


class MemoryRecordResponse(BaseModel):
    text: str
    relevance: float


class MemoryRecord(MemoryRecordResponse):
    timestamp: int


class MemoryRecordWithEmbedding(MemoryRecord):
    embedding: NDArray


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


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]


class PruneFactsResponse(BaseModel):
    timestamps_to_remove: list[int]
