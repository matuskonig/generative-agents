import abc
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Union

from numpydantic import NDArray
from pydantic import BaseModel, Field


class RecordSourceTypeBase(BaseModel):
    """Base class for memory record sources with tagged string representation.

    Uses tag property for discriminated union-style string serialization in
    memory output. Each subclass represents a different type of memory source
    (for example system, conversation, agent note) with distinct tags for display.
    """

    type: str

    @property
    @abc.abstractmethod
    def tag(self) -> str:
        """String tag for serialization in memory output."""
        pass


class BuildInSourceType:
    class System(RecordSourceTypeBase):
        type: str = Field(default="system", init=False, frozen=True)

        @property
        def tag(self) -> str:
            return "[SYSTEM]"

    class Conversation(RecordSourceTypeBase):
        other_agent: str
        type: str = Field(default="conversation", init=False, frozen=True)

        @property
        def tag(self) -> str:
            return f"[CONVERSATION: {self.other_agent}]"

    class UnitaryAgentNoteKnowledge(RecordSourceTypeBase):
        other_agent: str
        type: str = Field(default="unitary_note", init=False, frozen=True)

        @property
        def tag(self) -> str:
            return f"[NOTE: {self.other_agent}]"


@dataclass
class MemoryQueryFilter:
    source_types: Iterable[type[RecordSourceTypeBase]] | None = None
    predicate: Callable[["MemoryRecord"], bool] | None = None


class PruneFactsResponse(BaseModel):
    timestamps_to_remove: list[int]


class MemoryRecordResponse(BaseModel):
    text: str
    relevance: float


class MemoryRecord(MemoryRecordResponse):
    timestamp: int
    source: RecordSourceTypeBase


class MemoryRecordWithEmbedding(MemoryRecord):
    embedding: NDArray


class BDIReasoningBase(BaseModel):
    notes: str = Field(
        description="Freetext field for the agent to write notes, reasoning and planning for the response"
    )


class BDIData(BDIReasoningBase):
    desires: list[str] = Field(description="Enumeration of plans")
    intention: str = Field(description="Selected plan")


class BDINoChanges(BDIReasoningBase):
    tag: Literal["no_change"]


class BDIChangeIntention(BDIReasoningBase):
    tag: Literal["change_intention"]
    intention: str


class BDIFullChange(BDIData):
    tag: Literal["full_change"]


class BDIResponse(BaseModel):
    """BDI (Belief-Desire-Intention) model update response.

    Uses Pydantic discriminated union to handle three possible update types:
    - BDINoChanges: Keep existing desires and intention
    - BDIChangeIntention: Keep desires, change only intention
    - BDIFullChange: Update desires, notes, and intention

    The discriminator field is 'tag' which identifies which variant was returned.
    """

    data: Union[BDINoChanges, BDIChangeIntention, BDIFullChange] = Field(
        discriminator="tag"
    )


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]
