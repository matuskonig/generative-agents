from typing import Literal, Union

from numpydantic import NDArray
from pydantic import BaseModel, Field


class PruneFactsResponse(BaseModel):
    timestamps_to_remove: list[int]


class MemoryRecordResponse(BaseModel):
    text: str
    relevance: float


class MemoryRecord(MemoryRecordResponse):
    timestamp: int
    # TODO: rozsirit o typ, nejako rozsirit i query metody a dotiahnut to na Affordable memory. budeme potrebovat i nejaku shared memory


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
    data: Union[BDINoChanges, BDIChangeIntention, BDIFullChange] = Field(
        discriminator="tag"
    )


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]
