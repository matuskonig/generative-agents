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


class FactResponse(BaseModel):
    facts: list[MemoryRecordResponse]
