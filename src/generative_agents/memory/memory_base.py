import abc
from typing import Sequence

from .models import MemoryRecord, MemoryRecordResponse


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
    async def store_facts(self, facts: Sequence[MemoryRecordResponse]) -> None:
        """Append new facts to the memory."""

    @abc.abstractmethod
    def remove_facts(self, timestamps: list[int]) -> None:
        pass
