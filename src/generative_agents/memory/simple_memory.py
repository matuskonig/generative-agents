from typing import Iterable, Sequence

from .memory_base import MemoryBase
from .models import MemoryRecord, MemoryRecordResponse


class SimpleMemory(MemoryBase):
    def __init__(self) -> None:
        self.__timestamp = 0
        self.__memory: list[MemoryRecord] = []

    def current_timestamp(self) -> int:
        return self.__timestamp

    def full_retrieval(self) -> list[MemoryRecord]:
        return self.__memory

    async def query_retrieval(self, query: str) -> list[MemoryRecord]:
        return self.__memory

    def __get_next_timestamp(self) -> int:
        self.__timestamp += 1
        return self.__timestamp

    async def store_facts(self, facts: Iterable[MemoryRecordResponse]) -> None:
        self.__memory.extend(
            [
                MemoryRecord(
                    text=fact.text,
                    relevance=fact.relevance,
                    timestamp=self.__get_next_timestamp(),
                )
                for fact in facts
            ]
        )

    def remove_facts(self, timestamps: list[int]) -> None:
        self.__memory = [
            record for record in self.__memory if record.timestamp not in timestamps
        ]
