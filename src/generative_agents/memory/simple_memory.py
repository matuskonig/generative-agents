from typing import Iterable

from .memory_base import MemoryBase, create_memory_filter_predicate
from .models import (
    MemoryQueryFilter,
    MemoryRecord,
    MemoryRecordResponse,
    RecordSourceTypeBase,
)


class SimpleMemory(MemoryBase):
    def __init__(self) -> None:
        self.__timestamp = 0
        self.__memory: list[MemoryRecord] = []

    def current_timestamp(self) -> int:
        return self.__timestamp

    def full_retrieval(
        self, query_filter: MemoryQueryFilter | None = None
    ) -> list[MemoryRecord]:
        predicate = create_memory_filter_predicate(query_filter)
        return [record for record in self.__memory if predicate(record)]

    async def query_retrieval(
        self, query: str, query_filter: MemoryQueryFilter | None = None
    ) -> list[MemoryRecord]:
        predicate = create_memory_filter_predicate(query_filter)
        return [record for record in self.__memory if predicate(record)]

    def __get_next_timestamp(self) -> int:
        self.__timestamp += 1
        return self.__timestamp

    async def store_facts(
        self, facts: Iterable[MemoryRecordResponse], source: RecordSourceTypeBase
    ) -> list[int]:
        final_records = [
            MemoryRecord(
                text=fact.text,
                relevance=fact.relevance,
                timestamp=self.__get_next_timestamp(),
                source=source,
            )
            for fact in facts
        ]
        self.__memory.extend(final_records)
        return [record.timestamp for record in final_records]

    def remove_facts(self, timestamps: Iterable[int]) -> None:
        set_timestamps = set(timestamps)
        self.__memory = [
            record for record in self.__memory if record.timestamp not in set_timestamps
        ]
