import abc
from typing import Callable, Iterable, Sequence

from .models import (
    MemoryQueryFilter,
    MemoryRecord,
    MemoryRecordResponse,
    RecordSourceTypeBase,
)


def create_memory_filter_predicate(
    query_filter: MemoryQueryFilter | None,
) -> Callable[[MemoryRecord], bool]:
    """Creates a composite filter predicate from MemoryQueryFilter.

    Combines multiple filter criteria (source types and custom predicate) into
    a single predicate using AND logic. Returns a pass-all predicate if no
    filters are specified.

    Args:
        query_filter: Filter configuration containing source_types and/or predicate

    Returns:
        Callable that returns True if record passes all filters
    """
    if query_filter is None:
        return lambda _: True

    predicates: list[Callable[[MemoryRecord], bool]] = []

    if query_filter.source_types is not None:
        source_types_tuple = tuple(query_filter.source_types)
        predicates.append(lambda record: isinstance(record.source, source_types_tuple))

    if query_filter.predicate is not None:
        predicates.append(query_filter.predicate)

    if not predicates:
        return lambda _: True

    return lambda record: all(predicate(record) for predicate in predicates)


class MemoryBase(abc.ABC):
    @abc.abstractmethod
    def current_timestamp(self) -> int:
        pass

    @abc.abstractmethod
    def full_retrieval(
        self, query_filter: MemoryQueryFilter | None = None
    ) -> Sequence[MemoryRecord]:
        """Return all facts in the memory as a list of strings."""
        pass

    @abc.abstractmethod
    async def query_retrieval(
        self, query: str, query_filter: MemoryQueryFilter | None = None
    ) -> Sequence[MemoryRecord]:
        """Return a list of facts that match the query."""
        pass

    @abc.abstractmethod
    async def store_facts(
        self,
        facts: Iterable[MemoryRecordResponse],
        source: RecordSourceTypeBase,
    ) -> Iterable[int]:
        """Append new facts to the memory.

        Returns:
            List of unique timestamps for the newly stored facts.
        """

    @abc.abstractmethod
    def remove_facts(self, timestamps: Iterable[int]) -> None:
        pass
