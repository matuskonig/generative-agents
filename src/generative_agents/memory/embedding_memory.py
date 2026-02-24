from typing import Callable, Iterable, Sequence

import numpy as np

from ..llm_backend import LLMBackend
from .memory_base import MemoryBase, create_memory_filter_predicate
from .models import (
    MemoryQueryFilter,
    MemoryRecord,
    MemoryRecordResponse,
    MemoryRecordWithEmbedding,
    RecordSourceTypeBase,
)


def fixed_count_strategy_factory(
    count: int,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        return min(count, len(records))

    return inner


def mean_std_count_strategy_factory(
    std_coef: float = 0.5,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    """Selects memories with scores above mean + (std_dev * std_coef).

    Uses statistical threshold to select only statistically significant memories.
    Memories with scores significantly above average are selected.

    Args:
        std_coef: Multiplier for standard deviation (0.5 = half std dev above mean)
    """

    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        if len(records) == 0:
            return 0
        scores = np.array([score for score, _ in records])
        mean = np.mean(scores)
        std_dev = np.std(scores)
        treshold = mean + std_dev * std_coef
        return sum(1 for score in scores if score >= treshold)

    return inner


def top_std_count_strategy_factory(
    std_coef: float = 1.0,
) -> Callable[[Sequence[tuple[float, MemoryRecord]]], int]:
    """Selects memories with scores within std_dev of the maximum score.

    Keeps memories that are close to the highest-scoring memory, filtering out
    those significantly below the peak.

    Args:
        std_coef: Distance from max score in terms of std_dev (1.0 = within 1 std_dev)
    """

    def inner(records: Sequence[tuple[float, MemoryRecord]]) -> int:
        if len(records) == 0:
            return 0
        scores = np.array([score for score, _ in records])
        max = np.max(scores)
        std_dev = np.std(scores)
        treshold = max - std_dev * std_coef
        return sum(1 for score in scores if score >= treshold)

    return inner


class EmbeddingMemory(MemoryBase):
    def __init__(
        self,
        context: LLMBackend,
        count_selector: Callable[[Sequence[tuple[float, MemoryRecord]]], int],
        time_weight: float = 1.0,
        time_smoothing: float = 0.7,
        relevance_weight: float = 1.0,
        similairity_weight: float = 1.0,
    ) -> None:
        self.__context = context
        self.__count_selector = count_selector
        self.__memory: list[MemoryRecordWithEmbedding] = []
        self.__timestamp = 0

        self.__time_weight = time_weight
        self.__time_smoothing = time_smoothing
        self.__relevance_weight = relevance_weight
        self.__similarity_weight = similairity_weight

    def __get_next_timestamp(self) -> int:
        self.__timestamp += 1
        return self.__timestamp

    def current_timestamp(self) -> int:
        return self.__timestamp

    def full_retrieval(
        self, query_filter: MemoryQueryFilter | None = None
    ) -> list[MemoryRecordWithEmbedding]:
        predicate = create_memory_filter_predicate(query_filter)
        return [record for record in self.__memory if predicate(record)]

    def __get_memory_record_score(
        self, query_emb: np.ndarray, record: MemoryRecordWithEmbedding
    ) -> float:
        """Computes composite score for memory retrieval ranking.

        Combines three factors:
        - Time similarity: newer memories score higher (exponentially decayed)
        - Relevance: LLM-assigned importance of the memory
        - Cosine similarity: semantic similarity to query

        Args:
            query_emb: Embedding vector of the query
            record: Memory record with embedding to score
        """
        time_similarity: float = (
            record.timestamp / self.__timestamp
        ) ** self.__time_smoothing
        cosine_similarity: float = np.dot(record.embedding, query_emb) / (
            np.linalg.norm(record.embedding) * np.linalg.norm(query_emb)
        )
        return (
            self.__time_weight * time_similarity
            + self.__relevance_weight * record.relevance
            + self.__similarity_weight * cosine_similarity
        )

    async def query_retrieval(
        self, query: str, query_filter: MemoryQueryFilter | None = None
    ) -> list[MemoryRecordWithEmbedding]:
        predicate = create_memory_filter_predicate(query_filter)
        filtered_memory = [record for record in self.__memory if predicate(record)]

        if len(filtered_memory) == 0:
            return []

        query_embedding = await self.__context.embed_text(query)
        scored_records = sorted(
            [
                (self.__get_memory_record_score(query_embedding, record), record)
                for record in filtered_memory
            ],
            reverse=True,
        )

        selected_count = self.__count_selector(scored_records)
        return [record for _, record in scored_records[:selected_count]]

    async def store_facts(
        self,
        facts: Iterable[MemoryRecordResponse],
        source: RecordSourceTypeBase,
    ) -> list[int]:
        embeddings = await self.__context.embed_text([fact.text for fact in facts])
        final_records = [
            MemoryRecordWithEmbedding(
                timestamp=self.__get_next_timestamp(),
                text=fact.text,
                relevance=fact.relevance,
                embedding=embedding,
                source=source,
            )
            for fact, embedding in zip(facts, embeddings)
        ]
        self.__memory.extend(final_records)
        return [record.timestamp for record in final_records]

    def remove_facts(self, timestamps: Iterable[int]) -> None:
        set_timestamps = set(timestamps)
        self.__memory = [
            record for record in self.__memory if record.timestamp not in set_timestamps
        ]
