from typing import TYPE_CHECKING

import numpy as np
import pytest

from generative_agents.memory.embedding_memory import (
    EmbeddingMemory,
    fixed_count_strategy_factory,
    mean_std_count_strategy_factory,
    top_std_count_strategy_factory,
)
from generative_agents.memory.models import (
    MemoryRecord,
    MemoryRecordResponse,
    RecordSourceTypeBase,
)

if TYPE_CHECKING:
    from tests.conftest import MockLLMBackend


class TestEmbeddingMemoryStoreFacts:
    @pytest.mark.asyncio
    async def test_store_single_fact(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Test fact", relevance=0.5)]
        timestamps = await embedding_memory.store_facts(facts, system_source)
        assert timestamps == [1]
        result = embedding_memory.full_retrieval()
        assert len(result) == 1
        assert result[0].text == "Test fact"

    @pytest.mark.asyncio
    async def test_store_multiple_facts(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
            MemoryRecordResponse(text="Fact 3", relevance=0.7),
        ]
        timestamps = await embedding_memory.store_facts(facts, system_source)
        assert timestamps == [1, 2, 3]
        result = embedding_memory.full_retrieval()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_store_assigns_embeddings(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Test fact", relevance=0.5)]
        await embedding_memory.store_facts(facts, system_source)
        result = embedding_memory.full_retrieval()
        assert result[0].embedding is not None
        assert isinstance(result[0].embedding, np.ndarray)


class TestEmbeddingMemoryFullRetrieval:
    @pytest.mark.asyncio
    async def test_empty_memory(self, embedding_memory: EmbeddingMemory) -> None:
        result = embedding_memory.full_retrieval()
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_all_facts(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        timestamps = await embedding_memory.store_facts(facts, system_source)
        result = embedding_memory.full_retrieval()
        assert [r.timestamp for r in result] == timestamps


class TestEmbeddingMemoryQueryRetrieval:
    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty(
        self, embedding_memory: EmbeddingMemory
    ) -> None:
        result = await embedding_memory.query_retrieval("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_records_with_embeddings(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Test fact", relevance=0.5)]
        await embedding_memory.store_facts(facts, system_source)
        result = await embedding_memory.query_retrieval("test query")
        assert len(result) == 1
        assert isinstance(result[0].embedding, np.ndarray)

    @pytest.mark.asyncio
    async def test_respects_count_selector(
        self, mock_llm_backend: "MockLLMBackend", system_source: RecordSourceTypeBase
    ) -> None:
        from generative_agents.memory.embedding_memory import EmbeddingMemory

        memory = EmbeddingMemory(
            context=mock_llm_backend,
            count_selector=fixed_count_strategy_factory(1),
        )
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        await memory.store_facts(facts, system_source)
        result = await memory.query_retrieval("test")
        assert len(result) == 1
        assert result[0].text == "Fact 2"  # Fact 2 has higher relevance and timestamp


class TestEmbeddingMemoryRemoveFacts:
    @pytest.mark.asyncio
    async def test_remove_existing_fact(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        [first_timestamp, second_timestamp] = await embedding_memory.store_facts(
            facts, system_source
        )
        embedding_memory.remove_facts([first_timestamp])
        result = embedding_memory.full_retrieval()
        assert len(result) == 1
        assert result[0].timestamp == second_timestamp

    @pytest.mark.asyncio
    async def test_remove_non_existent_no_change(
        self, embedding_memory: EmbeddingMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Fact 1", relevance=0.5)]
        [first] = await embedding_memory.store_facts(facts, system_source)
        embedding_memory.remove_facts([first + 1])  # non-existent timestamp
        result = embedding_memory.full_retrieval()
        assert len(result) == 1


class TestCountStrategyFactories:
    def test_fixed_count_strategy(self, system_source: RecordSourceTypeBase) -> None:
        strategy = fixed_count_strategy_factory(2)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
            (
                0.9,
                MemoryRecord(
                    timestamp=2, text="Fact 2", relevance=0.6, source=system_source
                ),
            ),
            (
                0.8,
                MemoryRecord(
                    timestamp=3, text="Fact 3", relevance=0.7, source=system_source
                ),
            ),
        ]
        assert strategy(records) == 2

    def test_fixed_count_strategy_more_than_available(
        self, system_source: RecordSourceTypeBase
    ) -> None:
        strategy = fixed_count_strategy_factory(10)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
            (
                0.9,
                MemoryRecord(
                    timestamp=2, text="Fact 2", relevance=0.6, source=system_source
                ),
            ),
            (
                0.8,
                MemoryRecord(
                    timestamp=3, text="Fact 3", relevance=0.7, source=system_source
                ),
            ),
        ]
        assert strategy(records) == 3

    def test_fixed_count_strategy_empty(self) -> None:
        strategy = fixed_count_strategy_factory(2)
        assert strategy([]) == 0

    def test_mean_std_strategy(self, system_source: RecordSourceTypeBase) -> None:
        strategy = mean_std_count_strategy_factory(std_coef=0.5)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
            (
                0.9,
                MemoryRecord(
                    timestamp=2, text="Fact 2", relevance=0.6, source=system_source
                ),
            ),
            (
                0.8,
                MemoryRecord(
                    timestamp=3, text="Fact 3", relevance=0.7, source=system_source
                ),
            ),
        ]
        count = strategy(records)
        assert count > 0
        assert count < len(records)

    def test_mean_std_strategy_empty(self) -> None:
        strategy = mean_std_count_strategy_factory(std_coef=0.5)
        assert strategy([]) == 0

    def test_mean_std_strategy_single_record(
        self, system_source: RecordSourceTypeBase
    ) -> None:
        strategy = mean_std_count_strategy_factory(std_coef=0.5)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            )
        ]
        assert strategy(records) == 1

    def test_mean_std_strategy_all_same_scores(
        self, system_source: RecordSourceTypeBase
    ) -> None:
        strategy = mean_std_count_strategy_factory(std_coef=0.5)
        records = [
            (
                0.5,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
            (
                0.5,
                MemoryRecord(
                    timestamp=2, text="Fact 2", relevance=0.6, source=system_source
                ),
            ),
            (
                0.5,
                MemoryRecord(
                    timestamp=3, text="Fact 3", relevance=0.7, source=system_source
                ),
            ),
        ]
        assert strategy(records) == 3

    def test_top_std_strategy(self, system_source: RecordSourceTypeBase) -> None:
        strategy = top_std_count_strategy_factory(std_coef=1.0)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
            (
                0.9,
                MemoryRecord(
                    timestamp=2, text="Fact 2", relevance=0.6, source=system_source
                ),
            ),
            (
                0.8,
                MemoryRecord(
                    timestamp=3, text="Fact 3", relevance=0.7, source=system_source
                ),
            ),
        ]
        count = strategy(records)
        assert count > 0

    def test_top_std_strategy_empty(self) -> None:
        strategy = top_std_count_strategy_factory(std_coef=1.0)
        assert strategy([]) == 0

    def test_top_std_strategy_single_record(
        self, system_source: RecordSourceTypeBase
    ) -> None:
        strategy = top_std_count_strategy_factory(std_coef=0.5)
        records = [
            (
                1.0,
                MemoryRecord(
                    timestamp=1, text="Fact 1", relevance=0.5, source=system_source
                ),
            ),
        ]
        assert strategy(records) == 1
