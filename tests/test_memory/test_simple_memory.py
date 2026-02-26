import pytest

from generative_agents.memory import (
    MemoryQueryFilter,
    MemoryRecordResponse,
    RecordSourceTypeBase,
    SimpleMemory,
)


class TestSimpleMemoryStoreFacts:
    @pytest.mark.asyncio
    async def test_store_single_fact_returns_timestamp(
        self,
        simple_memory: SimpleMemory,
        system_source: RecordSourceTypeBase,
    ) -> None:
        current_timestamp = simple_memory.current_timestamp()
        facts = [MemoryRecordResponse(text="Test fact", relevance=0.5)]
        timestamps = await simple_memory.store_facts(facts, system_source)
        assert timestamps == [current_timestamp + 1]

    @pytest.mark.asyncio
    async def test_store_multiple_facts_returns_incremental_timestamps(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        current_timestamp = simple_memory.current_timestamp()
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
            MemoryRecordResponse(text="Fact 3", relevance=0.7),
        ]
        timestamps = await simple_memory.store_facts(facts, system_source)
        assert timestamps == [
            current_timestamp + 1,
            current_timestamp + 2,
            current_timestamp + 3,
        ]

    @pytest.mark.asyncio
    async def test_store_empty_facts_returns_empty_list(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        timestamps = await simple_memory.store_facts([], system_source)
        assert timestamps == []


class TestSimpleMemoryFullRetrieval:
    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty_list(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        result = simple_memory.full_retrieval()
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_all_stored_facts(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        await simple_memory.store_facts(facts, system_source)
        result = simple_memory.full_retrieval()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_retrieval_preserves_data(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Test fact", relevance=0.8)]
        await simple_memory.store_facts(facts, system_source)
        result = simple_memory.full_retrieval()
        assert result[0].text == "Test fact"
        assert result[0].relevance == 0.8
        assert result[0].timestamp == 1
        assert result[0].source == system_source


class TestSimpleMemoryQueryRetrieval:
    @pytest.mark.asyncio
    async def test_query_retrieval_returns_all_in_simple_memory(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        await simple_memory.store_facts(facts, system_source)
        result = await simple_memory.query_retrieval("any query")
        assert len(result) == 2


class TestSimpleMemoryRemoveFacts:
    @pytest.mark.asyncio
    async def test_remove_existing_fact(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        timestamps = await simple_memory.store_facts(facts, system_source)
        simple_memory.remove_facts([timestamps[0]])
        result = simple_memory.full_retrieval()
        assert len(result) == 1
        assert result[0].timestamp == 2

    @pytest.mark.asyncio
    async def test_remove_non_existent_fact_no_change(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Fact 1", relevance=0.5)]
        await simple_memory.store_facts(facts, system_source)
        simple_memory.remove_facts([999])
        result = simple_memory.full_retrieval()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_remove_all_facts(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Fact 1", relevance=0.5),
            MemoryRecordResponse(text="Fact 2", relevance=0.6),
        ]
        timestamps = await simple_memory.store_facts(facts, system_source)
        simple_memory.remove_facts(timestamps)
        result = simple_memory.full_retrieval()
        assert result == []

    @pytest.mark.asyncio
    async def test_remove_duplicate_timestamps_handled(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Fact 1", relevance=0.5)]
        [timestamp] = await simple_memory.store_facts(facts, system_source)
        simple_memory.remove_facts([timestamp, timestamp])
        result = simple_memory.full_retrieval()
        assert result == []


class TestSimpleMemoryFilters:
    @pytest.mark.asyncio
    async def test_full_retrieval_with_source_filter(
        self,
        simple_memory: SimpleMemory,
        system_source: RecordSourceTypeBase,
        conversation_source: RecordSourceTypeBase,
    ) -> None:
        await simple_memory.store_facts(
            [MemoryRecordResponse(text="Fact 1", relevance=0.5)], system_source
        )
        await simple_memory.store_facts(
            [MemoryRecordResponse(text="Fact 2", relevance=0.5)], conversation_source
        )
        result = simple_memory.full_retrieval(
            MemoryQueryFilter(source_types=[type(system_source)])
        )
        assert len(result) == 1
        assert result[0].text == "Fact 1"

    @pytest.mark.asyncio
    async def test_full_retrieval_with_predicate_filter(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [
            MemoryRecordResponse(text="Important fact", relevance=0.9),
            MemoryRecordResponse(text="Less important", relevance=0.3),
        ]
        await simple_memory.store_facts(facts, system_source)
        result = simple_memory.full_retrieval(
            MemoryQueryFilter(predicate=lambda r: r.relevance > 0.5)
        )
        assert len(result) == 1
        assert result[0].text == "Important fact"

    @pytest.mark.asyncio
    async def test_full_retrieval_no_match_returns_empty(
        self, simple_memory: SimpleMemory, system_source: RecordSourceTypeBase
    ) -> None:
        facts = [MemoryRecordResponse(text="Fact 1", relevance=0.5)]
        await simple_memory.store_facts(facts, system_source)
        result = simple_memory.full_retrieval(
            MemoryQueryFilter(predicate=lambda r: r.timestamp > 100)
        )
        assert result == []
