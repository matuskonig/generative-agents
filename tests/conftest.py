from typing import Any, overload

import numpy as np
import pytest
from numpy.random import Generator

from generative_agents.llm_backend import LLMBackendBase
from generative_agents.memory.embedding_memory import (
    EmbeddingMemory,
    fixed_count_strategy_factory,
)
from generative_agents.memory.models import BuildInSourceType, RecordSourceTypeBase
from generative_agents.memory.simple_memory import SimpleMemory


class MockLLMBackend(LLMBackendBase):
    def __init__(self, embedding_dim: int = 768) -> None:
        self._embedding_dim = embedding_dim
        self._text_responses: list[str] = []
        self._structured_responses: dict[type, Any] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self.embed_text_call_count = 0

    async def get_text_response(
        self,
        prompt: str,
        params: Any | None = None,
    ) -> str:
        if self._text_responses:
            return self._text_responses.pop(0)
        return ""

    async def get_structued_response(
        self,
        prompt: str,
        response_format: type,
        params: Any | None = None,
    ) -> Any:
        key = response_format
        if key in self._structured_responses:
            return self._structured_responses[key]
        return None

    @overload
    async def embed_text(self, input: str) -> np.ndarray: ...

    @overload
    async def embed_text(self, input: list[str]) -> list[np.ndarray]: ...

    async def embed_text(
        self,
        input: str | list[str],
    ) -> np.ndarray | list[np.ndarray]:
        self.embed_text_call_count += 1
        if isinstance(input, str):
            return np.ones(self._embedding_dim) * 0.1
        return [np.ones(self._embedding_dim) * 0.1 for _ in input]

    def add_text_response(self, response: str) -> None:
        self._text_responses.append(response)

    def add_structured_response(self, response_format: type, response: Any) -> None:
        self._structured_responses[response_format] = response


@pytest.fixture
def mock_llm_backend() -> MockLLMBackend:
    return MockLLMBackend()


@pytest.fixture
def simple_memory() -> SimpleMemory:
    return SimpleMemory()


@pytest.fixture
def embedding_memory(mock_llm_backend: MockLLMBackend) -> EmbeddingMemory:
    return EmbeddingMemory(
        context=mock_llm_backend, count_selector=fixed_count_strategy_factory(10)
    )


@pytest.fixture
def seeded_rng() -> Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def system_source() -> RecordSourceTypeBase:
    return BuildInSourceType.System()


@pytest.fixture
def conversation_source() -> RecordSourceTypeBase:
    return BuildInSourceType.Conversation(other_agent="Alice")
