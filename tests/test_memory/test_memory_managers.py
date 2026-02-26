from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from generative_agents.memory.memory_managers import (
    CompositeBehaviorMemoryManager,
    ConstantContextBehavior,
)
from generative_agents.memory.simple_memory import SimpleMemory
from generative_agents.types import AgentModelBase, LLMAgentBase

if TYPE_CHECKING:
    from tests.conftest import MockLLMBackend


class StubAgentData(AgentModelBase):
    def __init__(self, full_name: str) -> None:
        self._full_name = full_name

    @property
    def full_name(self) -> str:
        return self._full_name


class StubAgent(LLMAgentBase):
    def __init__(self, name: str) -> None:
        self._data = StubAgentData(name)
        self._get_intro_called = False

    @property
    def data(self) -> AgentModelBase:
        return self._data

    async def get_agent_introduction_message(self) -> str:
        self._get_intro_called = True
        return f"Hello, I am {self._data.full_name}"


class TestCompositeBehaviorMemoryManager:
    def test_empty_behaviors_list(self, mock_llm_backend: "MockLLMBackend") -> None:
        memory = SimpleMemory()
        agent = StubAgent("TestAgent")
        manager = CompositeBehaviorMemoryManager(
            memory=memory,
            agent=agent,
            context=mock_llm_backend,
            behaviors=[],
        )
        result = manager.get_tagged_full_memory()
        assert "memory" in result

    def test_get_behavior_raises_if_not_found(
        self, mock_llm_backend: "MockLLMBackend"
    ) -> None:
        memory = SimpleMemory()
        agent = StubAgent("TestAgent")
        manager = CompositeBehaviorMemoryManager(
            memory=memory,
            agent=agent,
            context=mock_llm_backend,
            behaviors=[],
        )
        with pytest.raises(ValueError, match="Behavior of type.*not found"):
            manager.get_behavior(ConstantContextBehavior.get_impl_type())
        with pytest.raises(ValueError, match="Behavior of type.*not found"):
            manager.get_behavior(ConstantContextBehavior.Impl)

    def test_get_behavior_returns_correct_behavior(
        self, mock_llm_backend: "MockLLMBackend"
    ) -> None:
        memory = SimpleMemory()
        agent = StubAgent("TestAgent")
        behavior = ConstantContextBehavior("Test instructions")
        manager = CompositeBehaviorMemoryManager(
            memory=memory,
            agent=agent,
            context=mock_llm_backend,
            behaviors=[behavior],
        )

        result1 = manager.get_behavior(ConstantContextBehavior.Impl)
        assert isinstance(result1, ConstantContextBehavior.Impl)

        result2 = manager.get_behavior(ConstantContextBehavior.get_impl_type())
        assert isinstance(result2, ConstantContextBehavior.Impl)

    def test_get_tagged_full_memory_includes_behavior_extension(
        self, mock_llm_backend: "MockLLMBackend"
    ) -> None:
        memory = SimpleMemory()
        agent = StubAgent("TestAgent")
        behavior = ConstantContextBehavior("Test instructions")
        manager = CompositeBehaviorMemoryManager(
            memory=memory,
            agent=agent,
            context=mock_llm_backend,
            behaviors=[behavior],
        )
        result = manager.get_tagged_full_memory()
        assert "instructions" in result
        assert "Test instructions" in result
