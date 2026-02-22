"""Valentine party experiment demonstrating information spread in a social network.

This experiment simulates a small social network where information about a party
propagates through conversations between agents.
"""

import asyncio
import logging
import os
from typing import Literal

import dotenv
import httpx
import networkx as nx
import numpy as np
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from generative_agents import (
    AgentModelBase,
    BDIPlanningBehavior,
    CompositeBehaviorFactoryBase,
    CompositeBehaviorMemoryManager,
    ConversationManager,
    ConversationMemoryUpdatingBehavior,
    EmbeddingMemory,
    LLMBackend,
    LLMConversationAgent,
    MemoryForgettingBehavior,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    SequentialConversationSelector,
    SimpleMemory,
    get_record_removal_linear_probability,
    mean_std_count_strategy_factory,
)


class ExperimentAgent(AgentModelBase):
    """Agent model for the Valentine party experiment."""

    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    sex: Literal["F", "M"] = Field(..., description="Sex")
    description: str = Field(..., description="Agent characteristics and description")

    @property
    def full_name(self) -> str:
        """Return full name as first name + last name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def agent_characteristics(self) -> str:
        """Return agent characteristics as JSON."""
        return self.model_dump_json()


class ExperimentData(BaseModel):
    """Data structure for the Valentine party experiment."""

    agents: list[ExperimentAgent]
    # use implicit agent ordering, 0-indexed
    edges: list[tuple[int, int]]


async def main():
    """Main function running the Valentine party experiment."""
    seed = np.random.default_rng(42)

    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Set up logger for experiment output
    logger = get_xml_file_logger("logs/valentine_party.log", level=logging.DEBUG)

    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
        http_client=httpx.AsyncClient(
            http2=True,
            timeout=120.0,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=20),
        ),
    )

    # Create LLM backend for generating responses
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
        embedding_provider=OpenAIEmbeddingProvider(
            client=client,
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),  # type: ignore
        ),
    )

    behaviors: list[CompositeBehaviorFactoryBase] = [
        ConversationMemoryUpdatingBehavior(),
        BDIPlanningBehavior(),
        MemoryForgettingBehavior(get_record_removal_linear_probability(0.5), seed=seed),
    ]

    # Load Valentine party dataset
    with open("./data/valentine_party.json", "r") as f:
        raw_data = ExperimentData.model_validate_json(f.read())

    agents = [
        LLMConversationAgent(
            data,
            context,
            lambda agent: CompositeBehaviorMemoryManager(
                EmbeddingMemory(
                    context, count_selector=mean_std_count_strategy_factory(0.5)
                ),
                agent,
                context,
                behaviors,
            ),
        )
        for data in raw_data.agents
    ]
    [isabella, maria, klaus] = agents

    id_mapping = {i: agent for i, agent in enumerate(agents)}

    # Build social network graph from edges
    structure_graph = nx.Graph()
    structure_graph.add_edges_from(
        (id_mapping[first], id_mapping[second]) for (first, second) in raw_data.edges
    )

    # Initialize all agents with introduction messages
    await asyncio.gather(*[agent.get_agent_introduction_message() for agent in agents])
    logger.info("Agents initialized.")
    for agent in agents:
        logger.info(
            f"Introducing {agent.data.full_name}",
            extra={"introduction": await agent.get_agent_introduction_message()},
        )

    # Set up sequential conversation selector starting with Isabella and Maria
    conversation_selector = SequentialConversationSelector(
        structure=structure_graph,
        seed=np.random.default_rng(42),
        initial_conversation=[(isabella, maria)],
    )
    manager = ConversationManager(
        conversation_selector=conversation_selector,
        max_conversation_utterances=4,
        logger=logger,
    )

    # Run epochs of conversation simulation
    for i in range(2):
        await manager.run_simulation_epoch()

    # Query all agents about the party
    question = "When is the party happening ? Did you hear about the party ?"
    results = asyncio.gather(*[agent.ask_agent(question) for agent in agents])
    for agent, answer in zip(agents, await results):
        print(f"{agent.data.full_name}: {answer}")

    # Print usage statistics
    print()
    print("Usage statistics")
    print(f"Total time: {context.total_time:.02f} s")
    print(f"Completion tokens: {context.completion_tokens}")
    print(f"Prompt tokens: {context.prompt_tokens}")
    print(f"Tokens per second: {context.completion_tokens / context.total_time:.2f}")
    print(
        f"Total requests: {context.total_requests}, avg input: {context.prompt_tokens/context.total_requests}, avt output: {context.completion_tokens/context.total_requests}"
    )
    print(f"Avg time per request: {context.total_time/context.total_requests}")

    # Log final memory state for each agent
    for agent in agents:
        logger.debug(
            agent.data.full_name,
            extra={"memory": agent.memory_manager.get_tagged_full_memory()},
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
