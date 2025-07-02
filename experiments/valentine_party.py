from pydantic import BaseModel, Field
import dotenv
import os
from typing import Literal
import networkx as nx
from openai import AsyncOpenAI
import numpy as np
import asyncio
import logging
import os

from logger_utils import get_xml_file_logger
from generative_agents import (
    AgentModelBase,
    ConversationManager,
    LLMBackend,
    SequentialConversationSelector,
    LLMAgent,
    BDIMemoryManager,
    SimpleMemory,
    SimpleMemoryManager,
    EmbeddingMemory,
    get_fact_removal_probability_factory,
    mean_std_count_strategy_factory,
)
import httpx


class ExperimentAgent(AgentModelBase):
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    sex: Literal["F", "M"] = Field(..., description="Sex")
    description: str = Field(..., description="Agent characteristics and description")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def agent_characteristics(self) -> str:
        return self.model_dump_json()


class ExperimentData(BaseModel):
    agents: list[ExperimentAgent]
    # use implicit agent ordering, 0-indexed
    edges: list[tuple[int, int]]


async def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")
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
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
        embedding_model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    )

    with open("./data/valentine_party.json", "r") as f:
        raw_data = ExperimentData.model_validate_json(f.read())

    agents = [
        LLMAgent(
            data,
            context,
            lambda agent: BDIMemoryManager(
                EmbeddingMemory(
                    context, count_selector=mean_std_count_strategy_factory(0.5)
                ),
                agent=agent,
                memory_removal_probability=get_fact_removal_probability_factory(0.5),
            ),
        )
        for data in raw_data.agents
    ]
    [isabella, maria, klaus] = agents

    id_mapping = {i: agent for i, agent in enumerate(agents)}

    structure_graph = nx.Graph()
    structure_graph.add_edges_from(
        (id_mapping[first], id_mapping[second]) for (first, second) in raw_data.edges
    )

    await asyncio.gather(*[agent.get_agent_introduction_message() for agent in agents])
    logger.info("Agents initialized.")
    for agent in agents:
        logger.info(
            f"Introducing {agent.data.full_name}",
            extra={"introduction": await agent.get_agent_introduction_message()},
        )

    conversation_selector = SequentialConversationSelector(
        structure=structure_graph,
        seed=np.random.default_rng(42),
        initial_conversation=[(isabella, maria)],
    )
    manager = ConversationManager(
        conversation_selector=conversation_selector,
        max_conversation_utterances=12,
        logger=logger,
    )
    for i in range(4):
        await manager.run_simulation_epoch()

    question = "When is the party happening ? Did you hear about the party ?"
    results = asyncio.gather(*[agent.ask_agent(question) for agent in agents])
    for agent, answer in zip(agents, await results):
        print(f"{agent.data.full_name}: {answer}")

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

    for agent in agents:
        logger.debug(
            agent.data.full_name,
            extra={"memory": agent.memory_manager.get_tagged_full_memory()},
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())

# TODO" comments
# TODO: prekopat intro message aby to nebolo fixne definovane
