from pydantic import BaseModel, Field

import dotenv
import os
from typing import Literal
import networkx as nx
from openai import AsyncOpenAI

from generative_multiagents import (
    AgentModelBase,
    ConversationManager,
    LLMBackend,
    SequentialConversationSelector,
    LLMAgent,
)
import numpy as np
import asyncio
import time
from functools import cached_property, lru_cache


# TODO: characteristiscs as description ?
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
    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=api_key)
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),
        temperature=1.5,
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),
    )

    with open("./data/valentine_party.json", "r") as f:
        raw_data = ExperimentData.model_validate_json(f.read())

    agents = [LLMAgent(data, context=context) for data in raw_data.agents]
    [isabella, maria, klaus] = agents

    id_mapping = {i: agent for i, agent in enumerate(agents)}

    structure_graph = nx.Graph()
    structure_graph.add_edges_from(
        (id_mapping[first], id_mapping[second]) for (first, second) in raw_data.edges
    )

    await asyncio.gather(*[agent.get_agent_introduction_message() for agent in agents])
    print("isabella")
    print(await isabella.get_agent_introduction_message())
    print()
    print("maria")
    print(await maria.get_agent_introduction_message())
    print()
    print("klaus")
    print(await klaus.get_agent_introduction_message())
    print()

    conversation_selector = SequentialConversationSelector(
        structure=structure_graph,
        seed=np.random.default_rng(42),
        initial_conversation=[(isabella, maria)],
    )
    manager = ConversationManager(
        conversation_selector=conversation_selector, max_conversation_utterances=12
    )
    for i in range(6):
        await manager.run_simulation()

    question = "When is the party happening ? Did you hear about the party ?"
    results = asyncio.gather(*[agent.ask_agent(question) for agent in agents])
    for agent, answer in zip(agents, await results):
        print(f"{agent.data.full_name}: {answer}")

    # print("isabella")
    # isabella.print_memory()
    # print("maria")
    # maria.print_memory()
    # print("klaus")
    # klaus.print_memory()


# TODO: memory compression

# TODO: private naming, _ for private, __ for stronger private, mangled on runtime

if __name__ == "__main__":
    dotenv.load_dotenv()

    asyncio.run(main())
