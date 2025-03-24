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
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
    )
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),
    )

    with open("./data/valentine_party.json", "r") as f:
        raw_data = ExperimentData.model_validate_json(f.read())

    agents = [LLMAgent(data, context=context) for data in raw_data.agents]
    [isabella, maria, klaus] = agents
    # print(isabella.data.agent_characteristics)
    # print(maria.data.agent_characteristics)
    # print(klaus.data.agent_characteristics)

    id_mapping = {i: agent for i, agent in enumerate(agents)}

    structure_graph = nx.Graph()
    structure_graph.add_edges_from(
        (id_mapping[first], id_mapping[second]) for (first, second) in raw_data.edges
    )
    start = time.time()
    await asyncio.gather(*[agent.agent_greeting_message for agent in agents])
    print("isabella", await isabella.agent_greeting_message)
    print("maria", await maria.agent_greeting_message)
    print("klaus", await klaus.agent_greeting_message)
    print(time.time() - start)
    conversation_selector = SequentialConversationSelector(
        structure=structure_graph,
        seed=np.random.default_rng(42),
        initial_conversation=[(isabella, maria)],
    )
    manager = ConversationManager(
        conversation_selector=conversation_selector,
        max_conversation_utterances=10,
    )
    await manager.run_simulation()
    print(await klaus.ask_agent("Do you know about the party ? Respond either yes or no and provide the exact date and time of the party"))
    print("total time: ", f"{time.time() - start}")


if __name__ == "__main__":
    dotenv.load_dotenv()

    asyncio.run(main())
