from pydantic import BaseModel, Field
import dotenv
import os
from typing import Literal, override
import networkx as nx
from openai import AsyncOpenAI
import numpy as np
import asyncio
import logging

from generative_agents import (
    AgentModelBase,
    ConversationManager,
    LLMBackend,
    SequentialConversationSelector,
    LLMAgent,
)


class XMLExtraAdapter(logging.LoggerAdapter):
    @override
    def process(self, msg, kwargs):
        if "extra" in kwargs and len(kwargs["extra"]) > 0:
            content = "\n".join(
                f"<{key}>\n{value}\n</{key}>" for key, value in kwargs["extra"].items()
            )
            kwargs["extra"]["content"] = "\n" + content
        else:
            kwargs["extra"] = {"content": ""}
        return msg, kwargs


def get_logger(file_name: str, level) -> logging.Logger:
    logger = logging.Logger(file_name, level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s%(content)s"
    )
    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return XMLExtraAdapter(logger)


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
    logger = get_logger("logs/valentine_party.log", level=logging.DEBUG)

    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=api_key)
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),
        temperature=1.2,
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

    def get_memory(agent: LLMAgent):
        return "\n".join(
            [
                agent.memory.get_agent_memory(),
                "Facts about others:",
                agent.memory.dump_agents_knowledge(),
            ]
        )

    for agent in agents:
        logger.debug(agent.data.full_name, extra={"memory": get_memory(agent)})


# TODO: memory compression

# TODO: private naming, _ for private, __ for stronger private, mangled on runtime

if __name__ == "__main__":
    dotenv.load_dotenv()

    asyncio.run(main())
