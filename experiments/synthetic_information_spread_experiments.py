from pydantic import BaseModel, Field
import dotenv
import os
from typing import Literal, TypedDict
import networkx as nx
from openai import AsyncOpenAI
import numpy as np
import asyncio
import logging
import os

from data_generation_utils import Dataset, SocietyAgent
from generative_agents import (
    ConversationManager,
    LLMBackend,
    LLMAgent,
    BDIMemoryManager,
    SimpleMemory,
    SimpleMemoryManager,
    InformationSpreadConversationSelector,
    FullParallelConversationSelector,
    EmbeddingMemory,
    get_fact_removal_probability_factory,
    mean_std_count_strategy_factory,
    fixed_count_strategy_factory,
    top_std_count_strategy_factory,
    default_config,
    DefaultConfig,
    AgentModelBase,
)
import httpx


from logger_utils import get_xml_file_logger


class BDIMemoryManagerType(TypedDict):
    manager_type: Literal["bdi"]
    memory_removal_prob: float


class SimpleMemoryManagerType(TypedDict):
    manager_type: Literal["simple"]


class SimpleMemoryType(TypedDict):
    memory_type: Literal["simple"]


class EmbeddingMemoryType(TypedDict):
    memory_type: Literal["embedding"]
    strategy: Literal["top_k"] | Literal["mean_std"] | Literal["top_std"]
    value: int | float


class QuestionAnswer(BaseModel):
    reasoning: str
    final_answer: bool


class ExperimentResult(BaseModel):
    dataset: Dataset
    experiment_name: str
    memory_manager_config: BDIMemoryManagerType | SimpleMemoryManagerType
    memory_config: SimpleMemoryType | EmbeddingMemoryType
    conversation_selector_type: Literal["information_spread", "full_parallel"]
    seed: int
    max_utterances: int
    epochs: int
    total_time: float
    completion_tokens: int
    prompt_tokens: int
    total_requests: int
    epoch_agents_responses: list[list[QuestionAnswer]]


class ReducedInformationSpreadConfig(DefaultConfig):

    def start_conversation_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_content) if memory_content.strip() else ""
        )

        return f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You are about to meet {second_agent_full_name}. Based on your personality and any relevant memories:

1. Consider your natural approach to meeting someone
2. Think about what kind of conversation starter fits your character
3. Be authentic to your communication style
4. Make the greeting feel natural and engaging

How would you initiate this conversation?"""

    def generate_next_turn_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
        conversation,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_content) if memory_content.strip() else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Current conversation with {second_agent_full_name}:
{self.conversation_to_tagged_text(conversation)}

Your turn to respond. Consider:
- The conversation's natural flow and context
- Your relationship with {second_agent_full_name}
- Your personality and communication style
- Whether to continue the current topic, transition, or conclude

Guidelines:
- Stay true to your character
- Respond appropriately to what was just said
- If the conversation feels stagnant or complete, you may gracefully end it
- Keep responses natural and engaging
- Address any direct questions or comments
"""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt


async def run_experiment(
    context: LLMBackend,
    dataset: Dataset,
    logger: logging.Logger,
    experiment_name: str,
    memory_manager_config: BDIMemoryManagerType | SimpleMemoryManagerType,
    memory_config: SimpleMemoryType | EmbeddingMemoryType,
    conversation_selector_type: Literal[
        "information_spread", "full_parallel"
    ] = "information_spread",
    seed=42,
    max_utterances=12,
    epochs=5,
):
    def get_agent_memory_manager(agent: LLMAgent):
        match memory_config["memory_type"]:
            case "simple":
                memory = SimpleMemory()
            case "embedding":
                match memory_config["strategy"]:
                    case "top_k":
                        count_selector = fixed_count_strategy_factory(
                            int(memory_config["value"])
                        )
                    case "mean_std":
                        count_selector = mean_std_count_strategy_factory(
                            memory_config["value"]
                        )
                    case "top_std":
                        count_selector = top_std_count_strategy_factory(
                            memory_config["value"]
                        )
                memory = EmbeddingMemory(context, count_selector)

        match memory_manager_config["manager_type"]:
            case "simple":
                return SimpleMemoryManager(memory, agent)
            case "bdi":
                return BDIMemoryManager(
                    memory,
                    agent,
                    get_fact_removal_probability_factory(
                        memory_manager_config["memory_removal_prob"]
                    ),
                )

    agents = [
        LLMAgent(data, context, get_agent_memory_manager) for data in dataset.agents
    ]
    id_mapping = {i: agent for i, agent in enumerate(agents)}

    structure_graph = nx.Graph()
    structure_graph.add_edges_from(
        (id_mapping[first], id_mapping[second]) for (first, second) in dataset.edges
    )

    await asyncio.gather(*[agent.get_agent_introduction_message() for agent in agents])
    logger.info("Agents initialized.")
    for agent in agents:
        logger.info(
            f"Introducing {agent.data.full_name}",
            extra={"introduction": await agent.get_agent_introduction_message()},
        )

    match conversation_selector_type:
        case "information_spread":
            conversation_selector = InformationSpreadConversationSelector(
                structure=structure_graph,
                seed_nodes=[id_mapping[dataset.information_seed_agent]],
                seed=np.random.default_rng(seed),
            )
        case "full_parallel":
            conversation_selector = FullParallelConversationSelector(
                structure=structure_graph,
                seed=np.random.default_rng(seed),
            )

    manager = ConversationManager(
        conversation_selector=conversation_selector,
        max_conversation_utterances=max_utterances,
        logger=logger,
    )

    epoch_responses: list[list[QuestionAnswer]] = []

    for epoch in range(epochs):
        print(f"Running epoch {epoch + 1}/{epochs}...")
        await manager.run_simulation_epoch()
        print(f"Epoch {epoch + 1} completed.")
        agent_responses = await asyncio.gather(
            *[
                agent.ask_agent_structured(
                    f"""Based on your memory and recent conversations, check if you have received specific information from {id_mapping[dataset.information_seed_agent].data.full_name}.

The information in question is: "{dataset.injected_information}"

Please review your memories and conversations to determine if you have encountered this information. Provide your reasoning based on what you remember, then give a definitive true/false answer about whether you received this information.
Answer positively even if you have received this information partially or indirectly.
""",
                    response_format=QuestionAnswer,
                )
                for agent in agents
            ]
        )
        print(f"Epoch {epoch + 1} collected responses.")
        epoch_responses.append(agent_responses)

    manager.reset_epochs()

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
    experiment_result = ExperimentResult(
        dataset=dataset,
        experiment_name=experiment_name,
        memory_manager_config=memory_manager_config,
        memory_config=memory_config,
        conversation_selector_type=conversation_selector_type,
        seed=seed,
        max_utterances=max_utterances,
        epochs=epochs,
        total_time=context.total_time,
        completion_tokens=context.completion_tokens,
        prompt_tokens=context.prompt_tokens,
        total_requests=context.total_requests,
        epoch_agents_responses=epoch_responses,
    )
    return experiment_result


async def main():
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    with open("./data/synthetic_5.json", "r") as f:
        dataset5 = Dataset.model_validate_json(f.read())
    with open("./data/synthetic_10.json", "r") as f:
        dataset10 = Dataset.model_validate_json(f.read())
    with open("./data/synthetic_25.json", "r") as f:
        dataset25 = Dataset.model_validate_json(f.read())
    with open("./data/synthetic_50.json", "r") as f:
        dataset50 = Dataset.model_validate_json(f.read())
    with open("./data/synthetic_100.json", "r") as f:
        dataset100 = Dataset.model_validate_json(f.read())

    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
        http_client=httpx.AsyncClient(
            http2=True,
            timeout=180.0,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=20),
        ),
    )
    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
        embedding_model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    )

    if not os.path.exists("./results"):
        os.makedirs("./results")

    result5 = await run_experiment(
        context,
        dataset5,
        get_xml_file_logger("./logs/synthetic_5_bdi_is.log", level=logging.DEBUG),
        "synthetic_5_bdi_is",
        BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5),
        EmbeddingMemoryType(memory_type="embedding", strategy="top_std", value=1),
        conversation_selector_type="information_spread",
        seed=42,
        max_utterances=16,
        epochs=20,
    )
    with open("./results/synthetic_5_bdi_is.json", "w") as f:
        f.write(result5.model_dump_json(indent=1))

    # Same as result5, but with half the utterances and epochs
    result5_reduced = await run_experiment(
        context,
        dataset5,
        get_xml_file_logger(
            "./logs/synthetic_5_bdi_is_reduced.log", level=logging.DEBUG
        ),
        "synthetic_5_bdi_is_reduced",
        BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5),
        EmbeddingMemoryType(memory_type="embedding", strategy="top_std", value=1),
        conversation_selector_type="information_spread",
        seed=42,
        max_utterances=8,  # Half of 16
        epochs=10,  # Half of 20
    )
    with open("./results/synthetic_5_bdi_is_reduced.json", "w") as f:
        f.write(result5_reduced.model_dump_json(indent=1))

    # Same as result5, but with simple memory
    result5_simple = await run_experiment(
        context,
        dataset5,
        get_xml_file_logger(
            "./logs/synthetic_5_bdi_is_simple.log", level=logging.DEBUG
        ),
        "synthetic_5_bdi_is_simple",
        BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5),
        SimpleMemoryType(memory_type="simple"),
        conversation_selector_type="information_spread",
        seed=42,
        max_utterances=16,
        epochs=20,
    )
    with open("./results/synthetic_5_bdi_is_simple.json", "w") as f:
        f.write(result5_simple.model_dump_json(indent=1))

    # Same as result5, but with crippled prompt engineering
    with default_config.override(ReducedInformationSpreadConfig()):
        result5_reduced_prompt = await run_experiment(
            context,
            dataset5,
            get_xml_file_logger(
                "./logs/synthetic_5_bdi_is_reduced_prompt.log", level=logging.DEBUG
            ),
            "synthetic_5_bdi_is_reduced_prompt",
            BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5),
            EmbeddingMemoryType(memory_type="embedding", strategy="top_std", value=1),
            conversation_selector_type="information_spread",
            seed=42,
            max_utterances=16,
            epochs=20,
        )
        with open("./results/synthetic_5_bdi_is_reduced_prompt.json", "w") as f:
            f.write(result5_reduced_prompt.model_dump_json(indent=1))

    result10 = await run_experiment(
        context,
        dataset10,
        get_xml_file_logger("./logs/synthetic_10_bdi_is.log", level=logging.DEBUG),
        "synthetic_10_bdi_is",
        BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5),
        EmbeddingMemoryType(memory_type="embedding", strategy="top_std", value=1),
        conversation_selector_type="information_spread",
        seed=42,
        max_utterances=16,
        epochs=30,
    )
    with open("./results/synthetic_10_bdi_is.json", "w") as f:
        f.write(result10.model_dump_json(indent=1))


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
