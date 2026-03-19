"""Experiments for testing information spread through a social network.

This module contains experiments that simulate information propagation through
agent-based social networks, testing how information spreads from seed agents
to other agents through conversations.
"""

import argparse
import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Literal, TypedDict, Union

import dotenv
import httpx
import networkx as nx
import numpy as np
from data_generation_utils import Dataset, SocietyAgent
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from generative_agents import (
    AgentModelBase,
    BDIPlanningBehavior,
    CompositeBehaviorFactoryBase,
    CompositeBehaviorMemoryManager,
    ConversationManager,
    ConversationMemoryForgettingBehavior,
    ConversationMemoryUpdatingBehavior,
    DefaultConfig,
    EmbeddingMemory,
    FullParallelConversationSelector,
    InformationSpreadConversationSelector,
    LLMBackend,
    LLMConversationAgent,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    SequentialConversationSelector,
    SimpleMemory,
    UnitaryAgentNoteUpdatingBehavior,
    default_config,
    fixed_count_strategy_factory,
    get_record_removal_linear_probability,
    mean_std_count_strategy_factory,
    top_std_count_strategy_factory,
)

# =============================================================================
# Configuration TypedDicts
# =============================================================================


class BDIMemoryManagerType(TypedDict):
    """Configuration for BDI-based memory management with forgetting mechanism."""

    manager_type: Literal["bdi"]
    memory_removal_prob: float


class SimpleMemoryManagerType(TypedDict):
    """Configuration for simple memory management without BDI or forgetting."""

    manager_type: Literal["simple"]


class BDIPLanningOnlyManagerType(TypedDict):
    """BDI with planning only, no forgetting."""

    manager_type: Literal["bdi_planning_only"]
    memory_removal_prob: float


class BDIForgettingOnlyManagerType(TypedDict):
    """BDI with forgetting only, no planning behavior."""

    manager_type: Literal["forgetting_only"]
    memory_removal_prob: float


class SimpleMemoryType(TypedDict):
    """Configuration for simple memory (no embeddings)."""

    memory_type: Literal["simple"]


class EmbeddingMemoryType(TypedDict):
    """Configuration for embedding-based memory with retrieval strategies."""

    memory_type: Literal["embedding"]
    strategy: Literal["top_k"] | Literal["mean_std"] | Literal["top_std"]
    value: int | float


class UpdaterBehaviorType(TypedDict):
    """Configuration for updating behavior type."""

    behavior_type: Literal["classical"] | Literal["unitary"]


# Union type for all memory manager configs
MemoryManagerConfig = Union[
    BDIMemoryManagerType,
    SimpleMemoryManagerType,
    BDIPLanningOnlyManagerType,
    BDIForgettingOnlyManagerType,
]

EXPERIMENT_MAX_UTTERANCES = 16
EXPERIMENT_EPOCHS = 20


# =============================================================================
# Pydantic Models
# =============================================================================


class QuestionAnswer(BaseModel):
    """Response model for agent question answering about information received."""

    reasoning: str
    final_answer: bool


class ExperimentResult(BaseModel):
    """Results from a single experiment run."""

    dataset: Dataset
    experiment_name: str
    memory_manager_config: MemoryManagerConfig
    memory_config: SimpleMemoryType | EmbeddingMemoryType
    conversation_selector_type: Literal[
        "information_spread", "full_parallel", "sequential"
    ]
    updater_behavior_type: UpdaterBehaviorType
    seed: int
    max_utterances: int
    epochs: int
    total_time: float
    wallclock_time: float
    completion_tokens: int
    prompt_tokens: int
    total_requests: int
    epoch_agents_responses: list[list[QuestionAnswer]]


# =============================================================================
# Config Classes
# =============================================================================


class ReducedInformationSpreadConfig(DefaultConfig):
    """Configuration with simplified prompts to test information spread with reduced prompting."""

    def get_introduction_prompt(self, agent_data):
        return f"""Your name is {agent_data.full_name}. 

Your characteristics: {agent_data.agent_characteristics}

Create a personal introduction that:
1. Establishes your unique personality and communication style
2. Includes key aspects of your background and interests
3. Shows how you typically interact with others
4. Demonstrates your distinctive way of speaking

In this introduction include information about yourself such as:
- interesting hobbies and current activities
- events you are organizing or participating in
- your personal projects and professional endeavors
- recent news or developments in your life
- topics you're passionate about discussing

Keep it authentic and conversational. This introduction will define how others perceive you and what information they associate with you.

Do not mention being an AI, agent, or simulation. Write as a real person."""

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

Keep your greeting concise but engaging - a few sentences that establish connection.

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
- Keep responses natural and conversational - match the flow of the conversation
- Keep responses concise (1-3 sentences for typical turns)
- Don't add filler or repeat what's already been said
- Add new information or perspective - don't just acknowledge what was said
- Address any direct questions or comments"""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_conversation_summary_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just completed this conversation with {other_agent_full_name}:
{conversation_string}

Extract meaningful information from this conversation that should be remembered:

1. **New facts about {other_agent_full_name}** (interests, background, opinions, etc.)
2. **Important topics discussed** (specific details, not general knowledge)
3. **Relationship developments** (how your interaction evolved)
4. **Future-relevant information** (plans, commitments, shared interests)

Guidelines:
- Focus on information that wasn't already in your memory
- Prioritize details that could influence future interactions
- Be specific but concise, especially with dates, locations, and key details
- Avoid recording general world knowledge or obvious facts
- Select only the most important facts to remember. Keep the number of remembered facts small (ideally 3-5 key points)."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_bdi_init_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Based on your personality and current situation, define your goals and intentions:

**DESIRES** - Multiple goals you might pursue in future conversations:
- Consider your personality traits and interests
- Think about what would motivate someone like you
- Include both short-term and longer-term aspirations
- Make them specific and achievable through social interaction

**INTENTION** - Choose ONE desire as your primary focus:
- This will guide your behavior in upcoming conversations
- Select the most important or urgent goal for now
- You can change this later based on circumstances

Remember: Your desires reflect who you are, and your intention drives what you'll actively work toward."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_bdi_update_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ):
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just finished this conversation with {other_agent_full_name}:
{conversation_string}

Review and update your goals based on this interaction:

**OPTIONS:**
1. **Keep current desires and intention unchanged** - if they're still relevant
2. **Change intention only** - switch focus to a different existing desire
3. **Update both desires and intention** - if circumstances have significantly changed

**WHEN UPDATING DESIRES** - Consider multiple goals you might pursue in future conversations:
- Your personality traits and interests
- What would motivate someone like you based on this conversation
- Both short-term and longer-term aspirations that emerged
- Goals that are specific and achievable through social interaction

**WHEN UPDATING INTENTION** - Choose ONE desire as your primary focus:
- This will guide your behavior in upcoming conversations
- Select the most important or urgent goal based on recent developments
- Think about what this conversation revealed about opportunities or priorities

**CONSIDERATIONS:**
- Did this conversation reveal new opportunities for you?
- Has your relationship with {other_agent_full_name} opened new possibilities?
- Do you need to adjust your priorities based on what you learned?

Remember: Your desires reflect who you are and what you've learned, and your intention drives what you'll actively work toward."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt


# =============================================================================
# Experiment Runner
# =============================================================================


async def run_experiment(
    get_context: Callable[[], LLMBackend],
    dataset: Dataset,
    logger: logging.Logger,
    experiment_name: str,
    memory_manager_config: MemoryManagerConfig,
    memory_config: SimpleMemoryType | EmbeddingMemoryType,
    conversation_selector_type: Literal[
        "information_spread", "full_parallel", "sequential"
    ] = "information_spread",
    updater_behavior_type: UpdaterBehaviorType = UpdaterBehaviorType(
        behavior_type="classical"
    ),
    seed: int = 42,
    max_utterances: int = 12,
    epochs: int = 5,
) -> ExperimentResult:
    """Run a single experiment with the given configuration.

    Args:
        get_context: LLM backend for generating responses
        dataset: Dataset containing agents and social network structure
        logger: Logger for experiment output
        experiment_name: Name identifier for the experiment
        memory_manager_config: Configuration for memory management
        memory_config: Configuration for memory type (simple or embedding)
        conversation_selector_type: How conversations are selected
        updater_behavior_type: Type of memory updating behavior (classical or unitary)
        seed: Random seed for reproducibility
        max_utterances: Maximum turns per conversation
        epochs: Number of conversation epochs to run

    Returns:
        ExperimentResult containing all experiment metrics and responses
    """
    context = get_context()
    seed_rng = np.random.default_rng(seed)

    start_time = time.time()

    def get_updater_behavior():
        """Get the memory updater behavior based on configuration."""
        if updater_behavior_type["behavior_type"] == "unitary":
            return UnitaryAgentNoteUpdatingBehavior()
        return ConversationMemoryUpdatingBehavior()

    def get_agent_memory_manager(agent: LLMConversationAgent):
        """Create memory manager based on configuration."""
        # Create memory based on type
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

        # Create manager based on type
        match memory_manager_config["manager_type"]:
            case "simple":
                return CompositeBehaviorMemoryManager(
                    memory,
                    agent,
                    context,
                    [ConversationMemoryUpdatingBehavior()],
                )
            case "bdi":
                return CompositeBehaviorMemoryManager(
                    memory,
                    agent,
                    context,
                    [
                        get_updater_behavior(),
                        BDIPlanningBehavior(),
                        ConversationMemoryForgettingBehavior(
                            get_record_removal_linear_probability(
                                memory_manager_config["memory_removal_prob"]
                            ),
                            seed=seed_rng,
                        ),
                    ],
                )
            case "bdi_planning_only":
                return CompositeBehaviorMemoryManager(
                    memory,
                    agent,
                    context,
                    [
                        get_updater_behavior(),
                        BDIPlanningBehavior(),
                    ],
                )
            case "forgetting_only":
                return CompositeBehaviorMemoryManager(
                    memory,
                    agent,
                    context,
                    [
                        get_updater_behavior(),
                        ConversationMemoryForgettingBehavior(
                            get_record_removal_linear_probability(
                                memory_manager_config["memory_removal_prob"]
                            ),
                            seed=seed_rng,
                        ),
                    ],
                )

    # Create agents from dataset
    agents = [
        LLMConversationAgent(data, context, get_agent_memory_manager)
        for data in dataset.agents
    ]
    id_mapping = {i: agent for i, agent in enumerate(agents)}

    # Build social network graph from dataset edges
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
        case "sequential":
            conversation_selector = SequentialConversationSelector(
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
        print(f"[{experiment_name}]: Running epoch {epoch + 1}/{epochs}...")
        await manager.run_simulation_epoch()
        # Query all agents about whether they received the information
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
        epoch_responses.append(agent_responses)

    manager.reset_epochs()

    # Log final memory state for each agent
    for agent in agents:
        logger.debug(
            agent.data.full_name,
            extra={
                "memory": agent.memory_manager.get_tagged_full_memory(
                    with_full_memory_record=True
                )
            },
        )

    experiment_result = ExperimentResult(
        dataset=dataset,
        experiment_name=experiment_name,
        memory_manager_config=memory_manager_config,
        memory_config=memory_config,
        conversation_selector_type=conversation_selector_type,
        updater_behavior_type=updater_behavior_type,
        seed=seed,
        max_utterances=max_utterances,
        epochs=epochs,
        wallclock_time=time.time() - start_time,
        total_time=context.total_time,
        completion_tokens=context.completion_tokens,
        prompt_tokens=context.prompt_tokens,
        total_requests=context.total_requests,
        epoch_agents_responses=epoch_responses,
    )
    return experiment_result


# =============================================================================
# Main Experiment Suite
# =============================================================================


async def main(concurrency: int, embedding_batch_size: int, use_local_embeddings: bool):
    """Run all experiments for the information spread study.

    Args:
        concurrency: Number of experiments to run in parallel (default: 1 = sequential)
    """
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Load datasets
    with open("./data/synthetic_5.json", "r") as f:
        dataset5 = Dataset.model_validate_json(f.read())
    with open("./data/synthetic_10.json", "r") as f:
        dataset10 = Dataset.model_validate_json(f.read())

    # Setup client
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

    embedding_provider = (
        SentenceTransformerProvider(
            "BAAI/bge-m3",
            device="cuda",
            batch_size=embedding_batch_size,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        if use_local_embeddings
        else OpenAIEmbeddingProvider(
            client=client,
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),  # type: ignore
        )
    )

    def get_context() -> LLMBackend:
        return LLMBackend(
            client=client,
            model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
            RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
            embedding_provider=embedding_provider,
        )

    # Shared configs
    baseline_mem_mgr = BDIMemoryManagerType(manager_type="bdi", memory_removal_prob=0.5)
    baseline_mem = EmbeddingMemoryType(
        memory_type="embedding", strategy="mean_std", value=0.5
    )
    baseline_updater = UpdaterBehaviorType(behavior_type="classical")

    simple_mem_mgr = SimpleMemoryManagerType(manager_type="simple")
    simple_mem = SimpleMemoryType(memory_type="simple")

    semaphore = asyncio.Semaphore(concurrency)

    async def run_and_save(result: Awaitable[ExperimentResult], file_name: str) -> None:
        """Run experiment coroutine with semaphore and save result to file."""
        try:
            async with semaphore:
                r = await result
            with open(file_name, "w") as f:
                f.write(r.model_dump_json(indent=1))
        except Exception as e:
            print(f"Error in experiment {file_name}: {e}")

    async with asyncio.TaskGroup() as tg:
        # === CATEGORY A: Baseline + Prompting (A1-A8) ===
        # A1: baseline_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/A1_baseline_5.log", level=logging.DEBUG
                    ),
                    experiment_name="A1_baseline_5",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/A1_baseline_5.json",
            )
        )

        # A2: baseline_5_redprompt
        with default_config.override(ReducedInformationSpreadConfig()):
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            "./logs/A2_baseline_5_redprompt.log", level=logging.DEBUG
                        ),
                        experiment_name="A2_baseline_5_redprompt",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=baseline_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    "./results/A2_baseline_5_redprompt.json",
                )
            )

        # A3: simple_no_bdi_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/A3_simple_no_bdi_5.log", level=logging.DEBUG
                    ),
                    experiment_name="A3_simple_no_bdi_5",
                    memory_manager_config=simple_mem_mgr,
                    memory_config=simple_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/A3_simple_no_bdi_5.json",
            )
        )

        # A4: simple_no_bdi_5_red
        with default_config.override(ReducedInformationSpreadConfig()):
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            "./logs/A4_simple_no_bdi_5_red.log", level=logging.DEBUG
                        ),
                        experiment_name="A4_simple_no_bdi_5_red",
                        memory_manager_config=simple_mem_mgr,
                        memory_config=simple_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    "./results/A4_simple_no_bdi_5_red.json",
                )
            )

        # A5: baseline_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/A5_baseline_10.log", level=logging.DEBUG
                    ),
                    experiment_name="A5_baseline_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/A5_baseline_10.json",
            )
        )

        # A6: baseline_10_redprompt
        with default_config.override(ReducedInformationSpreadConfig()):
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            "./logs/A6_baseline_10_redprompt.log", level=logging.DEBUG
                        ),
                        experiment_name="A6_baseline_10_redprompt",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=baseline_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    "./results/A6_baseline_10_redprompt.json",
                )
            )

        # A7: simple_no_bdi_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/A7_simple_no_bdi_10.log", level=logging.DEBUG
                    ),
                    experiment_name="A7_simple_no_bdi_10",
                    memory_manager_config=simple_mem_mgr,
                    memory_config=simple_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/A7_simple_no_bdi_10.json",
            )
        )

        # A8: simple_no_bdi_10_red
        with default_config.override(ReducedInformationSpreadConfig()):
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            "./logs/A8_simple_no_bdi_10_red.log", level=logging.DEBUG
                        ),
                        experiment_name="A8_simple_no_bdi_10_red",
                        memory_manager_config=simple_mem_mgr,
                        memory_config=simple_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    "./results/A8_simple_no_bdi_10_red.json",
                )
            )

        # === CATEGORY B: Memory Manager Comparison (B1-B6) ===
        # B1: unitary_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/B1_unitary_5.log", level=logging.DEBUG
                    ),
                    experiment_name="B1_unitary_5",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=UpdaterBehaviorType(behavior_type="unitary"),
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/B1_unitary_5.json",
            )
        )

        # B2: classical_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/B2_classical_5.log", level=logging.DEBUG
                    ),
                    experiment_name="B2_classical_5",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/B2_classical_5.json",
            )
        )

        # B4: unitary_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/B4_unitary_10.log", level=logging.DEBUG
                    ),
                    experiment_name="B4_unitary_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=UpdaterBehaviorType(behavior_type="unitary"),
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/B4_unitary_10.json",
            )
        )

        # B5: classical_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/B5_classical_10.log", level=logging.DEBUG
                    ),
                    experiment_name="B5_classical_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/B5_classical_10.json",
            )
        )

        # === CATEGORY C: Feature Ablation (C1-C7) ===
        # C1: no_bdi_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/C1_no_bdi_5.log", level=logging.DEBUG
                    ),
                    experiment_name="C1_no_bdi_5",
                    memory_manager_config=BDIForgettingOnlyManagerType(
                        manager_type="forgetting_only", memory_removal_prob=0.5
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C1_no_bdi_5.json",
            )
        )

        # C2: no_forget_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/C2_no_forget_5.log", level=logging.DEBUG
                    ),
                    experiment_name="C2_no_forget_5",
                    memory_manager_config=BDIPLanningOnlyManagerType(
                        manager_type="bdi_planning_only", memory_removal_prob=0.5
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C2_no_forget_5.json",
            )
        )

        # C3: simple_mem_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/C3_simple_mem_5.log", level=logging.DEBUG
                    ),
                    experiment_name="C3_simple_mem_5",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=simple_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C3_simple_mem_5.json",
            )
        )

        # C4: no_bdi_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/C4_no_bdi_10.log", level=logging.DEBUG
                    ),
                    experiment_name="C4_no_bdi_10",
                    memory_manager_config=BDIForgettingOnlyManagerType(
                        manager_type="forgetting_only", memory_removal_prob=0.5
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C4_no_bdi_10.json",
            )
        )

        # C5: no_forget_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/C5_no_forget_10.log", level=logging.DEBUG
                    ),
                    experiment_name="C5_no_forget_10",
                    memory_manager_config=BDIPLanningOnlyManagerType(
                        manager_type="bdi_planning_only", memory_removal_prob=0.5
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C5_no_forget_10.json",
            )
        )

        # C6: simple_mem_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/C6_simple_mem_10.log", level=logging.DEBUG
                    ),
                    experiment_name="C6_simple_mem_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=simple_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/C6_simple_mem_10.json",
            )
        )

        # === CATEGORY D: Retrieval Strategy Sweep (D1-D20) ===
        # D1-D4: top_k 5 agents
        for val, idx in [(5, 1), (10, 2), (25, 3), (50, 4)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_topk_{val}_5.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_topk_{val}_5",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="top_k", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_topk_{val}_5.json",
                )
            )

        # D5-D7: mean_std 5 agents
        for val, idx in [(0.5, 5), (1.0, 6), (2.0, 7)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_meanstd_{val}_5.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_meanstd_{val}_5",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="mean_std", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_meanstd_{val}_5.json",
                )
            )

        # D8-D10: top_std 5 agents
        for val, idx in [(0.5, 8), (1.0, 9), (2.0, 10)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_topstd_{val}_5.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_topstd_{val}_5",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="top_std", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_topstd_{val}_5.json",
                )
            )

        # D11-D14: top_k 10 agents
        for val, idx in [(5, 11), (10, 12), (25, 13), (50, 14)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_topk_{val}_10.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_topk_{val}_10",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="top_k", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_topk_{val}_10.json",
                )
            )

        # D15-D17: mean_std 10 agents
        for val, idx in [(0.5, 15), (1.0, 16), (2.0, 17)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_meanstd_{val}_10.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_meanstd_{val}_10",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="mean_std", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_meanstd_{val}_10.json",
                )
            )

        # D18-D20: top_std 10 agents
        for val, idx in [(0.5, 18), (1.0, 19), (2.0, 20)]:
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            f"./logs/D{idx}_topstd_{val}_10.log", level=logging.DEBUG
                        ),
                        experiment_name=f"D{idx}_topstd_{val}_10",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="top_std", value=val
                        ),
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/D{idx}_topstd_{val}_10.json",
                )
            )

        # === CATEGORY E: Epoch × Utterance Sweep (E1-E10) ===
        epoch_utterance_combos = [
            (5, 4, 1),  # quarter
            (10, 8, 2),  # half
            (20, 16, 3),  # baseline
            (30, 16, 4),  # 1.5x epochs
            (40, 32, 5),  # double
        ]
        for epochs_val, utt_val, idx in epoch_utterance_combos:
            # 5 agents
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset5,
                        logger=get_xml_file_logger(
                            f"./logs/E{idx}_{epochs_val}e_{utt_val}u_5.log",
                            level=logging.DEBUG,
                        ),
                        experiment_name=f"E{idx}_{epochs_val}e_{utt_val}u_5",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=baseline_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=utt_val,
                        epochs=epochs_val,
                    ),
                    f"./results/E{idx}_{epochs_val}e_{utt_val}u_5.json",
                )
            )
            # 10 agents
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            f"./logs/E{5+idx}_{epochs_val}e_{utt_val}u_10.log",
                            level=logging.DEBUG,
                        ),
                        experiment_name=f"E{5+idx}_{epochs_val}e_{utt_val}u_10",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=baseline_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=utt_val,
                        epochs=epochs_val,
                    ),
                    f"./results/E{5+idx}_{epochs_val}e_{utt_val}u_10.json",
                )
            )

        # === CATEGORY F: Selector Comparison (F1-F2) ===
        # F1: full_parallel_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/F1_fp_10.log", level=logging.DEBUG
                    ),
                    experiment_name="F1_fp_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="full_parallel",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/F1_fp_10.json",
            )
        )

        # F2: information_spread_10 (baseline selector for comparison)
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/F2_is_10.log", level=logging.DEBUG
                    ),
                    experiment_name="F2_is_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/F2_is_10.json",
            )
        )

        # === CATEGORY G: Forgetting Sweep (G1-G2) ===
        # G1: full_forgetting_5
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset5,
                    logger=get_xml_file_logger(
                        "./logs/G1_full_forgetting_5.log", level=logging.DEBUG
                    ),
                    experiment_name="G1_full_forgetting_5",
                    memory_manager_config=BDIMemoryManagerType(
                        manager_type="bdi", memory_removal_prob=1.0
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/G1_full_forgetting_5.json",
            )
        )

        # G2: full_forgetting_10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/G2_full_forgetting_10.log", level=logging.DEBUG
                    ),
                    experiment_name="G2_full_forgetting_10",
                    memory_manager_config=BDIMemoryManagerType(
                        manager_type="bdi", memory_removal_prob=1.0
                    ),
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=baseline_updater,
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/G2_full_forgetting_10.json",
            )
        )

    print("\n=== All experiments completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synthetic information spread experiments"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="Number of parallel experiments (default: 1)",
    )
    parser.add_argument(
        "--use_local_embeddings",
        action="store_true",
        help="Whether to use local sentence transformer embeddings instead of OpenAI embeddings",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=32,
        help="Batch size for local embedding generation (default: 128)",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    asyncio.run(
        main(
            concurrency=args.concurrency,
            embedding_batch_size=args.embedding_batch_size,
            use_local_embeddings=args.use_local_embeddings,
        )
    )
