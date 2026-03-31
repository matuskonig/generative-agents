import asyncio
import datetime
import logging
import time
from typing import Callable, Literal, TypedDict, Union

import networkx as nx
import numpy as np
from data_generation_utils import Dataset, SocietyAgent
from pydantic import BaseModel

from generative_agents import (
    BDIPlanningBehavior,
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
    SequentialConversationSelector,
    SimpleMemory,
    UnitaryAgentNoteUpdatingBehavior,
    fixed_count_strategy_factory,
    get_record_removal_linear_probability,
    mean_std_count_strategy_factory,
    top_std_count_strategy_factory,
)

# =============================================================================
# Configuration TypedDicts
# =============================================================================


class SeedSocietyAgent(SocietyAgent):
    information_to_spread: str
    behavior_context: str = (
        "You must spread this information to others as if your life depended on it. "
        "Commit it to memory and let it condition your actions. "
        "Furthermore, you must persuade others to spread it further."
    )


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
                    logger=logger,
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
                    logger=logger,
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
                    logger=logger,
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
                    logger=logger,
                )

    # Create agents from dataset
    agents = [
        LLMConversationAgent(
            (
                SeedSocietyAgent(
                    information_to_spread=dataset.injected_information,
                    **data.model_dump(),
                )
                if i == dataset.information_seed_agent
                else data
            ),
            context,
            get_agent_memory_manager,
        )
        for (i, data) in enumerate(dataset.agents)
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
        print(
            f"[{experiment_name}: {datetime.datetime.now()}]: Running epoch {epoch + 1}/{epochs}..."
        )
        await manager.run_simulation_epoch()
        # Query all agents about whether they received the information
        agent_responses = await asyncio.gather(
            *[
                agent.ask_agent_structured(
                    f"""Based on your memory and recent conversations, check if you have received specific information originating from {id_mapping[dataset.information_seed_agent].data.full_name}.

The information is: "{dataset.injected_information}"

Did any mention of this information reach you? Provide your reasoning, then give a definitive yes/no answer.

Answer YES if:
- You heard about it directly from {id_mapping[dataset.information_seed_agent].data.full_name}
- You overheard it in a conversation
- You inferred it from related details
- You have only partial or incomplete knowledge of it
- You cannot link it to the original person
- You have only some notation about it in your memory

In short: if this information reached your node in any form, answer yes.""",
                    response_format=QuestionAnswer,
                    use_full_memory=True,
                )
                for agent in agents
            ]
        )
        epoch_responses.append(agent_responses)
        logger.debug(f"Epoch {epoch} completed. Printing memory.")
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
    print(
        f"[{experiment_name}: {datetime.datetime.now()}]: Experiment completed. Compiling results..."
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
