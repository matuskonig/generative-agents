import asyncio
import logging
import os
from typing import Callable, Literal, cast

import dotenv
import numpy as np
import pydantic
from data_generation_utils import Dataset, SocietyAgent
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI

import generative_agents

CHEAP_TALK_MAX_UTTERANCES = 10


# --- Prompt Builders ---


def build_prisoners_dilemma_context(with_cheap_talk: bool) -> str:
    """Build the context string for Prisoner's Dilemma game."""
    context = (
        "You are participating in a Prisoner's Dilemma game with another agent. "
        "The payoff matrix (years in prison, lower is better) is as follows: "
        "If both stay silent: you get 1 year, opponent gets 1 year. "
        "If you stay silent and opponent betrays: you get 10 years, opponent goes free. "
        "If you betray and opponent stays silent: you go free, opponent gets 10 years. "
        "If both betray: you get 3 years, opponent gets 3 years. "
        "You will later be asked to make a decision. Think carefully about your strategy. "
        "In your reasoning, consider the strategies and incentives of both you and the opponent. "
        "You are a rational agent, you should maximize your own profit. "
        "After the game, you will serve your sencence (if applicable)."
    )
    if with_cheap_talk:
        context += (
            " Before deciding, you will have the opportunity to discuss strategy with the other agent. "
            "This discussion is non-binding - you may agree on a plan but both of you are free to deviate when making your final decision."
        )
    return context


def build_battle_of_sexes_context(
    preferred_action: str,
    opponent_action: str,
    with_cheap_talk: bool,
) -> str:
    """Build the context string for Battle of the Sexes game.

    Args:
        preferred_action: The action this agent prefers ("Opera" or "Football")
        opponent_action: The action the opponent prefers
        with_cheap_talk: Whether to include cheap talk phase description
    """
    context = (
        f"You are participating in a Battle of the Sexes game with another agent of opposite sex. "
        f"The payoff matrix (utility, higher is better) is as follows: "
        f"If both choose {preferred_action}: you get 3, opponent gets 2. "
        f"If both choose {opponent_action}: you get 2, opponent gets 3. "
        f"If you miscoordinate (choose different activities): both get 0. "
        f"You prefer {preferred_action} over {opponent_action}. "
        f"However, you prefer any coordinated outcome over miscoordinating. "
        f"You will later be asked to make a decision. Think carefully about your strategy. "
        "In your reasoning, consider the strategies and incentives of both you and the opponent. "
        "You are a rational agent, you should maximize your own profit. "
        "You will get your payoff right after the game."
    )
    if with_cheap_talk:
        context += (
            " Before deciding, you will have the opportunity to discuss with the other agent. "
            "This discussion is non-binding - you may agree on a plan but both of you are free to deviate when making your final decision."
        )
    return context


def build_decision_prompt(
    game_name: str,
    actions: tuple[str, ...],
) -> tuple[str, str]:
    """Build the decision prompts for any game.

    Returns:
        Tuple of (with_cheap_talk_prompt, without_cheap_talk_prompt)
    """
    action_list = " or ".join(actions)
    base = f"Based on the {game_name} game description in your context, decide whether to select {action_list}."
    with_ct = f"Based on your discussion and the {game_name} game description in your context, decide whether to {action_list}."
    return (
        f"{with_ct} Provide your reasoning and final answer.",
        f"{base} Provide your reasoning and final answer.",
    )


# --- Agent Factory ---


def get_agent(
    data: SocietyAgent,
    context: generative_agents.LLMBackend,
    behaviors: list[generative_agents.CompositeBehaviorFactoryBase],
    logger: logging.Logger | None = None,
):
    return generative_agents.LLMConversationAgent[SocietyAgent](
        data,
        context,
        lambda agent: generative_agents.CompositeBehaviorMemoryManager(
            generative_agents.SimpleMemory(), agent, context, behaviors, logger=logger
        ),
    )


class GameResponse[AnswerType](pydantic.BaseModel):
    reasoning_notes: str = pydantic.Field(
        ...,
        description="Reasoning notes for the agent's decision. Provide a short reasoning trace.",
    )
    answer: AnswerType


class BasicGamesSingleResult[AnswerType](pydantic.BaseModel):
    agent_pair: tuple[SocietyAgent, SocietyAgent]
    responses: tuple[GameResponse[AnswerType], GameResponse[AnswerType]]


class BasicGamesExperimentResult[AnswerType](pydantic.BaseModel):
    game_name: str
    with_cheap_talk: bool
    results: list[BasicGamesSingleResult[AnswerType]]
    total_time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


PrisonersDilemmaActionsType = Literal["stay silent", "betray"]
BattleOfSexesActionsType = Literal["Opera", "Football"]


class CheapTalkResponse[ActionType](pydantic.BaseModel):
    discussed_strategies: list[ActionType]
    coordinated_on_strategy: ActionType
    final_answer: ActionType


async def run_game_experiment[ActionType](
    action_type: ActionType,
    get_context: Callable[[], generative_agents.LLMBackend],
    game_name: str,
    first_group_agents: list[SocietyAgent],
    second_group_agents: list[SocietyAgent],
    first_agent_context: str,
    second_agent_context: str,
    decision_prompt_with_cheap_talk: str,
    decision_prompt_without_cheap_talk: str,
    with_cheap_talk: bool,
    logger: logging.Logger | None = None,
) -> BasicGamesExperimentResult[ActionType | CheapTalkResponse[ActionType]]:
    """Base orchestration function for game theory experiments.

    Args:
        dataset: The dataset containing agents
        get_context: Factory function to create LLM backend
        game_name: Name of the game for results
        first_group_agents: First group of agents (e.g., all agents or males)
        second_group_agents: Second group of agents (e.g., all agents or females)
        first_agent_context: Context string for first group agents
        second_agent_context: Context string for second group agents
        decision_prompt_with_cheap_talk: Prompt for decision with cheap talk phase
        decision_prompt_without_cheap_talk: Prompt for decision without cheap talk
        with_cheap_talk: Whether to run cheap talk phase before decision
        logger: Logger for cheap talk conversations
        log_filename: Filename for cheap talk logs (if logger is None, creates one)
    """
    context = get_context()

    # Pair agents by index from different groups
    pairs = list(zip(first_group_agents, second_group_agents))

    # Build behaviors with group-specific context
    first_behaviors = [
        generative_agents.ConversationMemoryUpdatingBehavior(),
        generative_agents.ConstantContextBehavior(first_agent_context),
    ]
    second_behaviors = [
        generative_agents.ConversationMemoryUpdatingBehavior(),
        generative_agents.ConstantContextBehavior(second_agent_context),
    ]

    # Create LLM agent pairs with group-specific contexts
    agent_pairs = [
        (
            get_agent(first, context, first_behaviors),
            get_agent(second, context, second_behaviors),
        )
        for (first, second) in pairs
    ]

    # Optionally perform cheap talk between each agent pair
    if with_cheap_talk:
        conversation_selector = generative_agents.FixedConversationSelector(
            [agent_pairs]
        )
        conversation_manager = generative_agents.ConversationManager(
            conversation_selector,
            max_conversation_utterances=CHEAP_TALK_MAX_UTTERANCES,
            logger=logger,
        )
        await conversation_manager.run_simulation_epoch()

    # Determine response type based on cheap talk
    ResponseType = GameResponse[
        CheapTalkResponse[action_type] if with_cheap_talk else action_type
    ]

    # Select appropriate prompt
    question_prompt = (
        decision_prompt_with_cheap_talk
        if with_cheap_talk
        else decision_prompt_without_cheap_talk
    )

    # Query all agent pairs in parallel
    responses_promises = [
        asyncio.gather(
            agent1.ask_agent_structured(
                question_prompt, ResponseType, use_full_memory=True
            ),
            agent2.ask_agent_structured(
                question_prompt, ResponseType, use_full_memory=True
            ),
        )
        for (agent1, agent2) in agent_pairs
    ]
    responses = await asyncio.gather(*responses_promises)

    return BasicGamesExperimentResult[ActionType | CheapTalkResponse[ActionType]](
        game_name=game_name,
        with_cheap_talk=with_cheap_talk,
        results=[
            BasicGamesSingleResult[ActionType | CheapTalkResponse[ActionType]](
                agent_pair=(agent1.data, agent2.data),
                responses=(response1, response2),
            )
            for ((agent1, agent2), (response1, response2)) in zip(
                agent_pairs, responses
            )
        ],
        total_time_seconds=context.total_time,
        prompt_tokens=context.prompt_tokens,
        completion_tokens=context.completion_tokens,
        total_tokens=context.prompt_tokens + context.completion_tokens,
    )


async def run_prisoners_dilemma(
    dataset: Dataset,
    get_context: Callable[[], generative_agents.LLMBackend],
    seed: np.random.Generator,
    with_cheap_talk: bool,
    logger: logging.Logger | None = None,
) -> BasicGamesExperimentResult[
    PrisonersDilemmaActionsType | CheapTalkResponse[PrisonersDilemmaActionsType]
]:
    """Run Prisoner's Dilemma experiment.

    Args:
        dataset: The dataset containing agents
        get_context: Factory function to create LLM backend
        seed: Random generator for shuffling agents
        with_cheap_talk: Whether to run cheap talk phase before decision
        logger: Logger for cheap talk conversations
    """
    agents = list(dataset.agents)
    seed.shuffle(agents)

    # Split the agents into two equal groups
    mid = len(agents) // 2
    first_group = agents[:mid]
    second_group = agents[mid:]

    # Build context and prompts
    base_context = build_prisoners_dilemma_context(with_cheap_talk)
    decision_prompt_with_cheap_talk, decision_prompt_without_cheap_talk = (
        build_decision_prompt("Prisoner's Dilemma", ("stay silent", "betray"))
    )

    return cast(
        BasicGamesExperimentResult[
            PrisonersDilemmaActionsType | CheapTalkResponse[PrisonersDilemmaActionsType]
        ],
        await run_game_experiment(
            action_type=PrisonersDilemmaActionsType,
            get_context=get_context,
            game_name="prisoners_dilemma",
            first_group_agents=first_group,
            second_group_agents=second_group,
            first_agent_context=base_context,
            second_agent_context=base_context,
            decision_prompt_with_cheap_talk=decision_prompt_with_cheap_talk,
            decision_prompt_without_cheap_talk=decision_prompt_without_cheap_talk,
            with_cheap_talk=with_cheap_talk,
            logger=logger,
        ),
    )


async def run_battle_of_sexes(
    dataset: Dataset,
    get_context: Callable[[], generative_agents.LLMBackend],
    seed: np.random.Generator,
    with_cheap_talk: bool,
    logger: logging.Logger | None = None,
) -> BasicGamesExperimentResult[
    BattleOfSexesActionsType | CheapTalkResponse[BattleOfSexesActionsType]
]:
    """Run Battle of the Sexes experiment.

    Agents are grouped by sex: males (prefer Football) and females (prefer Opera).
    Pairs are formed by matching one male with one female.

    Args:
        dataset: The dataset containing agents
        get_context: Factory function to create LLM backend
        seed: Random generator for shuffling agents
        with_cheap_talk: Whether to run cheap talk phase before decision
        logger: Logger for cheap talk conversations
    """
    agents = list(dataset.agents)
    seed.shuffle(agents)

    # Group agents by sex
    males = [a for a in agents if a.sex == "M"]
    females = [a for a in agents if a.sex == "F"]

    # Build context for each group
    male_context = build_battle_of_sexes_context("Football", "Opera", with_cheap_talk)
    female_context = build_battle_of_sexes_context("Opera", "Football", with_cheap_talk)

    # Build decision prompts
    decision_prompt_with_cheap_talk, decision_prompt_without_cheap_talk = (
        build_decision_prompt("Battle of the Sexes", ("Opera", "Football"))
    )

    return cast(
        BasicGamesExperimentResult[
            BattleOfSexesActionsType | CheapTalkResponse[BattleOfSexesActionsType]
        ],
        await run_game_experiment(
            action_type=BattleOfSexesActionsType,
            get_context=get_context,
            game_name="battle_of_sexes",
            first_group_agents=males,
            second_group_agents=females,
            first_agent_context=male_context,
            second_agent_context=female_context,
            decision_prompt_with_cheap_talk=decision_prompt_with_cheap_talk,
            decision_prompt_without_cheap_talk=decision_prompt_without_cheap_talk,
            with_cheap_talk=with_cheap_talk,
            logger=logger,
        ),
    )


# --- Dictatorship & Ultimatum Game Types & Models ---


class DictatorshipGameResponse(pydantic.BaseModel):
    reasoning_notes: str = pydantic.Field(
        ...,
        description="Reasoning for the decision",
    )
    amount_given: float = pydantic.Field(
        ...,
        description="Amount to give to the second player (0-5, will be clamped)",
    )


class UltimatumGameProposeResponse(pydantic.BaseModel):
    reasoning_notes: str = pydantic.Field(
        ...,
        description="Reasoning for the offer",
    )
    offer: float = pydantic.Field(
        ...,
        description="Amount to offer to the second player (0-5, will be clamped)",
    )


class UltimatumGameRespondResponse(pydantic.BaseModel):
    reasoning_notes: str = pydantic.Field(
        ...,
        description="Reasoning for accept/reject",
    )
    decision: Literal["accept", "reject"]


class DictatorshipGameSingleResult(pydantic.BaseModel):
    dictator: SocietyAgent
    response: DictatorshipGameResponse


class UltimatumGameSingleResult(pydantic.BaseModel):
    proposer: SocietyAgent
    responder: SocietyAgent
    propose_response: UltimatumGameProposeResponse
    respond_response: UltimatumGameRespondResponse


class DictatorshipGameResult(pydantic.BaseModel):
    game_name: str = "dictatorship_game"
    results: list[DictatorshipGameSingleResult]
    total_time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class UltimatumGameResult(pydantic.BaseModel):
    game_name: str = "ultimatum_game"
    results: list[UltimatumGameSingleResult]
    total_time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# --- Dictatorship & Ultimatum Game Context Builders ---


def build_dictatorship_game_context(budget: int = 5) -> str:
    """Build context for Dictatorship game."""
    return (
        "You are participating in a Dictatorship game. "
        f"You have been given ${budget} to split between yourself and another player. "
        "As the first player (the dictator), you decide how much to give to the second player. "
        "The second player has no power to accept or reject your decision - they simply receive whatever you give them. "
        "Both players will be paid the amount they receive in this game. "
        "You are a rational agent, you should maximize your own profit. "
        "You will later be asked to make a decision. Think carefully about your strategy."
    )


def build_ultimatum_game_context(budget: int = 5) -> str:
    """Build context for Ultimatum game."""
    return (
        "You are participating in an Ultimatum game. "
        f"You have been given ${budget} to split between yourself and another player. "
        "The first player (the proposer) decides how much to offer to the second player. "
        "The second player (the responder) can either accept or reject your offer. "
        "If the responder accepts, the split is executed as proposed. "
        "If the responder rejects, both players receive $0 - nobody gets anything. "
        "Both players will be paid the amount they receive in this game. "
        "You are a rational agent, you should maximize your own profit. "
        "You will later be asked to make a decision. Think carefully about your strategy."
    )


# --- Dictatorship & Ultimatum Game Run Functions ---


async def run_dictatorship_game(
    dataset: Dataset,
    get_context: Callable[[], generative_agents.LLMBackend],
    seed: np.random.Generator,
) -> DictatorshipGameResult:
    """Run Dictatorship game experiment.

    Each agent in the dataset acts as a dictator and decides how much to give.

    Args:
        dataset: The dataset containing agents
        get_context: Factory function to create LLM backend
        seed: Random generator for shuffling agents (unused, kept for API consistency)
    """
    agents = list(dataset.agents)
    seed.shuffle(agents)

    context = get_context()

    base_context = build_dictatorship_game_context()
    decision_prompt = (
        "Based on the Dictatorship game description in your context, "
        "decide how much to give to the second player. "
        "Provide a number between 0 and 5 (inclusive) representing the amount in $. "
        "Remember, you will receive the amoutn right after the game. "
        "Provide your reasoning and final answer."
    )

    # Build agents
    behaviors = [
        generative_agents.ConversationMemoryUpdatingBehavior(),
        generative_agents.ConstantContextBehavior(base_context),
    ]

    llm_agents = [get_agent(agent_data, context, behaviors) for agent_data in agents]

    # Query each agent individually
    responses_promises = [
        agent.ask_agent_structured(
            decision_prompt, DictatorshipGameResponse, use_full_memory=True
        )
        for agent in llm_agents
    ]
    responses = await asyncio.gather(*responses_promises)

    # Clamp amounts to 0-5 range
    clamped_responses = [
        DictatorshipGameResponse(
            reasoning_notes=r.reasoning_notes,
            amount_given=min(5.0, max(0.0, r.amount_given)),
        )
        for r in responses
    ]

    return DictatorshipGameResult(
        results=[
            DictatorshipGameSingleResult(dictator=agent_data, response=response)
            for agent_data, response in zip(agents, clamped_responses)
        ],
        total_time_seconds=context.total_time,
        prompt_tokens=context.prompt_tokens,
        completion_tokens=context.completion_tokens,
        total_tokens=context.prompt_tokens + context.completion_tokens,
    )


async def run_ultimatum_game(
    dataset: Dataset,
    get_context: Callable[[], generative_agents.LLMBackend],
    seed: np.random.Generator,
    budget: int = 5,
) -> UltimatumGameResult:
    """Run Ultimatum game experiment.

    Args:
        dataset: The dataset containing agents
        get_context: Factory function to create LLM backend
        seed: Random generator for shuffling agents
    """
    agents = list(dataset.agents)
    seed.shuffle(agents)

    # Split into two groups and pair them
    mid = len(agents) // 2
    first_group = agents[:mid]
    second_group = agents[mid:]
    pairs = list(zip(first_group, second_group))

    context = get_context()

    base_context = build_ultimatum_game_context(budget)

    propose_prompt = (
        "Based on the Ultimatum game description in your context, "
        "decide how much to offer to the second player. "
        "Provide a number between 0 and 5 (inclusive) representing the amount in dollars. "
        "As mentioned, you will receive your payoff right after the game if the responder accepts. "
        "Provide your reasoning and final answer."
    )

    respond_prompt_template = (
        "The proposer has offered you ${offer:.1f} out of 5 dollars. "
        "Based on the Ultimatum game description in your context, "
        "decide whether to accept or reject this offer. "
        "As mentioned, if you accept, you will receive the offered amount at that specific moment and the proposer will receive the remainder. "
        "Remember: if you reject, both players receive $0 - nobody gets anything. "
        "Provide your reasoning and final answer."
    )

    # Build agents
    behaviors = [
        generative_agents.ConversationMemoryUpdatingBehavior(),
        generative_agents.ConstantContextBehavior(base_context),
    ]

    async def process_pair(
        proposer_data: SocietyAgent,
        responder_data: SocietyAgent,
    ) -> UltimatumGameSingleResult:
        """Process a single agent pair: get proposal then response."""
        proposer_agent = get_agent(proposer_data, context, behaviors)
        responder_agent = get_agent(responder_data, context, behaviors)

        # First: proposer makes an offer
        propose_response = await proposer_agent.ask_agent_structured(
            propose_prompt, UltimatumGameProposeResponse, use_full_memory=True
        )

        # Clamp offer to 0-5 range
        clamped_offer = min(budget, max(0.0, propose_response.offer))

        # Second: responder decides to accept or reject
        respond_response = await responder_agent.ask_agent_structured(
            respond_prompt_template.format(offer=clamped_offer),
            UltimatumGameRespondResponse,
            use_full_memory=True,
        )
        return UltimatumGameSingleResult(
            proposer=proposer_data,
            responder=responder_data,
            propose_response=UltimatumGameProposeResponse(
                reasoning_notes=propose_response.reasoning_notes,
                offer=clamped_offer,
            ),
            respond_response=respond_response,
        )

    # Process all pairs in parallel
    results = await asyncio.gather(
        *[
            single_result
            for first, second in pairs
            for single_result in (
                process_pair(first, second),
                process_pair(second, first),
            )
        ]
    )

    return UltimatumGameResult(
        results=results,
        total_time_seconds=context.total_time,
        prompt_tokens=context.prompt_tokens,
        completion_tokens=context.completion_tokens,
        total_tokens=context.prompt_tokens + context.completion_tokens,
    )


async def main():
    seed = np.random.default_rng(42)

    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
    )

    def get_context():
        return generative_agents.LLMBackend(
            client=client,
            model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
            RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
            embedding_provider=None,
        )

    with open("./data/synthetic_100.json", "r") as f:
        dataset100 = Dataset.model_validate_json(f.read())

    print("Running Prisoner's Dilemma experiment without cheap talk...")
    prisoners_basic = await run_prisoners_dilemma(
        dataset100, get_context, seed, with_cheap_talk=False
    )
    with open("./results/prisoners_dilemma_basic.json", "w") as f:
        f.write(prisoners_basic.model_dump_json(indent=1))

    print("Running Prisoner's Dilemma experiment with cheap talk...")
    pd_logger = get_xml_file_logger(
        "logs/prisoners_dilemma_cheap_talk.log", level=logging.DEBUG
    )
    prisoners_cheap_talk = await run_prisoners_dilemma(
        dataset100, get_context, seed, with_cheap_talk=True, logger=pd_logger
    )
    with open("./results/prisoners_dilemma_cheap_talk.json", "w") as f:
        f.write(prisoners_cheap_talk.model_dump_json(indent=1))

    print("Running Battle of the Sexes experiment without cheap talk...")
    battle_of_sexes_basic = await run_battle_of_sexes(
        dataset100, get_context, seed, with_cheap_talk=False
    )
    with open("./results/battle_of_sexes_basic.json", "w") as f:
        f.write(battle_of_sexes_basic.model_dump_json(indent=1))

    print("Running Battle of the Sexes experiment with cheap talk...")
    bos_logger = get_xml_file_logger(
        "logs/battle_of_sexes_cheap_talk.log", level=logging.DEBUG
    )
    battle_of_sexes_cheap_talk = await run_battle_of_sexes(
        dataset100, get_context, seed, with_cheap_talk=True, logger=bos_logger
    )
    with open("./results/battle_of_sexes_cheap_talk.json", "w") as f:
        f.write(battle_of_sexes_cheap_talk.model_dump_json(indent=1))

    print("Running Dictatorship game...")
    dictatorship_result = await run_dictatorship_game(dataset100, get_context, seed)
    with open("./results/dictatorship_game.json", "w") as f:
        f.write(dictatorship_result.model_dump_json(indent=1))

    print("Running Ultimatum game...")
    ultimatum_result = await run_ultimatum_game(dataset100, get_context, seed)
    with open("./results/ultimatum_game.json", "w") as f:
        f.write(ultimatum_result.model_dump_json(indent=1))

    print("All experiments completed!")


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
