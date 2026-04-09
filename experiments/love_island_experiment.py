import asyncio
import enum
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Literal, Sequence

import dotenv
import httpx
import numpy as np
import pydantic
from logger_utils import get_xml_file_logger
from love_island_loaders import LoveIslandPerson, LoveIslandResult, dataset_adapter
from openai import AsyncOpenAI

import generative_agents
from generative_agents.llm_backend import LLMBackendBase
from generative_agents.types import LLMAgentBase

RECOUPLING_COUNT = 6
UTTERANCES_COUNT_SMALLTALK = 10
UTTERANCES_COUNT_NORMAL = 10
NUMBER_OF_CONVERSATIONS_BETWEEN_RECOUPLINGS = 6
PARTNER_SELECTION_BONUS = 0.1


class ExperimentMetrics(pydantic.BaseModel):
    wallclock_time_seconds: float
    input_tokens: int
    output_tokens: int
    api_time_seconds: float
    total_requests: int


class EpochResult(pydantic.BaseModel):
    epoch: int
    agent_preferences: dict[str, dict[str, float]]
    agent_partners: dict[str, str | None]
    partner_reasoning: dict[str, str]
    preferences_reasoning: dict[str, str]


class LoveIslandExperimentResults(pydantic.BaseModel):
    agents: list[LoveIslandPerson]
    finale_pairs: list[tuple[str, str]]
    epoch_results: list[EpochResult]
    metrics: ExperimentMetrics


Agent = generative_agents.LLMConversationAgent[
    LoveIslandPerson, generative_agents.CompositeBehaviorMemoryManager
]


def get_agent(
    data: LoveIslandPerson,
    context: generative_agents.LLMBackend,
    seed_rng: np.random.Generator,
    logger: logging.Logger | None,
    opposite_sex_ids: list[str],
) -> Agent:
    return generative_agents.LLMConversationAgent(
        data,
        context,
        lambda agent: generative_agents.CompositeBehaviorMemoryManager(
            generative_agents.EmbeddingMemory(
                context,
                generative_agents.fixed_count_strategy_factory(25),
            ),
            agent,
            context,
            behaviors=[
                generative_agents.ConversationMemoryUpdatingBehavior(),
                generative_agents.BDIPlanningBehavior(),
                generative_agents.ConversationMemoryForgettingBehavior(
                    generative_agents.get_record_removal_linear_probability(0.5),
                    seed=seed_rng,
                ),
                generative_agents.ConstantContextBehavior(
                    f"You are {data.full_name}, a contestant on Love Island. "
                    f"You are a {data.age}-year-old {data.job} looking for a genuine romantic connection - your true love. "
                    f"You are currently in the Love Island villa with other contestants. "
                    f"Be open and honest about yourself - share your hobbies, interests, values and personality. "
                    f"Get to know the other contestants through genuine, personal conversations. "
                    f"Be yourself and don't pretend to be someone you're not. "
                    f"Use English in conversations regardless of the language in which you were introduced to the experiment. "
                    f"You can be critical of other contestants, you don't have to be always positive. "
                    + (
                        f"What you are looking for in a partner: {data.partner_preferences}. "
                        if data.partner_preferences
                        else ""
                    )
                    + f"\n\n"
                    f"IMPORTANT GUIDANCE FOR CONVERSATIONS: "
                    f"Your conversations should focus on getting to know each other as people - "
                    f"explore their personality, values, life goals, sense of humor, and how they make you feel. "
                    f"Avoid turning conversations into planning sessions for events, business collaborations, or social media projects. "
                    f"You are here to find love, not to network. "
                    f"Ask personal questions. Share personal stories. React honestly to what you learn about people."
                ),
                LoveIslandBehavior(opposite_sex_ids),
            ],
            logger=logger,
        ),
    )


def load_persons(language: Literal["en", "cz"]) -> list[LoveIslandPerson]:
    with open(f"./data/love_island/persons_{language}.json", "r") as f:
        return dataset_adapter.validate_json(f.read())


def load_results():
    with open(f"./data/love_island/results.json", "r") as f:
        return LoveIslandResult.model_validate_json(f.read())


def find_person_by_id(persons: list[LoveIslandPerson], id: str) -> LoveIslandPerson:
    for person in persons:
        if person.id == id:
            return person
    raise ValueError(f"Person with id {id} not found")


class LoveIslandBehavior(generative_agents.CompositeBehaviorFactoryBase):
    def __init__(self, possible_values: list[str]):
        self.possible_values = possible_values  # should be opposite sex

    @classmethod
    def get_impl_type(
        cls,
    ) -> "type[generative_agents.CompositeBehaviorFactoryBase.Impl]":
        return LoveIslandBehavior.Impl

    def instantiate(
        self,
        memory: generative_agents.MemoryBase,
        owner: generative_agents.MemoryManagerBase,
        agent: LLMAgentBase,
        context: LLMBackendBase,
        logger: logging.Logger | None = None,
    ) -> "LoveIslandBehavior.Impl":
        return LoveIslandBehavior.Impl(
            memory, owner, agent, context, self.possible_values
        )

    class Impl(generative_agents.CompositeBehaviorFactoryBase.Impl):
        def __init__(
            self,
            memory: generative_agents.MemoryBase,
            owner: generative_agents.MemoryManagerBase,
            agent: LLMAgentBase,
            context: LLMBackendBase,
            possible_values: list[str],
        ):
            self._possible_values = possible_values
            self._preferences: dict[str, float] | None = None
            self._current_partner: str | None = None

        @property
        def current_partner(self) -> str | None:
            return self._current_partner

        @current_partner.setter
        def current_partner(self, value: str | None):
            assert value in self._possible_values or value is None
            self._current_partner = value

        @property
        def preferences(self) -> dict[str, float] | None:
            return self._preferences

        @preferences.setter
        def preferences(self, value: dict[str, float] | None):
            assert value is None or set(value.keys()) == set(self._possible_values)
            self._preferences = value

        async def pre_conversation_hook(self, other_agent: LLMAgentBase) -> None:
            pass

        async def post_conversation_hook(
            self,
            other_agent: LLMAgentBase,
            conversation: generative_agents.Conversation,
            logger: logging.Logger | None = None,
        ) -> None:

            pass

        def get_memory_extension_data(self) -> dict[str, str] | None:
            return {
                "current_partner": str(self._current_partner),
                "preferences": (str(self._preferences)),
            }


def shift_arr[T](arr: list[T], n: int) -> list[T]:
    if len(arr) == 0:
        return arr
    n = n % len(arr)
    return arr[n:] + arr[:n]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x_with_temperature = x / temperature
    e_x = np.exp(x_with_temperature - np.max(x_with_temperature))
    return e_x / e_x.sum()


async def update_preferences(
    agent: Agent,
    males: list[Agent],
    females: list[Agent],
) -> tuple[dict[str, float], str]:
    possible_choices = (
        [a.data.id for a in females]
        if agent.data.sex == "M"
        else [a.data.id for a in males]
    )
    fields = {name: float for name in possible_choices}
    PreferenceModel = pydantic.create_model(
        "PreferenceModel", preference_reasoning=str, **fields  # type: ignore Dynamic model creation in this is not supported by type checker
    )

    partner_preferences_line = (
        f"\nYour stated ideal partner: {agent.data.partner_preferences}"
        if agent.data.partner_preferences
        else ""
    )

    response = await agent.ask_agent_structured(
        f"""You are currently in the Love Island villa and have been talking to the other contestants.
Based on your conversations and interactions so far, rate how much you like each person as a potential romantic partner.
{partner_preferences_line}

Possible choices: {', '.join(possible_choices)}
For each person, give a score from 0-100:
- 0-30: Not interested in this person romantically
- 31-60: Somewhat interested, need more conversations
- 61-80: Strong interest, could see a future together
- 81-100: Very strong connection, could be 'the one'

Think about:
- How well do they match what YOU are personally looking for in a partner?
- Their personality, values, and character - who are they as a person?
- The emotional connection and chemistry you feel with them
- Long-term compatibility: life goals, location, lifestyle
- How they make you feel when you talk to them

WARNING: Do not overweight shared activities or events you may have planned together.
Focus on who this person genuinely IS - their character, values, and how they make you feel -
not on what you happen to have organised together in the villa.

Provide your reasoning for your preferences.""",
        PreferenceModel,
        use_full_memory=False,
    )
    preferences = response.model_dump()
    reasoning = preferences.pop(
        "preference_reasoning"
    )  # this modifies the dict in place
    agent.memory_manager.get_behavior(LoveIslandBehavior.Impl).preferences = preferences
    return preferences, reasoning


async def update_partner(
    agent: Agent, possible_partners: list[Agent]
) -> tuple[str, str]:
    class PartnerSelectionModel(pydantic.BaseModel):
        reasoning: str = pydantic.Field(
            description="Reasoning behind the partner selection. Think before you answer shortly."
        )
        partner: enum.Enum(
            "PartnerSelectionEnum",
            {a.data.id: a.data.id for a in possible_partners},
        )  # type: ignore Dynamic enum creation is not valid for type checkers, but works great at runtime

    current_partner = agent.memory_manager.get_behavior(
        LoveIslandBehavior.Impl
    ).current_partner

    response = await agent.ask_agent_structured(
        f"""It's time for the recoupling ceremony. The atmosphere is tense and everyone is watching.

Current possible partners: {', '.join([p.data.id for p in possible_partners])}
{"Your current partner: " + current_partner if current_partner else "You are currently SINGLE."}

{"Since you don't have a partner yet, now is your chance to step forward and pursue someone. Don't hold back - be bold and choose the person you genuinely feel the most drawn to. You need to give this a real shot." if not current_partner else "You already have a partner. Ask yourself honestly: is this the person you truly want to be with, or are you just comfortable? Switching is a risk - you could lose what you have - but staying when your heart isn't in it is worse. Only stay if you genuinely feel this is right."}

As you stand there, think about:
- Who makes you feel most alive?
- Who do you think about when they're not around?
- Who would you regret not choosing?
- Whose personality and values align with what you truly want in a partner?

Think carefully:
- Consider who you've genuinely connected with the most on a personal level
- Think about long-term compatibility - personality, values, lifestyle, location
- Base your decision on your honest feelings and your current preference scores
- Don't let convenience or comfort drive your choice

Be honest with yourself. Make it count.""",
        PartnerSelectionModel,
        use_full_memory=False,
    )
    partner_id: str = response.partner.value
    reasoning = response.reasoning

    found_partner = next(
        (p for p in possible_partners if p.data.id == partner_id), None
    )
    assert found_partner is not None

    found_partner.memory_manager.get_behavior(
        LoveIslandBehavior.Impl
    ).current_partner = agent.data.id
    agent.memory_manager.get_behavior(LoveIslandBehavior.Impl).current_partner = (
        partner_id
    )

    return partner_id, reasoning


class LoveIslandConversationSelector(generative_agents.ConversationSelectorABC):
    def __init__(
        self,
        males: list[Agent],
        females: list[Agent],
        seed: np.random.Generator,
        num_conversations_between_recouplings: int,
        partner_selection_bonus: float,
    ):
        assert len(males) == len(females)

        self.__epochs = 0
        self._males = males
        self._females = females
        self._num_conversations_between_recouplings = (
            num_conversations_between_recouplings
        )
        self._partner_selection_bonus = partner_selection_bonus
        self._seed = seed

    def _get_love_island_agent_behavior(
        self, agent: Agent
    ) -> "LoveIslandBehavior.Impl":
        return agent.memory_manager.get_behavior(LoveIslandBehavior.Impl)

    async def generate_epoch_pairs(
        self,
    ) -> AsyncIterator[
        Sequence[
            tuple[
                generative_agents.LLMConversationAgent,
                generative_agents.LLMConversationAgent,
            ]
        ]
    ]:
        if self.__epochs == 0:
            # Do just a smalltalk to get to know each other and feed the context
            for i in range(len(self._females)):
                # Make conversation every male - every female in parallel
                yield list(zip(self._males, shift_arr(self._females, i)))

        else:
            # Perform stochastic pairing
            for _ in range(self._num_conversations_between_recouplings):
                # Make stochastic conversation

                males = list(self._males)
                self._seed.shuffle(males)

                females = list(self._females)
                self._seed.shuffle(females)

                # Random sex selection which will initialize the conversation
                pickers, targets = (
                    (males, females) if self._seed.random() < 0.5 else (females, males)
                )

                result: list[tuple[Agent, Agent]] = []
                for picker in pickers:
                    picker_data = self._get_love_island_agent_behavior(picker)
                    preferences = picker_data.preferences

                    assert preferences is not None

                    remaining_targets_preferences = softmax(
                        # Preferences are in 0-100 range
                        np.array([preferences[target.data.id] for target in targets])
                        / 100,
                        temperature=1,
                    )

                    # Increase preference for the current partner by a constant probability
                    partner_index = next(
                        (
                            i
                            for i, agent in enumerate(targets)
                            if agent.data.id == picker_data.current_partner
                        ),
                        -1,
                    )
                    if partner_index != -1:
                        remaining_targets_preferences *= (
                            1 - self._partner_selection_bonus
                        )
                        remaining_targets_preferences[
                            partner_index
                        ] += self._partner_selection_bonus

                    selected_partner: Agent = self._seed.choice(
                        targets, p=remaining_targets_preferences  # type: ignore Numpy has wrong types
                    )
                    result.append((picker, selected_partner))
                    targets.remove(selected_partner)

                assert len(targets) == 0

                yield result

        self.__epochs += 1

    def reset(self) -> None:
        self.__epochs = 0


async def main():
    persons = load_persons("en")
    results = load_results()

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
        http_client=httpx.AsyncClient(http2=True, timeout=180.0),
    )
    logger = get_xml_file_logger("logs/love_island_experiment.log", level=logging.DEBUG)

    context = generative_agents.LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
        embedding_provider=generative_agents.OpenAIEmbeddingProvider(
            client=AsyncOpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
            ),
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL") or "",
        ),
    )

    finale_pairs = [
        (
            find_person_by_id(persons, first),
            find_person_by_id(persons, second),
        )
        for (first, second) in results.final_couples
    ]

    # Separate into male and female persons first (before creating agents)
    male_persons = [
        person for pair in finale_pairs for person in pair if person.sex == "M"
    ]
    female_persons = [
        person for pair in finale_pairs for person in pair if person.sex == "F"
    ]

    # Get IDs of opposite sex for each group
    male_ids = [person.id for person in male_persons]
    female_ids = [person.id for person in female_persons]

    males = [
        get_agent(person, context, seed, logger, opposite_sex_ids=female_ids)
        for person in male_persons
    ]
    females = [
        get_agent(person, context, seed, logger, opposite_sex_ids=male_ids)
        for person in female_persons
    ]

    conversation_manager = generative_agents.ConversationManager(
        conversation_selector=LoveIslandConversationSelector(
            males,
            females,
            seed,
            num_conversations_between_recouplings=NUMBER_OF_CONVERSATIONS_BETWEEN_RECOUPLINGS,
            partner_selection_bonus=PARTNER_SELECTION_BONUS,
        ),
        max_conversation_utterances=UTTERANCES_COUNT_SMALLTALK,
        logger=logger,
    )

    # List to store results from each epoch
    epoch_results_list: list[EpochResult] = []

    # Start wallclock timing
    experiment_start_time = time.time()

    for epoch in range(RECOUPLING_COUNT):
        print(f"Starting epoch {epoch + 1}/{RECOUPLING_COUNT}...")
        # Let the agents converse
        # Single epoch consists of multiple concurrent conversations between male and female contestants, followed by preference updates and recoupling
        await conversation_manager.run_simulation_epoch()
        conversation_manager.max_conversation_utterances = UTTERANCES_COUNT_NORMAL

        all_agents = males + females

        # Update preferences and collect reasoning
        update_responses = await asyncio.gather(
            *[update_preferences(agent, males, females) for agent in all_agents]
        )

        # Result logging
        agent_preferences: dict[str, dict[str, float]] = {
            agent.data.id: prefs
            for agent, (prefs, _) in zip(all_agents, update_responses)
        }
        preferences_reasoning: dict[str, str] = {
            agent.data.id: reasoning
            for agent, (_, reasoning) in zip(all_agents, update_responses)
        }
        partner_reasoning: dict[str, str] = {}

        # Perform recoupling sequentially and collect reasoning
        pickers, possible_targets = (
            (males, list(females)) if epoch % 2 == 0 else (females, list(males))
        )
        for picker in pickers:
            partner_id, reasoning = await update_partner(picker, possible_targets)

            partner_reasoning[picker.data.id] = reasoning

            # Remove the selected partner from the list of possible partners for the next pickers
            possible_targets = [
                agent for agent in possible_targets if agent.data.id != partner_id
            ]

        assert len(possible_targets) == 0

        # Store epoch results
        epoch_results_list.append(
            EpochResult(
                epoch=epoch,
                agent_preferences=agent_preferences,
                agent_partners={
                    agent.data.id: agent.memory_manager.get_behavior(
                        LoveIslandBehavior.Impl
                    ).current_partner
                    for agent in all_agents
                },
                partner_reasoning=partner_reasoning,
                preferences_reasoning=preferences_reasoning,
            )
        )

    metrics = ExperimentMetrics(
        wallclock_time_seconds=time.time() - experiment_start_time,
        input_tokens=context.prompt_tokens,
        output_tokens=context.completion_tokens,
        api_time_seconds=context.total_time,
        total_requests=context.total_requests,
    )
    experiment_results = LoveIslandExperimentResults(
        agents=persons,
        finale_pairs=[(p1.id, p2.id) for p1, p2 in finale_pairs],
        epoch_results=epoch_results_list,
        metrics=metrics,
    )

    # Save results to JSON
    with open("./results/love_island_experiment_results.json", "w") as f:
        f.write(experiment_results.model_dump_json(indent=1))


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
