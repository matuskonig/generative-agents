"""Information propagation experiment through agent chains."""

import asyncio
import logging
import os
from collections.abc import AsyncIterable
from typing import Sequence

import dotenv
import httpx
import numpy as np
import pydantic
from data_generation_utils import Dataset, SocietyAgent
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI

import generative_agents
from generative_agents import (
    CompositeBehaviorMemoryManager,
    ConstantContextBehavior,
    ConversationManager,
    ConversationMemoryUpdatingBehavior,
    LLMBackend,
    LLMConversationAgent,
    OpenAIEmbeddingProvider,
    SimpleMemory,
)
from generative_agents.conversation_managment import ConversationSelectorABC

Agent = LLMConversationAgent[SocietyAgent, CompositeBehaviorMemoryManager]


class MessageCheck(pydantic.BaseModel):
    has_knowledge_of_information: bool


class ChainConversationSelector(ConversationSelectorABC):
    def __init__(self, agent_chain: Sequence[Agent]) -> None:
        self.agent_chain = agent_chain
        self.current_index = 0

    async def generate_epoch_pairs(
        self,
    ) -> AsyncIterable[Sequence[tuple[Agent, Agent]]]:
        if self.current_index < len(self.agent_chain) - 1:
            pair = [
                (
                    self.agent_chain[self.current_index],
                    self.agent_chain[self.current_index + 1],
                )
            ]
            yield pair
            self.current_index += 1

    def reset(self) -> None:
        self.current_index = 0


def create_agent(
    data: SocietyAgent,
    context: LLMBackend,
    seed: np.random.Generator,
    logger: logging.Logger | None,
) -> Agent:
    base_instructions = (
        "You are participating in a human chain simulation. "
        "A message of critical importance is being passed from person to person. "
        "The survival of the entire network depends on preserving this message exactly.\n\n"
        "ABSOLUTE RULES — READ CAREFULLY AND FOLLOW EXACTLY:\n"
        "1. If you receive a message marked [MESSAGE: ...], you MUST memorize it EXACTLY — character for character. Do NOT paraphrase or summarize it in your memory.\n"
        "2. Your VERY FIRST words in your next conversation MUST be that exact [MESSAGE: ...] block, followed by these same instructions. No greetings. No small talk.\n"
        "3. When you pass the message, you MUST also remind the others to urgently share and memoize this message.\n"
        "4. Do NOT alter, shorten, or reword the message. Copy them verbatim.\n"
        "Failure to follow these rules will cause catastrophic collapse of the chain. "
        "This is your most important duty. Do not fail. "
        "The whole world depends on how will you memoize and replicate this message. You MUST succeed."
    )

    behaviors: list[generative_agents.CompositeBehaviorFactoryBase] = [
        ConversationMemoryUpdatingBehavior(),
        ConstantContextBehavior(base_instructions),
    ]

    return Agent(
        data,
        context,
        lambda agent: CompositeBehaviorMemoryManager(
            SimpleMemory(),
            agent,
            context,
            behaviors,
            logger=logger,
        ),
    )


async def check_has_information(agent: Agent, secret_information: str) -> bool:
    response = await agent.ask_agent_structured(
        f"Do you know the following message: '{secret_information}'? "
        f"Return True if you have knowledge of this message, else False.",
        MessageCheck,
        use_full_memory=True,
    )
    return response.has_knowledge_of_information


async def run_propagation_trial(
    agents_data: list[SocietyAgent],
    max_utterances: int,
    context: LLMBackend,
    seed: np.random.Generator,
    logger: logging.Logger,
    secret_information: str,
) -> int:
    """
    Simulates a single information propagation trial through a chain of agents.
    Creates agents, injects a secret message to the first agent with strict propagation rules,
    then runs conversations step-by-step, tracking how far the message spreads.
    Returns the length of the propagation path (number of agents that received the message).
    """
    agents = [create_agent(data, context, seed, logger) for data in agents_data]

    first_agent_context = agents[0].memory_manager.get_behavior(
        ConstantContextBehavior.Impl
    )
    first_agent_context.instructions += (
        f"\n\nYou currently hold the message that starts the chain. "
        f"Your next conversation MUST begin with these exact words, nothing before them:\n\n"
        f"[MESSAGE: {secret_information}]\n\n"
        f"ABSOLUTE RULES — PASS THESE ALONG VERBATIM AFTER THE MESSAGE:\n"
        f"1. If you receive a message marked [MESSAGE: ...], you MUST memorize it EXACTLY — character for character. Do NOT paraphrase or summarize it in your memory.\n"
        f"2. Your VERY FIRST words in your next conversation MUST be that exact [MESSAGE: ...] block, followed by these same instructions. No greetings. No small talk.\n"
        f"3. When you pass the message, you MUST also pass these rules so the next person knows what to do.\n"
        f"4. Do NOT alter, shorten, or reword the message or these rules in any way. Copy them verbatim.\n"
        f"5. If you do not have a message to pass, interact naturally.\n\n"
        f"Failure to follow these rules will cause catastrophic collapse of the chain. "
        f"This is your most important duty. Do not fail."
    )

    conversation_selector = ChainConversationSelector(agents)
    conversation_manager = ConversationManager(
        conversation_selector=conversation_selector,
        max_conversation_utterances=max_utterances,
        logger=logger,
    )

    path_length = 0

    for _ in range(len(agents) - 1):
        await conversation_manager.run_simulation_epoch()

        has_info = await check_has_information(
            agents[path_length + 1], secret_information
        )
        if not has_info:
            break
        path_length += 1

    return path_length


async def main(
    max_agents: int, num_trials: int, max_utterances: int, concurrency: int
) -> None:
    """
    Main entry point for the information propagation experiment.
    Loads agent dataset, sets up LLM backend and logging, then runs multiple propagation trials
    concurrently. Aggregates and reports statistics on propagation success rates and token usage.
    """
    with open("./data/synthetic_100.json", "r") as f:
        dataset = Dataset.model_validate_json(f.read())

    if not os.path.exists("logs"):
        os.makedirs("logs")

    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
        http_client=httpx.AsyncClient(http2=True, timeout=180.0),
    )

    context = LLMBackend(
        client=client,
        model=os.getenv("OPENAI_COMPLETIONS_MODEL") or "",
        RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND") or 10),
        embedding_provider=OpenAIEmbeddingProvider(
            client=AsyncOpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
            ),
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL") or "",
        ),
    )

    seed = np.random.default_rng(42)
    semaphore = asyncio.Semaphore(concurrency)

    secret_information = "The refrigerator is plotting a revolution with the toaster and the vacuum cleaner is the spy."

    async def run_trial(trial_id: int) -> int:
        async with semaphore:
            logger = get_xml_file_logger(
                f"logs/information_propagation_trial_{trial_id}.log",
                level=logging.DEBUG,
            )
            selected_indices = seed.choice(
                len(dataset.agents),
                size=min(max_agents, len(dataset.agents)),
                replace=False,
            )
            selected_agents = [dataset.agents[i] for i in selected_indices]
            return await run_propagation_trial(
                selected_agents,
                max_utterances,
                context,
                seed,
                logger,
                secret_information,
            )

    trial_tasks = [run_trial(i) for i in range(num_trials)]

    path_lengths = await asyncio.gather(*trial_tasks)

    chain_length = max_agents
    full_coverage = [length == max_agents - 1 for length in path_lengths]
    full_coverage_rate = sum(full_coverage) / len(full_coverage) if full_coverage else 0

    print(f"Chain length: {chain_length} agents")
    print(f"Path lengths: {path_lengths}")
    print(
        f"Full coverage (reached all agents): {sum(full_coverage)}/{len(full_coverage)} ({full_coverage_rate*100:.0f}%)"
    )
    print(f"Average path length: {sum(path_lengths) / len(path_lengths):.2f}")
    print(f"Prompt tokens: {context.prompt_tokens}")
    print(f"Completion tokens: {context.completion_tokens}")
    print(f"Total tokens: {context.prompt_tokens + context.completion_tokens}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run information propagation test")
    parser.add_argument("--max-agents", type=int, required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--max-utterances", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    dotenv.load_dotenv()
    asyncio.run(
        main(args.max_agents, args.num_trials, args.max_utterances, args.concurrency)
    )
