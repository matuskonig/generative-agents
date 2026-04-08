"""Experiments for testing information spread through a social network.

This module contains experiments that simulate information propagation through
agent-based social networks, testing how information spreads from seed agents
to other agents through conversations.
"""

import argparse
import asyncio
import logging
import os
from typing import Awaitable

import dotenv
import httpx
from abbl_study_utils import (
    BDIMemoryManagerType,
    EmbeddingMemoryType,
    ExperimentResult,
    SimpleMemoryForgettingManagerType,
    SimpleMemoryType,
    UpdaterBehaviorType,
    run_experiment,
)
from data_generation_utils import Dataset
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI

from generative_agents import (
    LLMBackend,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
)

EXPERIMENT_MAX_UTTERANCES = 10
EXPERIMENT_EPOCHS = 10
NUM_TRIALS = 10
BASE_SEED = 42

# Experiment names
MODIFIED_BASELINE_NAME = "study2_modified_baseline_agents25"
NO_BDI_SIMPLE_NAME = "study2_no_bdi_simple_agents25"


# =============================================================================
# Main Experiment Suite
# =============================================================================


async def main(concurrency: int, embedding_batch_size: int, use_local_embeddings: bool):
    """Run all experiments for the information spread study.

    This study compares two setups across 5 trials each:
    - Modified Baseline: Embedding memory + Top-K (k=25) + BDI + Full parallel selector
    - No-BDI Simple: Simple memory + Forgetting (no BDI) + Full parallel selector

    Args:
        concurrency: Number of experiments to run in parallel
        embedding_batch_size: Batch size for local embedding generation
        use_local_embeddings: Whether to use local sentence transformer embeddings
    """
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Load dataset
    with open("./data/synthetic_25.json", "r") as f:
        dataset25 = Dataset.model_validate_json(f.read())

    # Setup client
    api_key = os.getenv("OPENAI_API_KEY") or None
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
        http_client=httpx.AsyncClient(http2=True, timeout=180.0),
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
            client=AsyncOpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
            ),
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL") or "",
        )
    )

    def get_context() -> LLMBackend:
        return LLMBackend(
            client=client,
            model=os.getenv("OPENAI_COMPLETIONS_MODEL"),  # type: ignore
            RPS=int(os.getenv("MAX_REQUESTS_PER_SECOND")),  # type: ignore
            embedding_provider=embedding_provider,
        )

    updater_behavior = UpdaterBehaviorType(behavior_type="classical")
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
        for trial in range(NUM_TRIALS):
            seed = BASE_SEED + trial

            # Modified Baseline: Embedding memory + Top-K (k=25) + BDI + Full parallel selector
            baseline_experiment_name = f"{MODIFIED_BASELINE_NAME}_trial{trial}"
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset25,
                        logger=get_xml_file_logger(
                            f"./logs/{baseline_experiment_name}.log",
                            level=logging.DEBUG,
                        ),
                        experiment_name=baseline_experiment_name,
                        memory_manager_config=BDIMemoryManagerType(
                            manager_type="bdi", memory_removal_prob=0.5
                        ),
                        memory_config=EmbeddingMemoryType(
                            memory_type="embedding", strategy="top_k", value=25
                        ),
                        conversation_selector_type="full_parallel",
                        updater_behavior_type=updater_behavior,
                        seed=seed,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/{baseline_experiment_name}.json",
                )
            )

            # No-BDI Simple with Forgetting: Simple memory + forgetting + Full parallel selector
            simple_experiment_name = f"{NO_BDI_SIMPLE_NAME}_trial{trial}"
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset25,
                        logger=get_xml_file_logger(
                            f"./logs/{simple_experiment_name}.log", level=logging.DEBUG
                        ),
                        experiment_name=simple_experiment_name,
                        memory_manager_config=SimpleMemoryForgettingManagerType(
                            manager_type="simple_forgetting", memory_removal_prob=0.5
                        ),
                        memory_config=SimpleMemoryType(memory_type="simple"),
                        conversation_selector_type="full_parallel",
                        updater_behavior_type=updater_behavior,
                        seed=seed,
                        max_utterances=EXPERIMENT_MAX_UTTERANCES,
                        epochs=EXPERIMENT_EPOCHS,
                    ),
                    f"./results/{simple_experiment_name}.json",
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
        help="Number of parallel experiments",
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
