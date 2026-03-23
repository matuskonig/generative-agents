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
from data_generation_utils import Dataset
from logger_utils import get_xml_file_logger
from openai import AsyncOpenAI
from synthetic_experiment_utils import (
    BDIForgettingOnlyManagerType,
    BDIMemoryManagerType,
    BDIPLanningOnlyManagerType,
    EmbeddingMemoryType,
    ExperimentResult,
    ReducedInformationSpreadConfig,
    SimpleMemoryManagerType,
    SimpleMemoryType,
    UpdaterBehaviorType,
    run_experiment,
)

from generative_agents import (
    LLMBackend,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    default_config,
)

EXPERIMENT_MAX_UTTERANCES = 10
EXPERIMENT_EPOCHS = 10


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
        http_client=httpx.AsyncClient(http2=True, timeout=120.0),
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
        # B3: unitary10
        tg.create_task(
            run_and_save(
                run_experiment(
                    get_context=get_context,
                    dataset=dataset10,
                    logger=get_xml_file_logger(
                        "./logs/B3_unitary_10.log", level=logging.DEBUG
                    ),
                    experiment_name="B3_unitary_10",
                    memory_manager_config=baseline_mem_mgr,
                    memory_config=baseline_mem,
                    conversation_selector_type="information_spread",
                    updater_behavior_type=UpdaterBehaviorType(behavior_type="unitary"),
                    seed=42,
                    max_utterances=EXPERIMENT_MAX_UTTERANCES,
                    epochs=EXPERIMENT_EPOCHS,
                ),
                "./results/B3_unitary_10.json",
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
            (5, 5, 1),  # half both
            (5, 10, 2),  # 5 epochs, 10 utt
            (10, 5, 3),  # 10 epochs, 5 utt
            (10, 10, 4),  # baseline
            (20, 10, 5),  # 2x epochs
            (10, 20, 6),  # 2x utterances
            (20, 20, 7),  # 2x both
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
            experimement_name = (
                f"E{len(epoch_utterance_combos)+idx}_{epochs_val}e_{utt_val}u_10"
            )
            # 10 agents
            tg.create_task(
                run_and_save(
                    run_experiment(
                        get_context=get_context,
                        dataset=dataset10,
                        logger=get_xml_file_logger(
                            f"./logs/{experimement_name}.log",
                            level=logging.DEBUG,
                        ),
                        experiment_name=f"{experimement_name}",
                        memory_manager_config=baseline_mem_mgr,
                        memory_config=baseline_mem,
                        conversation_selector_type="information_spread",
                        updater_behavior_type=baseline_updater,
                        seed=42,
                        max_utterances=utt_val,
                        epochs=epochs_val,
                    ),
                    f"./results/{experimement_name}.json",
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
