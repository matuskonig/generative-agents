from .embedding_memory import (
    EmbeddingMemory,
    fixed_count_strategy_factory,
    mean_std_count_strategy_factory,
    top_std_count_strategy_factory,
)
from .memory_base import MemoryBase
from .memory_managers import (
    BDIPlanningBehavior,
    CompositeBehaviorFactoryBase,
    CompositeBehaviorMemoryManager,
    MemoryForgettingBehavior,
    MemoryManagerBase,
    MemoryUpdatingBehavior,
    get_record_removal_linear_probability,
)
from .models import (
    BDIChangeIntention,
    BDIData,
    BDIFullChange,
    BDINoChanges,
    BDIResponse,
    FactResponse,
    MemoryRecord,
    MemoryRecordResponse,
    MemoryRecordWithEmbedding,
    PruneFactsResponse,
)
from .simple_memory import SimpleMemory

__all__ = [
    "MemoryBase",
    "SimpleMemory",
    "EmbeddingMemory",
    "MemoryManagerBase",
    "PruneFactsResponse",
    "FactResponse",
    "MemoryRecord",
    "MemoryRecordResponse",
    "MemoryRecordWithEmbedding",
    "BDIData",
    "BDIResponse",
    "fixed_count_strategy_factory",
    "mean_std_count_strategy_factory",
    "top_std_count_strategy_factory",
    "get_record_removal_linear_probability",
    "CompositeBehaviorFactoryBase",
    "CompositeBehaviorMemoryManager",
    "MemoryUpdatingBehavior",
    "BDIPlanningBehavior",
    "MemoryForgettingBehavior",
]

# TODO: switch to file-based prompts together with

# TODO: add some freetext field to the BDI to support model writing notes, planning and reasoning
# TODO: add possibly something to extend the actions

# TODO: ako vyriesime generalizaciu a context injcetion do jednotlivych agentov (mimo background)
# TODO: make pruning a standalone configuration

# TODO Affordable memory
