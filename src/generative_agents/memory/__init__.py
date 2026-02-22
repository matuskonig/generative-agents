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
    ConversationMemoryUpdatingBehavior,
    MemoryForgettingBehavior,
    MemoryManagerBase,
    UnitaryAgentNoteUpdatingBehavior,
    get_record_removal_linear_probability,
)
from .models import (
    BDIChangeIntention,
    BDIData,
    BDIFullChange,
    BDINoChanges,
    BDIResponse,
    BuildInSourceType,
    FactResponse,
    MemoryQueryFilter,
    MemoryRecord,
    MemoryRecordResponse,
    MemoryRecordWithEmbedding,
    PruneFactsResponse,
    RecordSourceTypeBase,
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
    "MemoryQueryFilter",
    "BDIData",
    "BDIResponse",
    "fixed_count_strategy_factory",
    "mean_std_count_strategy_factory",
    "top_std_count_strategy_factory",
    "get_record_removal_linear_probability",
    "CompositeBehaviorFactoryBase",
    "CompositeBehaviorMemoryManager",
    "ConversationMemoryUpdatingBehavior",
    "UnitaryAgentNoteUpdatingBehavior",
    "BDIPlanningBehavior",
    "MemoryForgettingBehavior",
    "RecordSourceTypeBase",
    "BuildInSourceType",
]

# TODO: switch to file-based prompts together with

# TODO: ako vyriesime generalizaciu a context injcetion do jednotlivych agentov (mimo background)

# TODO Affordable memory
