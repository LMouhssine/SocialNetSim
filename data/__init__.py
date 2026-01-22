"""Data management module for SocialNetSim.

Provides:
- StateManager: Checkpointing and state coordination
- SimulationLogger: Structured append-only logging
- TrainingDataPreparer: ML-ready dataset preparation
"""

from .state_manager import (
    StateManager,
    ReproducibilityInfo,
    Checkpoint,
    JSONStateSerializer,
    PickleStateSerializer,
)
from .logger import (
    SimulationLogger,
    LogReader,
    LogEntry,
    LogEventType,
)
from .training_data import (
    TrainingDataPreparer,
    DatasetConfig,
    Dataset,
    DatasetType,
    DataAugmenter,
)

__all__ = [
    # State management
    "StateManager",
    "ReproducibilityInfo",
    "Checkpoint",
    "JSONStateSerializer",
    "PickleStateSerializer",
    # Logging
    "SimulationLogger",
    "LogReader",
    "LogEntry",
    "LogEventType",
    # Training data
    "TrainingDataPreparer",
    "DatasetConfig",
    "Dataset",
    "DatasetType",
    "DataAugmenter",
]
