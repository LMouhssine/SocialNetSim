"""Scenarios module for experiments and comparisons."""

from .experiment import Experiment, ExperimentConfig, ExperimentResult
from .comparator import ExperimentComparator
from .presets import (
    create_algorithm_comparison,
    create_moderation_impact_study,
    create_echo_chamber_study,
    create_virality_analysis,
)

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentComparator",
    "create_algorithm_comparison",
    "create_moderation_impact_study",
    "create_echo_chamber_study",
    "create_virality_analysis",
]
