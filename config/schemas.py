"""Pydantic configuration schemas for SocialNetSim."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DistributionConfig(BaseModel):
    """Configuration for statistical distributions."""

    type: str = "normal"
    mean: float = 0.5
    std: float = 0.15
    min_val: float = 0.0
    max_val: float = 1.0
    alpha: float = 2.0  # For beta distribution
    beta: float = 5.0  # For beta distribution


class UserTraitsConfig(BaseModel):
    """Configuration for user trait distributions."""

    ideology: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="normal", mean=0.0, std=0.3, min_val=-1.0, max_val=1.0)
    )
    confirmation_bias: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=2.0, beta=5.0)
    )
    misinfo_susceptibility: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=2.0, beta=8.0)
    )
    emotional_reactivity: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=3.0, beta=3.0)
    )
    activity_level: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=2.0, beta=5.0)
    )


class UserConfig(BaseModel):
    """Configuration for user generation."""

    num_users: int = Field(default=1000, ge=10, le=100000)
    num_interests: int = Field(default=20, ge=5, le=100)
    interests_per_user: tuple[int, int] = Field(default=(3, 8))
    traits: UserTraitsConfig = Field(default_factory=UserTraitsConfig)


class NetworkConfig(BaseModel):
    """Configuration for network generation (Barabasi-Albert model)."""

    edges_per_new_node: int = Field(default=3, ge=1, le=20)
    weight_degree: float = Field(default=0.5, ge=0.0, le=1.0)
    weight_interest: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_ideology: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("weight_ideology")
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        """Ensure weights sum to approximately 1.0."""
        return v


class TopicConfig(BaseModel):
    """Configuration for topic generation."""

    num_topics: int = Field(default=50, ge=10, le=500)
    topic_popularity_alpha: float = Field(default=1.5, description="Power law exponent for topic popularity")
    controversy_distribution: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=2.0, beta=5.0)
    )


class ContentConfig(BaseModel):
    """Configuration for content generation."""

    topics: TopicConfig = Field(default_factory=TopicConfig)
    quality_distribution: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(type="beta", alpha=3.0, beta=2.0)
    )
    misinformation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    avg_posts_per_step: float = Field(default=0.1, description="Average posts per user per step")


class EngagementConfig(BaseModel):
    """Configuration for engagement model."""

    base_view_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    base_like_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    base_share_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    base_comment_rate: float = Field(default=0.03, ge=0.0, le=1.0)

    # Engagement factors
    interest_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    ideology_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    quality_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    social_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Temporal factors
    freshness_decay: float = Field(default=0.1, description="Decay rate for post freshness")
    fatigue_recovery: float = Field(default=0.2, description="Recovery rate for user fatigue")
    max_fatigue: float = Field(default=1.0, ge=0.0, le=2.0)

    # Utility-based decision model (Phase 2)
    use_utility_model: bool = Field(default=True, description="Use utility-based decisions")
    attention_recovery_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    emotional_decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    memory_length: int = Field(default=100, ge=10, le=1000)


class FeedConfig(BaseModel):
    """Configuration for feed ranking algorithms."""

    algorithm: str = Field(default="engagement", pattern="^(chronological|engagement|diverse|interest)$")
    feed_size: int = Field(default=20, ge=5, le=100)
    recency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    velocity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    diversity_penalty: float = Field(default=0.1, description="Penalty for showing similar content")
    seen_penalty: float = Field(default=0.5, description="Penalty for previously seen posts")

    # Enhanced ranking (Phase 3)
    controversy_amplification_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    controversy_cap: float = Field(default=0.3, ge=0.0, le=1.0, description="Max controversy boost")
    social_proximity_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    use_bandit_optimization: bool = Field(default=False)
    bandit_type: str = Field(default="thompson", pattern="^(thompson|linucb)$")


class CascadeConfig(BaseModel):
    """Configuration for viral cascade mechanics."""

    base_spread_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    virality_boost: float = Field(default=0.5, ge=0.0, le=2.0)
    share_velocity_multiplier: float = Field(default=0.1, ge=0.0, le=1.0)
    threshold_min: int = Field(default=1, ge=1, description="Min friends who shared before user considers sharing")
    threshold_max: int = Field(default=5, ge=1, description="Max threshold for threshold model")
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    # Hawkes process settings (Phase 4)
    use_hawkes: bool = Field(default=True, description="Use Hawkes processes for virality")
    hawkes_baseline: float = Field(default=0.01, ge=0.0, le=1.0)
    hawkes_branching_ratio: float = Field(default=0.8, ge=0.0, le=2.0)
    hawkes_decay: float = Field(default=0.1, ge=0.0, le=1.0)

    # Information diffusion settings (Phase 4)
    saturation_constant: float = Field(default=100.0, ge=1.0, description="Share count for 50% saturation")
    backlash_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    immunity_duration: int = Field(default=20, ge=1, description="Steps of immunity after sharing")
    enable_delayed_effects: bool = Field(default=True)


class ModerationConfig(BaseModel):
    """Configuration for content moderation."""

    enabled: bool = Field(default=True)
    detection_accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    false_positive_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    suppression_factor: float = Field(default=0.5, ge=0.0, le=1.0, description="Factor to reduce misinfo visibility")
    removal_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence threshold for removal")


class EventConfig(BaseModel):
    """Configuration for random events."""

    enabled: bool = Field(default=True)
    event_probability: float = Field(default=0.05, ge=0.0, le=1.0, description="Probability of event per step")

    # Event type probabilities (should sum to 1)
    political_shock_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    misinfo_wave_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    viral_trend_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    algorithm_change_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    external_event_prob: float = Field(default=0.25, ge=0.0, le=1.0)

    # Event parameters
    min_duration: int = Field(default=5, ge=1)
    max_duration: int = Field(default=20, ge=1)
    engagement_multiplier_range: tuple[float, float] = Field(default=(1.5, 3.0))


class AIConfig(BaseModel):
    """Configuration for AI models."""

    virality_predictor: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "train_interval": 50,  # Steps between retraining
        }
    )
    churn_predictor: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "churn_threshold_steps": 10,  # Steps of inactivity = churn
        }
    )
    misinfo_detector: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
        }
    )

    # Enhanced ML features (Phase 7)
    enhanced_features: bool = Field(default=True, description="Use enhanced feature extraction")
    user_embedding_dim: int = Field(default=64, ge=8, le=256)
    experiment_logging: bool = Field(default=True)


class OpinionDynamicsConfig(BaseModel):
    """Configuration for opinion dynamics (Phase 5)."""

    enabled: bool = Field(default=True, description="Enable opinion dynamics")
    confidence_bound: float = Field(default=0.3, ge=0.0, le=1.0, description="Max opinion difference for interaction")
    convergence_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Opinion convergence rate")
    content_influence_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    peer_influence_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    stubbornness_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Opinion confidence for stubbornness")


class PerformanceConfig(BaseModel):
    """Configuration for high-performance simulation (Phase 6)."""

    use_vectorized: bool = Field(default=True, description="Use vectorized operations")
    batch_size: int = Field(default=1000, ge=100, le=100000)
    use_parallel: bool = Field(default=False, description="Use parallel processing")
    n_workers: int = Field(default=4, ge=1, le=32)
    max_memory_gb: float = Field(default=8.0, ge=1.0, le=64.0)
    interaction_retention_steps: int = Field(default=100, ge=10, description="Steps to retain interactions")
    prune_frequency: int = Field(default=10, ge=1, description="Steps between memory pruning")
    gc_frequency: int = Field(default=50, ge=1, description="Steps between garbage collection")


class SimulationConfig(BaseModel):
    """Main simulation configuration."""

    name: str = Field(default="default")
    description: str = Field(default="Default simulation configuration")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    num_steps: int = Field(default=100, ge=1, le=10000)

    # Sub-configurations
    user: UserConfig = Field(default_factory=UserConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    feed: FeedConfig = Field(default_factory=FeedConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    moderation: ModerationConfig = Field(default_factory=ModerationConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    ai: AIConfig = Field(default_factory=AIConfig)

    # New configurations (Phases 5-6)
    opinion_dynamics: OpinionDynamicsConfig = Field(default_factory=OpinionDynamicsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Output settings
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    save_interval: int = Field(default=10, ge=1, description="Steps between state saves")
    output_dir: str = Field(default="data/simulations")


def load_config(config_path: str | Path) -> SimulationConfig:
    """Load configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return SimulationConfig(**config_dict)


def load_scenario(scenario_name: str, base_config_path: str | Path | None = None) -> SimulationConfig:
    """Load a scenario configuration, optionally merging with base config."""
    config_dir = Path(__file__).parent
    scenario_path = config_dir / "scenarios" / f"{scenario_name}.yaml"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_name}")

    # Load base config if provided
    if base_config_path:
        base_config = load_config(base_config_path)
        base_dict = base_config.model_dump()
    else:
        base_dict = {}

    # Load scenario config
    with open(scenario_path) as f:
        scenario_dict = yaml.safe_load(f)

    # Deep merge scenario into base
    merged = _deep_merge(base_dict, scenario_dict)
    return SimulationConfig(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
