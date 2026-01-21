"""Pre-built experiment configurations."""

from config.schemas import SimulationConfig
from .experiment import Experiment, ExperimentConfig


def create_algorithm_comparison(
    base_config: SimulationConfig | None = None,
    num_steps: int = 100,
    num_users: int = 500,
    num_runs: int = 3,
) -> Experiment:
    """Create experiment comparing feed algorithm performance.

    Compares:
    - Chronological feed
    - Engagement-optimized feed
    - Diverse feed
    - Interest-based feed

    Args:
        base_config: Base configuration (defaults to standard)
        num_steps: Steps per simulation
        num_users: Number of users
        num_runs: Runs per variation

    Returns:
        Configured Experiment
    """
    if base_config is None:
        base_config = SimulationConfig(
            name="algorithm_comparison",
            seed=42,
            num_steps=num_steps,
        )
        base_config.user.num_users = num_users

    config = ExperimentConfig(
        name="Feed Algorithm Comparison",
        description="Compares engagement, diversity, and misinformation spread across feed algorithms",
        base_config=base_config,
        num_runs=num_runs,
        variations={
            "chronological": {
                "feed": {"algorithm": "chronological"},
            },
            "engagement": {
                "feed": {"algorithm": "engagement"},
            },
            "diverse": {
                "feed": {"algorithm": "diverse", "diversity_penalty": 0.2},
            },
            "interest": {
                "feed": {"algorithm": "interest"},
            },
        },
        metrics_to_track=[
            "total_interactions",
            "total_shares",
            "engagement_rate",
            "share_rate",
            "misinfo_share_rate",
            "peak_cascades",
            "avg_active_users",
        ],
    )

    return Experiment(config)


def create_moderation_impact_study(
    base_config: SimulationConfig | None = None,
    num_steps: int = 100,
    num_users: int = 500,
    num_runs: int = 3,
) -> Experiment:
    """Create experiment studying moderation impact on misinformation.

    Compares:
    - No moderation
    - Light moderation (low accuracy)
    - Standard moderation
    - Aggressive moderation (high accuracy, high removal)

    Args:
        base_config: Base configuration
        num_steps: Steps per simulation
        num_users: Number of users
        num_runs: Runs per variation

    Returns:
        Configured Experiment
    """
    if base_config is None:
        base_config = SimulationConfig(
            name="moderation_study",
            seed=42,
            num_steps=num_steps,
        )
        base_config.user.num_users = num_users
        base_config.content.misinformation_rate = 0.1  # Higher misinfo for study

    config = ExperimentConfig(
        name="Moderation Impact Study",
        description="Studies how different moderation strategies affect misinformation spread",
        base_config=base_config,
        num_runs=num_runs,
        variations={
            "no_moderation": {
                "moderation": {"enabled": False},
            },
            "light_moderation": {
                "moderation": {
                    "enabled": True,
                    "detection_accuracy": 0.5,
                    "suppression_factor": 0.3,
                    "removal_threshold": 0.95,
                },
            },
            "standard_moderation": {
                "moderation": {
                    "enabled": True,
                    "detection_accuracy": 0.8,
                    "suppression_factor": 0.5,
                    "removal_threshold": 0.9,
                },
            },
            "aggressive_moderation": {
                "moderation": {
                    "enabled": True,
                    "detection_accuracy": 0.9,
                    "suppression_factor": 0.7,
                    "removal_threshold": 0.75,
                },
            },
        },
        metrics_to_track=[
            "total_misinfo_posts",
            "total_misinfo_shares",
            "misinfo_share_rate",
            "total_interactions",
            "engagement_rate",
        ],
    )

    return Experiment(config)


def create_echo_chamber_study(
    base_config: SimulationConfig | None = None,
    num_steps: int = 150,
    num_users: int = 1000,
    num_runs: int = 3,
) -> Experiment:
    """Create experiment studying echo chamber formation.

    Compares:
    - Neutral network (balanced weights)
    - Interest-driven network (high interest weight)
    - Ideology-driven network (high ideology weight)
    - High confirmation bias population

    Args:
        base_config: Base configuration
        num_steps: Steps per simulation
        num_users: Number of users
        num_runs: Runs per variation

    Returns:
        Configured Experiment
    """
    if base_config is None:
        base_config = SimulationConfig(
            name="echo_chamber_study",
            seed=42,
            num_steps=num_steps,
        )
        base_config.user.num_users = num_users

    config = ExperimentConfig(
        name="Echo Chamber Formation Study",
        description="Studies how network structure and user traits affect polarization",
        base_config=base_config,
        num_runs=num_runs,
        variations={
            "neutral_network": {
                "network": {
                    "weight_degree": 0.5,
                    "weight_interest": 0.25,
                    "weight_ideology": 0.25,
                },
                "engagement": {
                    "ideology_weight": 0.2,
                },
            },
            "interest_driven": {
                "network": {
                    "weight_degree": 0.3,
                    "weight_interest": 0.5,
                    "weight_ideology": 0.2,
                },
                "feed": {"algorithm": "interest"},
            },
            "ideology_driven": {
                "network": {
                    "weight_degree": 0.3,
                    "weight_interest": 0.2,
                    "weight_ideology": 0.5,
                },
                "engagement": {
                    "ideology_weight": 0.5,
                },
            },
            "high_confirmation_bias": {
                "user": {
                    "traits": {
                        "confirmation_bias": {
                            "type": "beta",
                            "alpha": 5.0,
                            "beta": 2.0,
                        },
                    },
                },
                "engagement": {
                    "ideology_weight": 0.4,
                },
            },
        },
        metrics_to_track=[
            "total_interactions",
            "total_shares",
            "engagement_rate",
            "ideology_variance",
            "bimodality_coefficient",
        ],
    )

    return Experiment(config)


def create_virality_analysis(
    base_config: SimulationConfig | None = None,
    num_steps: int = 100,
    num_users: int = 500,
    num_runs: int = 3,
) -> Experiment:
    """Create experiment analyzing viral content dynamics.

    Compares:
    - Baseline cascade parameters
    - High spread rate
    - Velocity-driven spread
    - Threshold-based spread

    Args:
        base_config: Base configuration
        num_steps: Steps per simulation
        num_users: Number of users
        num_runs: Runs per variation

    Returns:
        Configured Experiment
    """
    if base_config is None:
        base_config = SimulationConfig(
            name="virality_analysis",
            seed=42,
            num_steps=num_steps,
        )
        base_config.user.num_users = num_users

    config = ExperimentConfig(
        name="Viral Content Analysis",
        description="Analyzes how cascade parameters affect viral spread patterns",
        base_config=base_config,
        num_runs=num_runs,
        variations={
            "baseline": {
                "cascade": {
                    "base_spread_rate": 0.1,
                    "virality_boost": 0.5,
                    "share_velocity_multiplier": 0.1,
                },
            },
            "high_spread": {
                "cascade": {
                    "base_spread_rate": 0.2,
                    "virality_boost": 0.7,
                    "share_velocity_multiplier": 0.1,
                },
            },
            "velocity_driven": {
                "cascade": {
                    "base_spread_rate": 0.1,
                    "virality_boost": 0.3,
                    "share_velocity_multiplier": 0.3,
                },
            },
            "low_threshold": {
                "cascade": {
                    "base_spread_rate": 0.1,
                    "threshold_min": 1,
                    "threshold_max": 2,
                },
            },
        },
        metrics_to_track=[
            "total_shares",
            "peak_cascades",
            "total_cascade_reach",
            "engagement_rate",
            "share_rate",
        ],
    )

    return Experiment(config)


def create_event_response_study(
    base_config: SimulationConfig | None = None,
    num_steps: int = 100,
    num_users: int = 500,
    num_runs: int = 3,
) -> Experiment:
    """Create experiment studying platform response to events.

    Compares:
    - No events
    - High event probability
    - Events with strong moderation
    - Events with algorithm adaptation

    Args:
        base_config: Base configuration
        num_steps: Steps per simulation
        num_users: Number of users
        num_runs: Runs per variation

    Returns:
        Configured Experiment
    """
    if base_config is None:
        base_config = SimulationConfig(
            name="event_response_study",
            seed=42,
            num_steps=num_steps,
        )
        base_config.user.num_users = num_users

    config = ExperimentConfig(
        name="Event Response Study",
        description="Studies how platforms can respond to external events",
        base_config=base_config,
        num_runs=num_runs,
        variations={
            "no_events": {
                "events": {"enabled": False},
            },
            "high_events": {
                "events": {
                    "enabled": True,
                    "event_probability": 0.15,
                },
            },
            "events_with_moderation": {
                "events": {
                    "enabled": True,
                    "event_probability": 0.1,
                },
                "moderation": {
                    "enabled": True,
                    "detection_accuracy": 0.85,
                    "suppression_factor": 0.6,
                },
            },
            "events_with_diverse_feed": {
                "events": {
                    "enabled": True,
                    "event_probability": 0.1,
                },
                "feed": {
                    "algorithm": "diverse",
                    "diversity_penalty": 0.25,
                },
            },
        },
        metrics_to_track=[
            "total_interactions",
            "total_shares",
            "engagement_rate",
            "misinfo_share_rate",
            "peak_cascades",
        ],
    )

    return Experiment(config)
