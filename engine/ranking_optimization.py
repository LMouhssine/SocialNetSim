"""Bandit-based feed ranking optimization.

Implements:
- Thompson Sampling for weight configuration exploration
- LinUCB for contextual bandits with user features
- Feed optimization manager
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import json

import numpy as np
from numpy.random import Generator


class BanditType(Enum):
    """Types of bandit algorithms."""

    THOMPSON = "thompson"
    LINUCB = "linucb"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class BanditArm:
    """Represents a single arm in the bandit.

    Attributes:
        arm_id: Unique identifier
        weights: Weight configuration for this arm
        alpha: Beta distribution alpha parameter (successes + 1)
        beta: Beta distribution beta parameter (failures + 1)
        total_pulls: Number of times arm was pulled
        total_reward: Cumulative reward
    """

    arm_id: str
    weights: dict[str, float]
    alpha: float = 1.0
    beta: float = 1.0
    total_pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Get mean reward for this arm."""
        if self.total_pulls == 0:
            return 0.0
        return self.total_reward / self.total_pulls

    def update(self, reward: float) -> None:
        """Update arm statistics with observed reward.

        Args:
            reward: Observed reward (0-1 for Thompson Sampling)
        """
        self.total_pulls += 1
        self.total_reward += reward

        # Update Beta distribution parameters
        self.alpha += reward
        self.beta += (1 - reward)

    def sample_theta(self, rng: Generator) -> float:
        """Sample from posterior Beta distribution.

        Args:
            rng: Random number generator

        Returns:
            Sampled value from Beta(alpha, beta)
        """
        return rng.beta(self.alpha, self.beta)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "arm_id": self.arm_id,
            "weights": self.weights,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_pulls": self.total_pulls,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BanditArm":
        """Create from dictionary."""
        return cls(
            arm_id=data["arm_id"],
            weights=data["weights"],
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            total_pulls=data.get("total_pulls", 0),
            total_reward=data.get("total_reward", 0.0),
        )


class ThompsonSamplingBandit:
    """Thompson Sampling bandit for weight configuration optimization.

    Each arm represents a different weight configuration for feed ranking.
    """

    def __init__(
        self,
        arms: list[BanditArm] | None = None,
        seed: int | None = None,
    ):
        """Initialize Thompson Sampling bandit.

        Args:
            arms: List of arms (weight configurations)
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)
        self.arms: dict[str, BanditArm] = {}

        if arms:
            for arm in arms:
                self.arms[arm.arm_id] = arm

    def add_arm(self, arm: BanditArm) -> None:
        """Add an arm to the bandit.

        Args:
            arm: Arm to add
        """
        self.arms[arm.arm_id] = arm

    def select_arm(self) -> BanditArm:
        """Select an arm using Thompson Sampling.

        Returns:
            Selected arm
        """
        if not self.arms:
            raise ValueError("No arms available")

        # Sample from each arm's posterior
        samples = {
            arm_id: arm.sample_theta(self.rng)
            for arm_id, arm in self.arms.items()
        }

        # Select arm with highest sample
        best_arm_id = max(samples, key=samples.get)
        return self.arms[best_arm_id]

    def update_arm(self, arm_id: str, reward: float) -> None:
        """Update arm with observed reward.

        Args:
            arm_id: ID of pulled arm
            reward: Observed reward (0-1)
        """
        if arm_id not in self.arms:
            raise ValueError(f"Unknown arm: {arm_id}")
        self.arms[arm_id].update(reward)

    def get_best_arm(self) -> BanditArm:
        """Get arm with highest mean reward.

        Returns:
            Best arm by mean reward
        """
        return max(self.arms.values(), key=lambda a: a.mean_reward)

    def get_statistics(self) -> dict[str, Any]:
        """Get bandit statistics.

        Returns:
            Dictionary of statistics
        """
        total_pulls = sum(a.total_pulls for a in self.arms.values())
        return {
            "n_arms": len(self.arms),
            "total_pulls": total_pulls,
            "arms": {
                arm_id: {
                    "pulls": arm.total_pulls,
                    "mean_reward": arm.mean_reward,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                }
                for arm_id, arm in self.arms.items()
            },
            "best_arm": self.get_best_arm().arm_id if self.arms else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> "ThompsonSamplingBandit":
        """Create from dictionary."""
        bandit = cls(seed=seed)
        for arm_data in data.get("arms", {}).values():
            bandit.add_arm(BanditArm.from_dict(arm_data))
        return bandit


@dataclass
class LinUCBArm:
    """Arm for LinUCB contextual bandit.

    Attributes:
        arm_id: Unique identifier
        weights: Weight configuration for this arm
        A: d x d matrix for ridge regression
        b: d x 1 vector for ridge regression
        d: Feature dimension
    """

    arm_id: str
    weights: dict[str, float]
    d: int
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)

    def __post_init__(self):
        """Initialize matrices."""
        if self.A is None:
            self.A = np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)

    def get_ucb(self, context: np.ndarray, alpha: float = 1.0) -> float:
        """Calculate UCB value for given context.

        Args:
            context: Context feature vector (d,)
            alpha: Exploration parameter

        Returns:
            UCB value
        """
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        # UCB = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
        mean = theta @ context
        uncertainty = alpha * np.sqrt(context @ A_inv @ context)

        return mean + uncertainty

    def update(self, context: np.ndarray, reward: float) -> None:
        """Update arm with observed context and reward.

        Args:
            context: Context feature vector
            reward: Observed reward
        """
        self.A = self.A + np.outer(context, context)
        self.b = self.b + reward * context


class ContextualLinUCBBandit:
    """LinUCB contextual bandit for feed optimization.

    Uses user features as context to personalize arm selection.
    """

    def __init__(
        self,
        d: int,
        alpha: float = 1.0,
        arms: list[LinUCBArm] | None = None,
        seed: int | None = None,
    ):
        """Initialize LinUCB bandit.

        Args:
            d: Context dimension (number of user features)
            alpha: Exploration parameter
            arms: List of arms
            seed: Random seed
        """
        self.d = d
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.arms: dict[str, LinUCBArm] = {}

        if arms:
            for arm in arms:
                self.arms[arm.arm_id] = arm

    def add_arm(self, arm_id: str, weights: dict[str, float]) -> None:
        """Add an arm to the bandit.

        Args:
            arm_id: Arm identifier
            weights: Weight configuration
        """
        self.arms[arm_id] = LinUCBArm(arm_id=arm_id, weights=weights, d=self.d)

    def select_arm(self, context: np.ndarray) -> LinUCBArm:
        """Select arm based on context using UCB.

        Args:
            context: User feature vector

        Returns:
            Selected arm
        """
        if not self.arms:
            raise ValueError("No arms available")

        # Calculate UCB for each arm
        ucb_values = {
            arm_id: arm.get_ucb(context, self.alpha)
            for arm_id, arm in self.arms.items()
        }

        # Select arm with highest UCB
        best_arm_id = max(ucb_values, key=ucb_values.get)
        return self.arms[best_arm_id]

    def update_arm(
        self,
        arm_id: str,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update arm with observed context and reward.

        Args:
            arm_id: ID of pulled arm
            context: Context feature vector
            reward: Observed reward
        """
        if arm_id not in self.arms:
            raise ValueError(f"Unknown arm: {arm_id}")
        self.arms[arm_id].update(context, reward)


@dataclass
class FeedOptimizationConfig:
    """Configuration for feed optimization.

    Attributes:
        enabled: Whether optimization is enabled
        bandit_type: Type of bandit algorithm
        alpha: Exploration parameter
        warmup_pulls: Minimum pulls per arm before optimization
        update_frequency: Steps between weight updates
        weight_configurations: Predefined weight configurations to try
    """

    enabled: bool = False
    bandit_type: BanditType = BanditType.THOMPSON
    alpha: float = 1.0
    warmup_pulls: int = 100
    update_frequency: int = 10
    weight_configurations: list[dict[str, float]] = field(default_factory=list)


class FeedOptimizer:
    """Manages bandit-based feed ranking optimization.

    Coordinates between feed ranker and bandit algorithms
    to optimize engagement metrics.
    """

    def __init__(
        self,
        config: FeedOptimizationConfig,
        seed: int | None = None,
    ):
        """Initialize feed optimizer.

        Args:
            config: Optimization configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Initialize bandit based on type
        if config.bandit_type == BanditType.THOMPSON:
            self.bandit = ThompsonSamplingBandit(seed=seed)
        else:
            # LinUCB requires context dimension, set up lazily
            self.bandit = None

        # Initialize arms from config
        self._initialize_arms()

        # Tracking
        self.current_arm: BanditArm | LinUCBArm | None = None
        self.step_count = 0
        self.total_reward = 0.0
        self.reward_history: list[float] = []

    def _initialize_arms(self) -> None:
        """Initialize arms from configuration."""
        if not self.config.weight_configurations:
            # Create default weight configurations
            self.config.weight_configurations = [
                # Balanced
                {"recency": 0.3, "velocity": 0.3, "relevance": 0.4},
                # Recency-focused
                {"recency": 0.5, "velocity": 0.2, "relevance": 0.3},
                # Engagement-focused
                {"recency": 0.2, "velocity": 0.4, "relevance": 0.4},
                # Relevance-focused
                {"recency": 0.2, "velocity": 0.2, "relevance": 0.6},
            ]

        if isinstance(self.bandit, ThompsonSamplingBandit):
            for i, weights in enumerate(self.config.weight_configurations):
                arm = BanditArm(arm_id=f"arm_{i}", weights=weights)
                self.bandit.add_arm(arm)

    def select_weights(
        self,
        user_context: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Select weight configuration for feed ranking.

        Args:
            user_context: Optional user feature vector (for contextual bandits)

        Returns:
            Weight configuration dictionary
        """
        if not self.config.enabled or self.bandit is None:
            # Return default weights
            return {"recency": 0.3, "velocity": 0.3, "relevance": 0.4}

        if isinstance(self.bandit, ThompsonSamplingBandit):
            self.current_arm = self.bandit.select_arm()
        elif isinstance(self.bandit, ContextualLinUCBBandit):
            if user_context is None:
                raise ValueError("LinUCB requires user context")
            self.current_arm = self.bandit.select_arm(user_context)
        else:
            return {"recency": 0.3, "velocity": 0.3, "relevance": 0.4}

        return self.current_arm.weights

    def record_reward(
        self,
        reward: float,
        user_context: np.ndarray | None = None,
    ) -> None:
        """Record reward for current arm selection.

        Args:
            reward: Observed reward (e.g., engagement rate)
            user_context: User context (for LinUCB)
        """
        if not self.config.enabled or self.current_arm is None:
            return

        # Update bandit
        if isinstance(self.bandit, ThompsonSamplingBandit):
            self.bandit.update_arm(self.current_arm.arm_id, reward)
        elif isinstance(self.bandit, ContextualLinUCBBandit) and user_context is not None:
            self.bandit.update_arm(self.current_arm.arm_id, user_context, reward)

        # Track
        self.total_reward += reward
        self.reward_history.append(reward)
        self.step_count += 1

    def calculate_engagement_reward(
        self,
        views: int,
        likes: int,
        shares: int,
        comments: int,
    ) -> float:
        """Calculate reward from engagement metrics.

        Args:
            views: Number of views
            likes: Number of likes
            shares: Number of shares
            comments: Number of comments

        Returns:
            Reward value (0-1)
        """
        if views == 0:
            return 0.0

        # Weighted engagement rate
        engagement = (likes + 2 * shares + 1.5 * comments) / views

        # Normalize to 0-1 range (assuming max engagement rate around 0.5)
        reward = min(1.0, engagement / 0.5)

        return reward

    def get_best_weights(self) -> dict[str, float]:
        """Get best performing weight configuration.

        Returns:
            Best weight configuration
        """
        if not self.config.enabled or self.bandit is None:
            return {"recency": 0.3, "velocity": 0.3, "relevance": 0.4}

        if isinstance(self.bandit, ThompsonSamplingBandit):
            return self.bandit.get_best_arm().weights
        else:
            return {"recency": 0.3, "velocity": 0.3, "relevance": 0.4}

    def get_statistics(self) -> dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "enabled": self.config.enabled,
            "bandit_type": self.config.bandit_type.value,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / max(1, self.step_count),
        }

        if isinstance(self.bandit, ThompsonSamplingBandit):
            stats["bandit_stats"] = self.bandit.get_statistics()

        return stats

    def should_update(self) -> bool:
        """Check if weights should be updated this step.

        Returns:
            True if update should occur
        """
        return self.step_count % self.config.update_frequency == 0


def create_user_context_vector(
    activity_level: float,
    emotional_reactivity: float,
    confirmation_bias: float,
    follower_count: int,
    avg_engagement_rate: float,
) -> np.ndarray:
    """Create user context vector for contextual bandits.

    Args:
        activity_level: User's activity level
        emotional_reactivity: User's emotional reactivity
        confirmation_bias: User's confirmation bias
        follower_count: Number of followers
        avg_engagement_rate: User's average engagement rate

    Returns:
        Context vector
    """
    # Normalize follower count
    normalized_followers = min(1.0, follower_count / 1000)

    return np.array([
        activity_level,
        emotional_reactivity,
        confirmation_bias,
        normalized_followers,
        avg_engagement_rate,
    ])
