"""Opinion dynamics with bounded confidence model.

Implements:
- Deffuant-Weisbuch bounded confidence model
- Opinion updates from peer interactions
- Content exposure influence
- Polarization tracking and measurement
"""

from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

import numpy as np
from numpy.random import Generator

from models import User, Post
from .state import SimulationState


@dataclass
class OpinionState:
    """Opinion state for a single user.

    Attributes:
        user_id: User identifier
        opinion: Current opinion (-1 to 1)
        confidence: Confidence in opinion (0-1)
        openness: Openness to different opinions (0-1)
        opinion_history: Historical opinion values
        interaction_count: Number of opinion-changing interactions
    """

    user_id: str
    opinion: float = 0.0
    confidence: float = 0.5
    openness: float = 0.5
    opinion_history: list[tuple[int, float]] = field(default_factory=list)
    interaction_count: int = 0

    def __post_init__(self):
        """Validate and clip values."""
        self.opinion = np.clip(self.opinion, -1.0, 1.0)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
        self.openness = np.clip(self.openness, 0.0, 1.0)

    def record_opinion(self, step: int) -> None:
        """Record current opinion for history tracking."""
        self.opinion_history.append((step, self.opinion))

    def update_opinion(
        self,
        influence: float,
        weight: float,
        step: int | None = None,
    ) -> float:
        """Update opinion based on external influence.

        Args:
            influence: External opinion influence (-1 to 1)
            weight: Weight of influence (0-1)
            step: Current step for history

        Returns:
            Change in opinion
        """
        old_opinion = self.opinion

        # Weighted average update
        effective_weight = weight * self.openness
        self.opinion = (
            self.opinion * (1 - effective_weight) +
            influence * effective_weight
        )
        self.opinion = np.clip(self.opinion, -1.0, 1.0)

        self.interaction_count += 1

        if step is not None:
            self.record_opinion(step)

        return self.opinion - old_opinion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "opinion": self.opinion,
            "confidence": self.confidence,
            "openness": self.openness,
            "opinion_history": self.opinion_history,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpinionState":
        """Create from dictionary."""
        state = cls(
            user_id=data["user_id"],
            opinion=data.get("opinion", 0.0),
            confidence=data.get("confidence", 0.5),
            openness=data.get("openness", 0.5),
        )
        state.opinion_history = data.get("opinion_history", [])
        state.interaction_count = data.get("interaction_count", 0)
        return state


@dataclass
class OpinionDynamicsConfig:
    """Configuration for opinion dynamics model.

    Attributes:
        confidence_bound: Epsilon for bounded confidence (max opinion diff for interaction)
        convergence_rate: Rate of opinion convergence (mu in Deffuant model)
        content_influence_weight: Weight of content exposure on opinion
        peer_influence_weight: Weight of peer interactions on opinion
        extremism_resistance: Resistance to moving toward extremes
        echo_chamber_threshold: Opinion similarity threshold for echo chamber
        record_interval: Steps between opinion history recordings
    """

    confidence_bound: float = 0.3
    convergence_rate: float = 0.1
    content_influence_weight: float = 0.5
    peer_influence_weight: float = 0.5
    extremism_resistance: float = 0.1
    echo_chamber_threshold: float = 0.2
    record_interval: int = 10


class BoundedConfidenceModel:
    """Deffuant-Weisbuch bounded confidence model.

    Users only interact (update opinions) if their opinions
    are within epsilon (confidence_bound) of each other.

    Update rule:
    If |opinion_i - opinion_j| < epsilon:
        opinion_i += mu * (opinion_j - opinion_i)
        opinion_j += mu * (opinion_i - opinion_j)
    """

    def __init__(
        self,
        config: OpinionDynamicsConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize bounded confidence model.

        Args:
            config: Model configuration
            seed: Random seed
        """
        self.config = config or OpinionDynamicsConfig()
        self.rng = np.random.default_rng(seed)

    def can_interact(
        self,
        opinion1: float,
        opinion2: float,
        openness1: float = 1.0,
        openness2: float = 1.0,
    ) -> bool:
        """Check if two agents can interact based on bounded confidence.

        Args:
            opinion1: First agent's opinion
            opinion2: Second agent's opinion
            openness1: First agent's openness
            openness2: Second agent's openness

        Returns:
            True if interaction can occur
        """
        # Effective confidence bound adjusted by openness
        effective_bound = self.config.confidence_bound * (openness1 + openness2) / 2
        return abs(opinion1 - opinion2) < effective_bound

    def interact(
        self,
        state1: OpinionState,
        state2: OpinionState,
        step: int,
    ) -> tuple[float, float]:
        """Process interaction between two agents.

        Args:
            state1: First agent's opinion state
            state2: Second agent's opinion state
            step: Current simulation step

        Returns:
            Tuple of (change1, change2) in opinions
        """
        if not self.can_interact(state1.opinion, state2.opinion, state1.openness, state2.openness):
            return 0.0, 0.0

        # Calculate update
        diff = state2.opinion - state1.opinion
        mu = self.config.convergence_rate

        # Each agent moves toward the other
        change1 = mu * diff * state1.openness
        change2 = -mu * diff * state2.openness

        # Apply extremism resistance
        if abs(state1.opinion) > 0.8:
            change1 *= (1 - self.config.extremism_resistance)
        if abs(state2.opinion) > 0.8:
            change2 *= (1 - self.config.extremism_resistance)

        # Update states
        state1.opinion = np.clip(state1.opinion + change1, -1.0, 1.0)
        state2.opinion = np.clip(state2.opinion + change2, -1.0, 1.0)

        state1.interaction_count += 1
        state2.interaction_count += 1

        if step % self.config.record_interval == 0:
            state1.record_opinion(step)
            state2.record_opinion(step)

        return change1, change2

    def update_from_content(
        self,
        state: OpinionState,
        content_opinion: float,
        engagement_strength: float,
        step: int,
    ) -> float:
        """Update opinion based on content exposure.

        Args:
            state: User's opinion state
            content_opinion: Opinion expressed in content
            engagement_strength: Strength of engagement (view < like < share)
            step: Current step

        Returns:
            Change in opinion
        """
        # Only update if content is within confidence bound
        if not self.can_interact(state.opinion, content_opinion, state.openness, 1.0):
            return 0.0

        # Content influence weighted by engagement strength
        influence_weight = (
            self.config.content_influence_weight *
            engagement_strength *
            state.openness
        )

        return state.update_opinion(content_opinion, influence_weight, step)


class OpinionDynamicsEngine:
    """Manages opinion dynamics for the simulation.

    Coordinates:
    - Opinion state tracking for all users
    - Peer interaction processing
    - Content exposure effects
    - Polarization measurement
    """

    def __init__(
        self,
        config: OpinionDynamicsConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize opinion dynamics engine.

        Args:
            config: Engine configuration
            seed: Random seed
        """
        self.config = config or OpinionDynamicsConfig()
        self.rng = np.random.default_rng(seed)

        # Bounded confidence model
        self.model = BoundedConfidenceModel(config, seed)

        # User opinion states
        self.opinion_states: dict[str, OpinionState] = {}

        # Tracking
        self.step_count = 0
        self.total_interactions = 0
        self.polarization_history: list[tuple[int, float]] = []

    def initialize_user(
        self,
        user: User,
        initial_opinion: float | None = None,
    ) -> OpinionState:
        """Initialize opinion state for a user.

        Args:
            user: User to initialize
            initial_opinion: Initial opinion (defaults to ideology)

        Returns:
            Created OpinionState
        """
        if initial_opinion is None:
            initial_opinion = user.traits.ideology

        # Openness inversely related to confirmation bias
        openness = 1.0 - user.traits.confirmation_bias * 0.5

        state = OpinionState(
            user_id=user.user_id,
            opinion=initial_opinion,
            confidence=0.5 + user.traits.confirmation_bias * 0.3,
            openness=openness,
        )

        self.opinion_states[user.user_id] = state
        return state

    def initialize_users(
        self,
        users: dict[str, User],
    ) -> None:
        """Initialize opinion states for a collection of users."""
        for user in users.values():
            self.initialize_user(user)

    def get_opinion_state(self, user_id: str) -> OpinionState | None:
        """Get user's opinion state.

        Args:
            user_id: User identifier

        Returns:
            OpinionState or None
        """
        return self.opinion_states.get(user_id)

    def process_peer_interaction(
        self,
        user1_id: str,
        user2_id: str,
        step: int,
    ) -> tuple[float, float]:
        """Process opinion interaction between two users.

        Args:
            user1_id: First user ID
            user2_id: Second user ID
            step: Current step

        Returns:
            Tuple of opinion changes
        """
        state1 = self.opinion_states.get(user1_id)
        state2 = self.opinion_states.get(user2_id)

        if state1 is None or state2 is None:
            return 0.0, 0.0

        changes = self.model.interact(state1, state2, step)
        self.total_interactions += 1

        return changes

    def process_content_exposure(
        self,
        user_id: str,
        content_ideology: float,
        engagement_type: str,
        step: int,
    ) -> float:
        """Process opinion change from content exposure.

        Args:
            user_id: User ID
            content_ideology: Ideology score of content
            engagement_type: Type of engagement (view, like, share)
            step: Current step

        Returns:
            Change in opinion
        """
        state = self.opinion_states.get(user_id)
        if state is None:
            return 0.0

        # Engagement strength
        engagement_strengths = {
            "view": 0.1,
            "like": 0.3,
            "share": 0.5,
            "comment": 0.4,
        }
        strength = engagement_strengths.get(engagement_type, 0.1)

        return self.model.update_from_content(
            state, content_ideology, strength, step
        )

    def step(self, step: int) -> None:
        """Process one step of opinion dynamics.

        Args:
            step: Current step
        """
        self.step_count = step

        # Record polarization periodically
        if step % self.config.record_interval == 0:
            polarization = self.compute_polarization()
            self.polarization_history.append((step, polarization))

    def compute_polarization(self) -> float:
        """Compute current polarization level.

        Uses variance of opinions as simple polarization measure.

        Returns:
            Polarization value (higher = more polarized)
        """
        if not self.opinion_states:
            return 0.0

        opinions = [s.opinion for s in self.opinion_states.values()]
        return float(np.var(opinions))

    def compute_polarization_metrics(
        self,
        users: dict[str, User] | None = None,
    ) -> dict[str, float | dict[str, list[str]]]:
        """Compute a bundle of polarization-related metrics."""
        metrics: dict[str, float | dict[str, list[str]]] = {
            "polarization": self.compute_polarization(),
            "bimodality": self.compute_bimodality(),
        }

        if users is not None:
            metrics["echo_chamber_index"] = self.compute_echo_chamber_index(users)
            metrics["disagreement_exposure"] = self.compute_disagreement_exposure(users)
            metrics["clusters"] = self.get_opinion_clusters()

        return metrics

    def compute_bimodality(self) -> float:
        """Compute bimodality coefficient (Sarle's).

        Higher values indicate more bimodal (polarized) distribution.

        Returns:
            Bimodality coefficient
        """
        if len(self.opinion_states) < 3:
            return 0.0

        opinions = np.array([s.opinion for s in self.opinion_states.values()])
        n = len(opinions)

        # Skewness
        mean = np.mean(opinions)
        std = np.std(opinions)
        if std < 1e-10:
            return 0.0

        skewness = np.mean(((opinions - mean) / std) ** 3)

        # Kurtosis
        kurtosis = np.mean(((opinions - mean) / std) ** 4) - 3

        # Sarle's bimodality coefficient
        # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))
        denominator = kurtosis + 3 * (n - 1) ** 2 / max(1, (n - 2) * (n - 3))
        if abs(denominator) < 1e-10:
            return 0.0

        bc = (skewness ** 2 + 1) / denominator
        return float(bc)

    def compute_echo_chamber_index(
        self,
        users: dict[str, User],
    ) -> float:
        """Compute echo chamber index.

        Ratio of within-cluster to between-cluster edges.

        Args:
            users: Dictionary of users

        Returns:
            Echo chamber index (higher = more echo chambers)
        """
        if not self.opinion_states:
            return 0.0

        within_cluster = 0
        between_cluster = 0

        for user_id, user in users.items():
            state1 = self.opinion_states.get(user_id)
            if state1 is None:
                continue

            for friend_id in user.following:
                state2 = self.opinion_states.get(friend_id)
                if state2 is None:
                    continue

                # Check if opinions are similar (same cluster)
                if abs(state1.opinion - state2.opinion) < self.config.echo_chamber_threshold:
                    within_cluster += 1
                else:
                    between_cluster += 1

        if between_cluster == 0:
            return float("inf") if within_cluster > 0 else 0.0

        return within_cluster / between_cluster

    def compute_disagreement_exposure(
        self,
        users: dict[str, User],
    ) -> float:
        """Compute average disagreement exposure.

        Lower values indicate filter bubbles.

        Args:
            users: Dictionary of users

        Returns:
            Average disagreement exposure
        """
        if not self.opinion_states:
            return 0.0

        total_exposure = 0.0
        count = 0

        for user_id, user in users.items():
            state1 = self.opinion_states.get(user_id)
            if state1 is None:
                continue

            user_exposure = 0.0
            friend_count = 0

            for friend_id in user.following:
                state2 = self.opinion_states.get(friend_id)
                if state2 is None:
                    continue

                user_exposure += abs(state1.opinion - state2.opinion)
                friend_count += 1

            if friend_count > 0:
                total_exposure += user_exposure / friend_count
                count += 1

        return total_exposure / max(1, count)

    def get_opinion_clusters(
        self,
        n_clusters: int = 2,
    ) -> dict[int, list[str]]:
        """Cluster users by opinion.

        Simple k-means style clustering.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary of cluster_id -> list of user_ids
        """
        if not self.opinion_states:
            return {}

        opinions = np.array([
            (uid, s.opinion)
            for uid, s in self.opinion_states.items()
        ], dtype=object)

        # Simple clustering: divide opinion range into n_clusters
        bins = np.linspace(-1, 1, n_clusters + 1)
        clusters: dict[int, list[str]] = defaultdict(list)

        for uid, op in opinions:
            cluster_idx = min(n_clusters - 1, np.digitize(float(op), bins) - 1)
            clusters[cluster_idx].append(uid)

        return dict(clusters)

    def get_statistics(self) -> dict[str, Any]:
        """Get opinion dynamics statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.opinion_states:
            return {}

        opinions = [s.opinion for s in self.opinion_states.values()]

        return {
            "n_users": len(self.opinion_states),
            "mean_opinion": float(np.mean(opinions)),
            "std_opinion": float(np.std(opinions)),
            "polarization": self.compute_polarization(),
            "bimodality": self.compute_bimodality(),
            "total_interactions": self.total_interactions,
            "min_opinion": float(np.min(opinions)),
            "max_opinion": float(np.max(opinions)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "opinion_states": {
                uid: s.to_dict()
                for uid, s in self.opinion_states.items()
            },
            "step_count": self.step_count,
            "total_interactions": self.total_interactions,
            "polarization_history": self.polarization_history,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: OpinionDynamicsConfig | None = None,
        seed: int | None = None,
    ) -> "OpinionDynamicsEngine":
        """Create from dictionary."""
        engine = cls(config, seed)

        for uid, state_data in data.get("opinion_states", {}).items():
            engine.opinion_states[uid] = OpinionState.from_dict(state_data)

        engine.step_count = data.get("step_count", 0)
        engine.total_interactions = data.get("total_interactions", 0)
        engine.polarization_history = data.get("polarization_history", [])

        return engine
