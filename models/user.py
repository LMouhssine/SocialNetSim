"""User profile and traits models."""

from dataclasses import dataclass, field
from typing import Any
from collections import deque, Counter

import numpy as np

from .enums import UserState


@dataclass
class InteractionMemory:
    """Memory of a past interaction.

    Attributes:
        post_id: Post interacted with
        author_id: Author of the post
        interaction_type: Type of interaction
        step: When interaction occurred
        emotional_impact: Emotional impact of interaction (-1 to 1)
        topics: Topics of the post
    """

    post_id: str
    author_id: str
    interaction_type: str
    step: int
    emotional_impact: float = 0.0
    topics: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "post_id": self.post_id,
            "author_id": self.author_id,
            "interaction_type": self.interaction_type,
            "step": self.step,
            "emotional_impact": self.emotional_impact,
            "topics": list(self.topics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionMemory":
        """Create from dictionary."""
        return cls(
            post_id=data["post_id"],
            author_id=data["author_id"],
            interaction_type=data["interaction_type"],
            step=data["step"],
            emotional_impact=data.get("emotional_impact", 0.0),
            topics=tuple(data.get("topics", [])),
        )


@dataclass
class AuthorInteractionSummary:
    """Summary of interactions with a specific author.

    Attributes:
        author_id: Author identifier
        total_interactions: Total interaction count
        positive_interactions: Positive interactions (likes, shares)
        negative_interactions: Negative interactions (reports, blocks)
        last_interaction_step: Step of last interaction
        average_sentiment: Average sentiment of interactions
    """

    author_id: str
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_step: int = 0
    average_sentiment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "author_id": self.author_id,
            "total_interactions": self.total_interactions,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "last_interaction_step": self.last_interaction_step,
            "average_sentiment": self.average_sentiment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthorInteractionSummary":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UserCognitiveState:
    """Cognitive and emotional state of a user.

    Models the user's current mental/emotional state which
    affects their behavior and engagement decisions.

    Attributes:
        attention_budget: Remaining attention (0-1), depletes with activity
        emotional_valence: Current emotional valence (-1 to 1)
        emotional_arousal: Current emotional arousal (0-1)
        current_confirmation_bias: Dynamic confirmation bias (can evolve)
        interaction_memory: Recent interaction memories
        topic_exposure_counts: Count of exposures to each topic
        author_interaction_history: Summary of interactions with authors
        opinion: Current opinion on main dimension (-1 to 1)
        opinion_confidence: Confidence in current opinion (0-1)
    """

    attention_budget: float = 1.0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5
    current_confirmation_bias: float = 0.3
    interaction_memory: deque = field(default_factory=lambda: deque(maxlen=100))
    topic_exposure_counts: Counter = field(default_factory=Counter)
    author_interaction_history: dict[str, AuthorInteractionSummary] = field(
        default_factory=dict
    )
    opinion: float = 0.0
    opinion_confidence: float = 0.5

    def __post_init__(self):
        """Validate and clip values."""
        self.attention_budget = np.clip(self.attention_budget, 0.0, 1.0)
        self.emotional_valence = np.clip(self.emotional_valence, -1.0, 1.0)
        self.emotional_arousal = np.clip(self.emotional_arousal, 0.0, 1.0)
        self.current_confirmation_bias = np.clip(self.current_confirmation_bias, 0.0, 1.0)
        self.opinion = np.clip(self.opinion, -1.0, 1.0)
        self.opinion_confidence = np.clip(self.opinion_confidence, 0.0, 1.0)

    def deplete_attention(self, amount: float) -> None:
        """Deplete attention budget.

        Args:
            amount: Amount to deplete (0-1)
        """
        self.attention_budget = max(0.0, self.attention_budget - amount)

    def recover_attention(self, rate: float) -> None:
        """Recover attention budget.

        Args:
            rate: Recovery rate (0-1)
        """
        self.attention_budget = min(1.0, self.attention_budget + rate)

    def update_emotional_state(
        self,
        valence_delta: float,
        arousal_delta: float,
        decay_rate: float = 0.1,
    ) -> None:
        """Update emotional state with new input and decay.

        Args:
            valence_delta: Change in valence
            arousal_delta: Change in arousal
            decay_rate: Rate of decay toward neutral
        """
        # Apply input
        self.emotional_valence += valence_delta
        self.emotional_arousal += arousal_delta

        # Decay toward neutral
        self.emotional_valence *= (1 - decay_rate)
        self.emotional_arousal = 0.5 + (self.emotional_arousal - 0.5) * (1 - decay_rate)

        # Clip
        self.emotional_valence = np.clip(self.emotional_valence, -1.0, 1.0)
        self.emotional_arousal = np.clip(self.emotional_arousal, 0.0, 1.0)

    def add_interaction_memory(self, memory: InteractionMemory) -> None:
        """Add an interaction to memory.

        Args:
            memory: Interaction memory to add
        """
        self.interaction_memory.append(memory)

        # Update topic exposure
        for topic in memory.topics:
            self.topic_exposure_counts[topic] += 1

        # Update author interaction history
        if memory.author_id not in self.author_interaction_history:
            self.author_interaction_history[memory.author_id] = AuthorInteractionSummary(
                author_id=memory.author_id
            )

        summary = self.author_interaction_history[memory.author_id]
        summary.total_interactions += 1
        summary.last_interaction_step = memory.step

        if memory.emotional_impact > 0:
            summary.positive_interactions += 1
        elif memory.emotional_impact < 0:
            summary.negative_interactions += 1

        # Update running average sentiment
        n = summary.total_interactions
        summary.average_sentiment = (
            summary.average_sentiment * (n - 1) + memory.emotional_impact
        ) / n

    def update_opinion(
        self,
        influence: float,
        influence_weight: float = 0.1,
    ) -> None:
        """Update opinion based on new influence.

        Args:
            influence: Opinion influence from interaction (-1 to 1)
            influence_weight: Weight of new influence
        """
        # Update opinion (weighted average)
        self.opinion = (
            self.opinion * (1 - influence_weight) +
            influence * influence_weight
        )
        self.opinion = np.clip(self.opinion, -1.0, 1.0)

    def get_topic_novelty(self, topic: str) -> float:
        """Get novelty score for a topic.

        Args:
            topic: Topic ID

        Returns:
            Novelty score (0-1), higher for less seen topics
        """
        count = self.topic_exposure_counts.get(topic, 0)
        return 1.0 / (1.0 + np.log1p(count))

    def get_author_affinity(self, author_id: str) -> float:
        """Get affinity score for an author.

        Args:
            author_id: Author ID

        Returns:
            Affinity score (-1 to 1)
        """
        if author_id not in self.author_interaction_history:
            return 0.0

        summary = self.author_interaction_history[author_id]
        if summary.total_interactions == 0:
            return 0.0

        # Weighted by interaction count (diminishing returns)
        interaction_factor = 1 - np.exp(-summary.total_interactions / 10)
        return summary.average_sentiment * interaction_factor

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attention_budget": self.attention_budget,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "current_confirmation_bias": self.current_confirmation_bias,
            "interaction_memory": [m.to_dict() for m in self.interaction_memory],
            "topic_exposure_counts": dict(self.topic_exposure_counts),
            "author_interaction_history": {
                k: v.to_dict() for k, v in self.author_interaction_history.items()
            },
            "opinion": self.opinion,
            "opinion_confidence": self.opinion_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserCognitiveState":
        """Create from dictionary."""
        state = cls(
            attention_budget=data.get("attention_budget", 1.0),
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_arousal=data.get("emotional_arousal", 0.5),
            current_confirmation_bias=data.get("current_confirmation_bias", 0.3),
            opinion=data.get("opinion", 0.0),
            opinion_confidence=data.get("opinion_confidence", 0.5),
        )

        # Load interaction memory
        for mem_data in data.get("interaction_memory", []):
            state.interaction_memory.append(InteractionMemory.from_dict(mem_data))

        # Load topic exposure
        state.topic_exposure_counts = Counter(data.get("topic_exposure_counts", {}))

        # Load author history
        for author_id, summary_data in data.get("author_interaction_history", {}).items():
            state.author_interaction_history[author_id] = AuthorInteractionSummary.from_dict(
                summary_data
            )

        return state


@dataclass
class UserTraits:
    """Psychological and behavioral traits for a user.

    Attributes:
        ideology: Political/ideological leaning from -1 (left) to 1 (right)
        confirmation_bias: Tendency to engage with aligned content (0-1)
        misinfo_susceptibility: Likelihood to believe/share misinformation (0-1)
        emotional_reactivity: Response to emotional content (0-1)
        activity_level: Base activity rate (0-1)
        openness: Openness to differing opinions (0-1)
        conscientiousness: Task diligence / regulation tendency (0-1)
    """

    ideology: float = 0.0
    confirmation_bias: float = 0.3
    misinfo_susceptibility: float = 0.2
    emotional_reactivity: float = 0.5
    activity_level: float = 0.3
    openness: float = 0.5
    conscientiousness: float = 0.5

    def __post_init__(self) -> None:
        """Validate trait ranges."""
        self.ideology = np.clip(self.ideology, -1.0, 1.0)
        self.confirmation_bias = np.clip(self.confirmation_bias, 0.0, 1.0)
        self.misinfo_susceptibility = np.clip(self.misinfo_susceptibility, 0.0, 1.0)
        self.emotional_reactivity = np.clip(self.emotional_reactivity, 0.0, 1.0)
        self.activity_level = np.clip(self.activity_level, 0.0, 1.0)
        self.openness = np.clip(self.openness, 0.0, 1.0)
        self.conscientiousness = np.clip(self.conscientiousness, 0.0, 1.0)

    def to_dict(self) -> dict[str, float]:
        """Convert traits to dictionary."""
        return {
            "ideology": self.ideology,
            "confirmation_bias": self.confirmation_bias,
            "misinfo_susceptibility": self.misinfo_susceptibility,
            "emotional_reactivity": self.emotional_reactivity,
            "activity_level": self.activity_level,
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "UserTraits":
        """Create traits from dictionary."""
        return cls(**data)


@dataclass
class User:
    """Represents a synthetic user in the social network.

    Attributes:
        user_id: Unique identifier
        interests: Set of topic IDs the user is interested in
        interest_weights: Strength of interest per topic (0-1)
        traits: Psychological/behavioral traits
        influence_score: Network influence measure (computed from network position)
        credibility_score: Trustworthiness measure (0-1)
        state: Current user state (active, inactive, etc.)
        followers: Set of user IDs following this user
        following: Set of user IDs this user follows
        created_step: Simulation step when user was created
        last_active_step: Last step user was active
        total_posts: Total posts created
        total_interactions: Total interactions made
        metadata: Additional arbitrary data
    """

    user_id: str
    interests: set[str] = field(default_factory=set)
    interest_weights: dict[str, float] = field(default_factory=dict)
    traits: UserTraits = field(default_factory=UserTraits)
    influence_score: float = 0.0
    credibility_score: float = 0.5
    state: UserState = UserState.ACTIVE
    followers: set[str] = field(default_factory=set)
    following: set[str] = field(default_factory=set)
    created_step: int = 0
    last_active_step: int = 0
    total_posts: int = 0
    total_interactions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def follower_count(self) -> int:
        """Get number of followers."""
        return len(self.followers)

    @property
    def following_count(self) -> int:
        """Get number of users being followed."""
        return len(self.following)

    def add_follower(self, user_id: str) -> None:
        """Add a follower."""
        self.followers.add(user_id)

    def remove_follower(self, user_id: str) -> None:
        """Remove a follower."""
        self.followers.discard(user_id)

    def follow(self, user_id: str) -> None:
        """Follow another user."""
        self.following.add(user_id)

    def unfollow(self, user_id: str) -> None:
        """Unfollow a user."""
        self.following.discard(user_id)

    def add_interest(self, topic_id: str, weight: float = 0.5) -> None:
        """Add an interest with given weight."""
        self.interests.add(topic_id)
        self.interest_weights[topic_id] = np.clip(weight, 0.0, 1.0)

    def get_interest_weight(self, topic_id: str) -> float:
        """Get interest weight for a topic (0 if not interested)."""
        return self.interest_weights.get(topic_id, 0.0)

    def update_influence(self, score: float) -> None:
        """Update influence score."""
        self.influence_score = max(0.0, score)

    def update_credibility(self, delta: float) -> None:
        """Update credibility score."""
        self.credibility_score = np.clip(self.credibility_score + delta, 0.0, 1.0)

    def record_activity(self, step: int) -> None:
        """Record user activity at given step."""
        self.last_active_step = step

    def record_post(self, step: int) -> None:
        """Record a new post."""
        self.total_posts += 1
        self.record_activity(step)

    def record_interaction(self, step: int) -> None:
        """Record an interaction."""
        self.total_interactions += 1
        self.record_activity(step)

    def is_active(self) -> bool:
        """Check if user is active."""
        return self.state == UserState.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "interests": list(self.interests),
            "interest_weights": self.interest_weights,
            "traits": self.traits.to_dict(),
            "influence_score": self.influence_score,
            "credibility_score": self.credibility_score,
            "state": str(self.state),
            "followers": list(self.followers),
            "following": list(self.following),
            "created_step": self.created_step,
            "last_active_step": self.last_active_step,
            "total_posts": self.total_posts,
            "total_interactions": self.total_interactions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Create user from dictionary."""
        return cls(
            user_id=data["user_id"],
            interests=set(data.get("interests", [])),
            interest_weights=data.get("interest_weights", {}),
            traits=UserTraits.from_dict(data.get("traits", {})),
            influence_score=data.get("influence_score", 0.0),
            credibility_score=data.get("credibility_score", 0.5),
            state=UserState(data.get("state", "active")),
            followers=set(data.get("followers", [])),
            following=set(data.get("following", [])),
            created_step=data.get("created_step", 0),
            last_active_step=data.get("last_active_step", 0),
            total_posts=data.get("total_posts", 0),
            total_interactions=data.get("total_interactions", 0),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        return hash(self.user_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return NotImplemented
        return self.user_id == other.user_id
