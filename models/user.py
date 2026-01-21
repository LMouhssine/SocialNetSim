"""User profile and traits models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .enums import UserState


@dataclass
class UserTraits:
    """Psychological and behavioral traits for a user.

    Attributes:
        ideology: Political/ideological leaning from -1 (left) to 1 (right)
        confirmation_bias: Tendency to engage with aligned content (0-1)
        misinfo_susceptibility: Likelihood to believe/share misinformation (0-1)
        emotional_reactivity: Response to emotional content (0-1)
        activity_level: Base activity rate (0-1)
    """

    ideology: float = 0.0
    confirmation_bias: float = 0.3
    misinfo_susceptibility: float = 0.2
    emotional_reactivity: float = 0.5
    activity_level: float = 0.3

    def __post_init__(self) -> None:
        """Validate trait ranges."""
        self.ideology = np.clip(self.ideology, -1.0, 1.0)
        self.confirmation_bias = np.clip(self.confirmation_bias, 0.0, 1.0)
        self.misinfo_susceptibility = np.clip(self.misinfo_susceptibility, 0.0, 1.0)
        self.emotional_reactivity = np.clip(self.emotional_reactivity, 0.0, 1.0)
        self.activity_level = np.clip(self.activity_level, 0.0, 1.0)

    def to_dict(self) -> dict[str, float]:
        """Convert traits to dictionary."""
        return {
            "ideology": self.ideology,
            "confirmation_bias": self.confirmation_bias,
            "misinfo_susceptibility": self.misinfo_susceptibility,
            "emotional_reactivity": self.emotional_reactivity,
            "activity_level": self.activity_level,
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
