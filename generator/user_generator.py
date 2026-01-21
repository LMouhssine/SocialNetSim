"""Synthetic user generation."""

from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import UserConfig
from models import User, UserTraits
from .distributions import sample_from_config
from .topic_generator import Topic


class UserGenerator:
    """Generates synthetic users with configurable trait distributions."""

    def __init__(
        self,
        config: UserConfig,
        topics: dict[str, Topic],
        seed: int | None = None,
    ):
        """Initialize user generator.

        Args:
            config: User generation configuration
            topics: Available topics for interest assignment
            seed: Random seed for reproducibility
        """
        self.config = config
        self.topics = topics
        self.rng = np.random.default_rng(seed)
        self.users: dict[str, User] = {}

    def generate_users(self, num_users: int | None = None) -> dict[str, User]:
        """Generate all users based on configuration.

        Args:
            num_users: Number of users to generate (defaults to config value)

        Returns:
            Dictionary mapping user_id to User objects
        """
        n = num_users or self.config.num_users

        for i in range(n):
            user = self._generate_user(i)
            self.users[user.user_id] = user

        return self.users

    def _generate_user(self, index: int) -> User:
        """Generate a single user.

        Args:
            index: User index

        Returns:
            Generated User
        """
        user_id = f"user_{index:06d}"

        # Generate traits
        traits = self._generate_traits()

        # Generate interests
        interests, interest_weights = self._generate_interests(traits)

        # Initial influence and credibility
        influence_score = 0.0  # Will be computed from network
        credibility_score = float(self.rng.beta(5, 2))  # Most users start credible

        return User(
            user_id=user_id,
            interests=interests,
            interest_weights=interest_weights,
            traits=traits,
            influence_score=influence_score,
            credibility_score=credibility_score,
            created_step=0,
        )

    def _generate_traits(self) -> UserTraits:
        """Generate user traits based on configuration distributions."""
        traits_config = self.config.traits

        ideology = float(sample_from_config(self.rng, traits_config.ideology))
        confirmation_bias = float(sample_from_config(self.rng, traits_config.confirmation_bias))
        misinfo_susceptibility = float(sample_from_config(self.rng, traits_config.misinfo_susceptibility))
        emotional_reactivity = float(sample_from_config(self.rng, traits_config.emotional_reactivity))
        activity_level = float(sample_from_config(self.rng, traits_config.activity_level))

        return UserTraits(
            ideology=ideology,
            confirmation_bias=confirmation_bias,
            misinfo_susceptibility=misinfo_susceptibility,
            emotional_reactivity=emotional_reactivity,
            activity_level=activity_level,
        )

    def _generate_interests(
        self,
        traits: UserTraits,
    ) -> tuple[set[str], dict[str, float]]:
        """Generate user interests based on traits.

        Users with stronger ideology tend to have interests in political topics.
        Interest weights are influenced by topic popularity and user traits.

        Args:
            traits: User traits

        Returns:
            Tuple of (interest set, interest weights dict)
        """
        min_interests, max_interests = self.config.interests_per_user
        n_interests = self.rng.integers(min_interests, max_interests + 1)

        topic_list = list(self.topics.values())

        # Weight topics by popularity and ideology alignment
        weights = []
        for topic in topic_list:
            weight = topic.popularity

            # Users with strong ideology prefer aligned topics
            if abs(traits.ideology) > 0.3:
                ideology_match = 1 - abs(topic.ideology_bias - traits.ideology) / 2
                weight *= 0.5 + 0.5 * ideology_match

            # Emotional users prefer controversial topics
            if traits.emotional_reactivity > 0.5:
                weight *= 1 + 0.5 * topic.controversy_score

            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample topics
        selected_indices = self.rng.choice(
            len(topic_list),
            size=min(n_interests, len(topic_list)),
            replace=False,
            p=weights,
        )

        interests = set()
        interest_weights = {}

        for idx in selected_indices:
            topic = topic_list[idx]
            interests.add(topic.topic_id)

            # Interest weight based on topic popularity and some randomness
            base_weight = topic.popularity
            noise = self.rng.uniform(0.5, 1.5)
            interest_weights[topic.topic_id] = min(1.0, base_weight * noise)

        return interests, interest_weights

    def generate_single_user(self, user_id: str | None = None) -> User:
        """Generate a single new user.

        Args:
            user_id: Optional specific user ID

        Returns:
            Generated User
        """
        if user_id is None:
            index = len(self.users)
            user_id = f"user_{index:06d}"

        user = self._generate_user(len(self.users))
        user.user_id = user_id  # Override if custom ID provided
        self.users[user.user_id] = user
        return user

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_users_by_ideology_range(
        self,
        min_ideology: float,
        max_ideology: float,
    ) -> list[User]:
        """Get users within an ideology range."""
        return [
            u for u in self.users.values()
            if min_ideology <= u.traits.ideology <= max_ideology
        ]

    def get_active_users(self) -> list[User]:
        """Get all active users."""
        return [u for u in self.users.values() if u.is_active()]

    def get_users_interested_in(self, topic_id: str) -> list[User]:
        """Get users interested in a specific topic."""
        return [u for u in self.users.values() if topic_id in u.interests]

    def sample_users(
        self,
        n: int,
        weight_by_activity: bool = True,
    ) -> list[User]:
        """Sample n users, optionally weighted by activity level.

        Args:
            n: Number of users to sample
            weight_by_activity: If True, more active users are more likely

        Returns:
            List of sampled users
        """
        user_list = list(self.users.values())

        if weight_by_activity:
            weights = np.array([u.traits.activity_level for u in user_list])
            # Add small epsilon to avoid zero weights
            weights = weights + 0.01
            weights = weights / weights.sum()
        else:
            weights = None

        indices = self.rng.choice(
            len(user_list),
            size=min(n, len(user_list)),
            replace=False,
            p=weights,
        )

        return [user_list[i] for i in indices]

    def get_trait_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics about trait distributions."""
        if not self.users:
            return {}

        traits_data = {
            "ideology": [],
            "confirmation_bias": [],
            "misinfo_susceptibility": [],
            "emotional_reactivity": [],
            "activity_level": [],
        }

        for user in self.users.values():
            traits_data["ideology"].append(user.traits.ideology)
            traits_data["confirmation_bias"].append(user.traits.confirmation_bias)
            traits_data["misinfo_susceptibility"].append(user.traits.misinfo_susceptibility)
            traits_data["emotional_reactivity"].append(user.traits.emotional_reactivity)
            traits_data["activity_level"].append(user.traits.activity_level)

        stats = {}
        for trait_name, values in traits_data.items():
            arr = np.array(values)
            stats[trait_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert all users to dictionary."""
        return {
            user_id: user.to_dict()
            for user_id, user in self.users.items()
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: UserConfig,
        topics: dict[str, Topic],
        seed: int | None = None,
    ) -> "UserGenerator":
        """Create generator from saved data."""
        generator = cls(config, topics, seed)
        generator.users = {
            user_id: User.from_dict(user_data)
            for user_id, user_data in data.items()
        }
        return generator
