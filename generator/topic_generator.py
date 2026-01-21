"""Topic and community generation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import TopicConfig
from .distributions import sample_power_law, sample_from_config


# Pre-defined topic categories for synthetic generation
TOPIC_CATEGORIES = [
    "politics",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health",
    "business",
    "lifestyle",
    "education",
    "environment",
]

# Topic name templates per category
TOPIC_TEMPLATES = {
    "politics": ["election", "policy", "government", "legislation", "diplomacy"],
    "technology": ["ai", "software", "gadgets", "cybersecurity", "innovation"],
    "entertainment": ["movies", "music", "gaming", "celebrities", "streaming"],
    "sports": ["football", "basketball", "soccer", "olympics", "fitness"],
    "science": ["research", "space", "physics", "biology", "climate"],
    "health": ["medicine", "wellness", "nutrition", "mental_health", "vaccines"],
    "business": ["markets", "startups", "economy", "crypto", "investing"],
    "lifestyle": ["travel", "food", "fashion", "relationships", "home"],
    "education": ["learning", "universities", "skills", "careers", "books"],
    "environment": ["sustainability", "conservation", "energy", "wildlife", "pollution"],
}


@dataclass
class Topic:
    """Represents a topic/community in the network.

    Attributes:
        topic_id: Unique identifier
        name: Human-readable name
        category: Parent category
        popularity: Base popularity score (0-1)
        controversy_score: How controversial the topic is (0-1)
        ideology_bias: Ideological leaning of typical content (-1 to 1)
        volatility: How much engagement varies (0-1)
        related_topics: IDs of related topics
        metadata: Additional data
    """

    topic_id: str
    name: str
    category: str
    popularity: float = 0.5
    controversy_score: float = 0.0
    ideology_bias: float = 0.0
    volatility: float = 0.3
    related_topics: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert topic to dictionary."""
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "category": self.category,
            "popularity": self.popularity,
            "controversy_score": self.controversy_score,
            "ideology_bias": self.ideology_bias,
            "volatility": self.volatility,
            "related_topics": list(self.related_topics),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Topic":
        """Create topic from dictionary."""
        return cls(
            topic_id=data["topic_id"],
            name=data["name"],
            category=data["category"],
            popularity=data.get("popularity", 0.5),
            controversy_score=data.get("controversy_score", 0.0),
            ideology_bias=data.get("ideology_bias", 0.0),
            volatility=data.get("volatility", 0.3),
            related_topics=set(data.get("related_topics", [])),
            metadata=data.get("metadata", {}),
        )


class TopicGenerator:
    """Generates synthetic topics and communities."""

    def __init__(self, config: TopicConfig, seed: int | None = None):
        """Initialize topic generator.

        Args:
            config: Topic generation configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.topics: dict[str, Topic] = {}

    def generate_topics(self) -> dict[str, Topic]:
        """Generate all topics based on configuration.

        Returns:
            Dictionary mapping topic_id to Topic objects
        """
        num_topics = self.config.num_topics
        topics_per_category = num_topics // len(TOPIC_CATEGORIES)
        extra_topics = num_topics % len(TOPIC_CATEGORIES)

        topic_count = 0

        for i, category in enumerate(TOPIC_CATEGORIES):
            # Distribute extra topics among first few categories
            n_topics = topics_per_category + (1 if i < extra_topics else 0)

            for j in range(n_topics):
                topic = self._generate_topic(category, topic_count)
                self.topics[topic.topic_id] = topic
                topic_count += 1

        # Generate related topics connections
        self._generate_topic_relations()

        return self.topics

    def _generate_topic(self, category: str, index: int) -> Topic:
        """Generate a single topic.

        Args:
            category: Topic category
            index: Topic index

        Returns:
            Generated Topic
        """
        # Generate topic name from templates
        templates = TOPIC_TEMPLATES.get(category, ["topic"])
        template_idx = index % len(templates)
        name = f"{category}_{templates[template_idx]}_{index}"

        topic_id = f"topic_{index:04d}"

        # Generate popularity using power law (few popular, many niche)
        popularity = self._generate_popularity()

        # Generate controversy score
        controversy = float(sample_from_config(self.rng, self.config.controversy_distribution))

        # Politics and some other topics tend to be more ideological
        if category == "politics":
            ideology_bias = float(self.rng.normal(0, 0.4))
            controversy = min(1.0, controversy + 0.2)
        elif category in ("business", "environment"):
            ideology_bias = float(self.rng.normal(0, 0.2))
        else:
            ideology_bias = float(self.rng.normal(0, 0.1))

        ideology_bias = np.clip(ideology_bias, -1.0, 1.0)

        # Volatility - how much engagement varies
        volatility = float(self.rng.beta(2, 3))

        return Topic(
            topic_id=topic_id,
            name=name,
            category=category,
            popularity=popularity,
            controversy_score=controversy,
            ideology_bias=ideology_bias,
            volatility=volatility,
        )

    def _generate_popularity(self) -> float:
        """Generate topic popularity using power law distribution."""
        # Power law gives few very popular topics, many niche ones
        raw = float(sample_power_law(
            self.rng,
            alpha=self.config.topic_popularity_alpha,
            min_val=0.01,
            max_val=10.0,
        ))

        # Normalize to 0-1 range
        return min(1.0, raw / 10.0)

    def _generate_topic_relations(self) -> None:
        """Generate relationships between topics."""
        topic_list = list(self.topics.values())

        for topic in topic_list:
            # Topics in same category are related
            same_category = [
                t for t in topic_list
                if t.category == topic.category and t.topic_id != topic.topic_id
            ]

            # Add 1-3 related topics from same category
            n_related = min(len(same_category), self.rng.integers(1, 4))
            if same_category:
                related_indices = self.rng.choice(
                    len(same_category), size=n_related, replace=False
                )
                for idx in related_indices:
                    related_topic = same_category[idx]
                    topic.related_topics.add(related_topic.topic_id)
                    related_topic.related_topics.add(topic.topic_id)

            # Occasionally add cross-category relations
            if self.rng.random() < 0.2:
                other_topics = [
                    t for t in topic_list
                    if t.category != topic.category and t.topic_id != topic.topic_id
                ]
                if other_topics:
                    # Use index-based selection to avoid numpy object array issues
                    cross_topic = other_topics[self.rng.integers(0, len(other_topics))]
                    topic.related_topics.add(cross_topic.topic_id)
                    cross_topic.related_topics.add(topic.topic_id)

    def get_topic(self, topic_id: str) -> Topic | None:
        """Get a topic by ID."""
        return self.topics.get(topic_id)

    def get_topics_by_category(self, category: str) -> list[Topic]:
        """Get all topics in a category."""
        return [t for t in self.topics.values() if t.category == category]

    def get_popular_topics(self, n: int = 10) -> list[Topic]:
        """Get the n most popular topics."""
        sorted_topics = sorted(
            self.topics.values(),
            key=lambda t: t.popularity,
            reverse=True,
        )
        return sorted_topics[:n]

    def get_controversial_topics(self, n: int = 10) -> list[Topic]:
        """Get the n most controversial topics."""
        sorted_topics = sorted(
            self.topics.values(),
            key=lambda t: t.controversy_score,
            reverse=True,
        )
        return sorted_topics[:n]

    def sample_topics(
        self,
        n: int,
        weight_by_popularity: bool = True,
    ) -> list[Topic]:
        """Sample n topics, optionally weighted by popularity.

        Args:
            n: Number of topics to sample
            weight_by_popularity: If True, more popular topics are more likely

        Returns:
            List of sampled topics
        """
        topic_list = list(self.topics.values())

        if weight_by_popularity:
            weights = np.array([t.popularity for t in topic_list])
            weights = weights / weights.sum()
        else:
            weights = None

        indices = self.rng.choice(
            len(topic_list),
            size=min(n, len(topic_list)),
            replace=False,
            p=weights,
        )

        return [topic_list[i] for i in indices]

    def to_dict(self) -> dict[str, Any]:
        """Convert all topics to dictionary."""
        return {
            topic_id: topic.to_dict()
            for topic_id, topic in self.topics.items()
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: TopicConfig,
        seed: int | None = None,
    ) -> "TopicGenerator":
        """Create generator from saved data."""
        generator = cls(config, seed)
        generator.topics = {
            topic_id: Topic.from_dict(topic_data)
            for topic_id, topic_data in data.items()
        }
        return generator
