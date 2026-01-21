"""World orchestrator for generating complete synthetic environments."""

from pathlib import Path
from typing import Any
import json

import numpy as np
import networkx as nx
from loguru import logger

from config.schemas import SimulationConfig
from models import User, Post
from .topic_generator import TopicGenerator, Topic
from .user_generator import UserGenerator
from .network_generator import NetworkGenerator
from .content_generator import ContentGenerator


class World:
    """Orchestrates generation of a complete synthetic social network world.

    This class coordinates all generators to create a coherent synthetic
    environment with users, topics, network connections, and content.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize world with configuration.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.seed = config.seed
        self.rng = np.random.default_rng(config.seed)

        # Generators (initialized during build)
        self.topic_generator: TopicGenerator | None = None
        self.user_generator: UserGenerator | None = None
        self.network_generator: NetworkGenerator | None = None
        self.content_generator: ContentGenerator | None = None

        # Generated data
        self.topics: dict[str, Topic] = {}
        self.users: dict[str, User] = {}
        self.graph: nx.DiGraph | None = None

        self._built = False

    def build(self) -> "World":
        """Build the complete synthetic world.

        Generates topics, users, network, and initializes content generator.

        Returns:
            Self for method chaining
        """
        logger.info(f"Building world: {self.config.name}")

        # 1. Generate topics
        logger.info("Generating topics...")
        self.topic_generator = TopicGenerator(
            self.config.content.topics,
            seed=self.seed,
        )
        self.topics = self.topic_generator.generate_topics()
        logger.info(f"Generated {len(self.topics)} topics")

        # 2. Generate users
        logger.info("Generating users...")
        self.user_generator = UserGenerator(
            self.config.user,
            self.topics,
            seed=self.seed,
        )
        self.users = self.user_generator.generate_users()
        logger.info(f"Generated {len(self.users)} users")

        # 3. Generate network
        logger.info("Generating social network...")
        self.network_generator = NetworkGenerator(
            self.config.network,
            self.users,
            seed=self.seed,
        )
        self.graph = self.network_generator.generate_network()
        logger.info(
            f"Generated network with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

        # 4. Initialize content generator
        self.content_generator = ContentGenerator(
            self.config.content,
            self.topics,
            seed=self.seed,
        )

        self._built = True
        logger.info("World build complete")

        return self

    def is_built(self) -> bool:
        """Check if world has been built."""
        return self._built

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_topic(self, topic_id: str) -> Topic | None:
        """Get a topic by ID."""
        return self.topics.get(topic_id)

    def get_followers(self, user_id: str) -> list[User]:
        """Get followers of a user."""
        if not self.network_generator:
            return []
        follower_ids = self.network_generator.get_followers(user_id)
        return [self.users[uid] for uid in follower_ids if uid in self.users]

    def get_following(self, user_id: str) -> list[User]:
        """Get users that a user follows."""
        if not self.network_generator:
            return []
        following_ids = self.network_generator.get_following(user_id)
        return [self.users[uid] for uid in following_ids if uid in self.users]

    def generate_post(
        self,
        author: User,
        step: int,
        forced_topics: set[str] | None = None,
        force_misinfo: bool | None = None,
    ) -> Post:
        """Generate a new post.

        Args:
            author: Post author
            step: Current simulation step
            forced_topics: Optional forced topics
            force_misinfo: Optional forced misinfo status

        Returns:
            Generated post
        """
        if not self.content_generator:
            raise RuntimeError("World not built - call build() first")

        return self.content_generator.generate_post(
            author, step, forced_topics, force_misinfo
        )

    def generate_share(
        self,
        original_post: Post,
        sharer: User,
        step: int,
    ) -> Post:
        """Generate a share of an existing post.

        Args:
            original_post: Post being shared
            sharer: User sharing
            step: Current step

        Returns:
            Share post
        """
        if not self.content_generator:
            raise RuntimeError("World not built - call build() first")

        return self.content_generator.generate_share_post(
            original_post, sharer, step
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive world statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "config_name": self.config.name,
            "built": self._built,
        }

        if not self._built:
            return stats

        # Topic stats
        stats["topics"] = {
            "total": len(self.topics),
            "by_category": {},
        }
        for topic in self.topics.values():
            cat = topic.category
            stats["topics"]["by_category"][cat] = stats["topics"]["by_category"].get(cat, 0) + 1

        # User stats
        if self.user_generator:
            stats["users"] = {
                "total": len(self.users),
                "trait_statistics": self.user_generator.get_trait_statistics(),
            }

        # Network stats
        if self.network_generator:
            stats["network"] = self.network_generator.get_network_statistics()

        return stats

    def save(self, path: str | Path) -> None:
        """Save world state to directory.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        # Save topics
        if self.topic_generator:
            with open(path / "topics.json", "w") as f:
                json.dump(self.topic_generator.to_dict(), f, indent=2)

        # Save users
        if self.user_generator:
            with open(path / "users.json", "w") as f:
                json.dump(self.user_generator.to_dict(), f, indent=2)

        # Save network
        if self.network_generator:
            with open(path / "network.json", "w") as f:
                json.dump(self.network_generator.to_dict(), f, indent=2)

        logger.info(f"World saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "World":
        """Load world state from directory.

        Args:
            path: Directory path to load from

        Returns:
            Loaded World instance
        """
        path = Path(path)

        # Load configuration
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = SimulationConfig(**config_dict)

        world = cls(config)

        # Load topics
        with open(path / "topics.json") as f:
            topics_data = json.load(f)
        world.topic_generator = TopicGenerator.from_dict(
            topics_data,
            config.content.topics,
            config.seed,
        )
        world.topics = world.topic_generator.topics

        # Load users
        with open(path / "users.json") as f:
            users_data = json.load(f)
        world.user_generator = UserGenerator.from_dict(
            users_data,
            config.user,
            world.topics,
            config.seed,
        )
        world.users = world.user_generator.users

        # Load network
        with open(path / "network.json") as f:
            network_data = json.load(f)
        world.network_generator = NetworkGenerator.from_dict(
            network_data,
            config.network,
            world.users,
            config.seed,
        )
        world.graph = world.network_generator.graph

        # Initialize content generator
        world.content_generator = ContentGenerator(
            config.content,
            world.topics,
            seed=config.seed,
        )

        world._built = True
        logger.info(f"World loaded from {path}")

        return world

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return f"World(name={self.config.name}, {status})"
