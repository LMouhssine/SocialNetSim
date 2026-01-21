"""Tests for world generation."""

import pytest
import numpy as np

from config.schemas import SimulationConfig
from generator import World


class TestWorld:
    """Tests for World class."""

    def test_build_creates_topics(self, small_config):
        """Test world build creates topics."""
        world = World(small_config)
        world.build()

        assert len(world.topics) == small_config.content.topics.num_topics

    def test_build_creates_users(self, small_config):
        """Test world build creates users."""
        world = World(small_config)
        world.build()

        assert len(world.users) == small_config.user.num_users

    def test_build_creates_network(self, small_config):
        """Test world build creates network."""
        world = World(small_config)
        world.build()

        assert world.graph is not None
        assert world.graph.number_of_nodes() == small_config.user.num_users
        assert world.graph.number_of_edges() > 0

    def test_users_have_interests(self, small_world):
        """Test all users have interests."""
        for user in small_world.users.values():
            assert len(user.interests) > 0

    def test_users_have_followers(self, small_world):
        """Test most users have followers."""
        users_with_followers = sum(
            1 for u in small_world.users.values()
            if len(u.followers) > 0
        )
        # Most users should have at least one follower
        assert users_with_followers > len(small_world.users) * 0.8

    def test_reproducibility(self):
        """Test same seed produces same world."""
        config1 = SimulationConfig(seed=42)
        config1.user.num_users = 50

        config2 = SimulationConfig(seed=42)
        config2.user.num_users = 50

        world1 = World(config1)
        world1.build()

        world2 = World(config2)
        world2.build()

        # Same number of edges
        assert world1.graph.number_of_edges() == world2.graph.number_of_edges()

        # Same user traits
        for user_id in world1.users:
            u1 = world1.users[user_id]
            u2 = world2.users[user_id]
            assert u1.traits.ideology == u2.traits.ideology

    def test_is_built_flag(self, small_config):
        """Test is_built flag."""
        world = World(small_config)
        assert not world.is_built()

        world.build()
        assert world.is_built()

    def test_get_followers(self, small_world):
        """Test get_followers method."""
        # Pick a user with followers
        for user in small_world.users.values():
            if len(user.followers) > 0:
                followers = small_world.get_followers(user.user_id)
                assert len(followers) == len(user.followers)
                break

    def test_generate_post(self, small_world):
        """Test post generation."""
        user = list(small_world.users.values())[0]
        post = small_world.generate_post(user, step=0)

        assert post is not None
        assert post.author_id == user.user_id
        assert len(post.content.topics) > 0

    def test_statistics(self, small_world):
        """Test get_statistics method."""
        stats = small_world.get_statistics()

        assert "topics" in stats
        assert "users" in stats
        assert "network" in stats
        assert stats["topics"]["total"] > 0
        assert stats["users"]["total"] == len(small_world.users)
