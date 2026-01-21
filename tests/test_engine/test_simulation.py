"""Tests for simulation engine."""

import pytest

from config.schemas import SimulationConfig
from generator import World
from engine.simulation import Simulation


class TestSimulation:
    """Tests for Simulation class."""

    def test_simulation_initialization(self, small_world):
        """Test simulation initializes correctly."""
        sim = Simulation(small_world)

        assert sim.world is small_world
        assert sim.current_step == 0
        assert len(sim.posts) == 0
        assert len(sim.interactions) == 0

    def test_simulation_step(self, small_world):
        """Test running a single simulation step."""
        sim = Simulation(small_world)
        sim.step()

        assert sim.current_step == 1
        # Should have some posts and interactions
        assert len(sim.posts) > 0

    def test_simulation_run(self, small_config):
        """Test running full simulation."""
        small_config.num_steps = 5
        world = World(small_config)
        world.build()

        sim = Simulation(world)
        sim.run()

        assert sim.current_step == 5
        assert len(sim.posts) > 0

    def test_simulation_metrics(self, small_world):
        """Test metrics are computed during simulation."""
        sim = Simulation(small_world)
        sim.run(steps=3)

        metrics = sim.get_metrics_history()
        assert len(metrics) == 3

        # Check first step has expected keys
        first_step = metrics[0]
        assert "step" in first_step
        assert "total_posts" in first_step
        assert "total_interactions" in first_step

    def test_simulation_with_events(self, small_world):
        """Test simulation handles random events."""
        sim = Simulation(small_world)
        # Run enough steps that events might trigger
        sim.run(steps=10)

        # Just verify it completes without error
        assert sim.current_step == 10

    def test_get_post_by_id(self, small_world):
        """Test retrieving posts by ID."""
        sim = Simulation(small_world)
        sim.step()

        if sim.posts:
            post_id = list(sim.posts.keys())[0]
            post = sim.get_post(post_id)
            assert post is not None
            assert post.post_id == post_id

    def test_get_user_feed(self, small_world):
        """Test getting a user's feed."""
        sim = Simulation(small_world)
        sim.run(steps=3)

        user_id = list(small_world.users.keys())[0]
        feed = sim.get_user_feed(user_id)

        # Feed should be a list (may be empty)
        assert isinstance(feed, list)

    def test_reproducibility(self):
        """Test same seed produces same simulation results."""
        config1 = SimulationConfig(seed=123)
        config1.user.num_users = 30
        config1.num_steps = 5

        config2 = SimulationConfig(seed=123)
        config2.user.num_users = 30
        config2.num_steps = 5

        world1 = World(config1)
        world1.build()
        sim1 = Simulation(world1)
        sim1.run()

        world2 = World(config2)
        world2.build()
        sim2 = Simulation(world2)
        sim2.run()

        # Should have same number of posts
        assert len(sim1.posts) == len(sim2.posts)
