"""Pytest configuration and fixtures."""

import pytest
import numpy as np

from config.schemas import SimulationConfig
from generator import World


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture
def small_config(seed):
    """Small configuration for fast tests."""
    config = SimulationConfig(
        name="test",
        seed=seed,
        num_steps=10,
    )
    config.user.num_users = 50
    config.content.topics.num_topics = 10
    return config


@pytest.fixture
def medium_config(seed):
    """Medium configuration for integration tests."""
    config = SimulationConfig(
        name="test_medium",
        seed=seed,
        num_steps=20,
    )
    config.user.num_users = 200
    config.content.topics.num_topics = 20
    return config


@pytest.fixture
def small_world(small_config):
    """Pre-built small world."""
    world = World(small_config)
    world.build()
    return world


@pytest.fixture
def medium_world(medium_config):
    """Pre-built medium world."""
    world = World(medium_config)
    world.build()
    return world


@pytest.fixture
def rng(seed):
    """Seeded random number generator."""
    return np.random.default_rng(seed)
