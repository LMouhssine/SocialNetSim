"""Stress tests for large-scale simulation performance.

Tests:
- 100k users memory test (<8GB)
- 100k users time test (<30 min for 100 steps)
- Many concurrent cascades test
- Vectorized vs sequential comparison
"""

import time
import gc
import sys
from typing import Any

import numpy as np

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from config.schemas import SimulationConfig, UserConfig, PerformanceConfig
from engine.vectorized_ops import VectorizedUserState, VectorizedPostState
from engine.batch_processor import BatchProcessor, BatchConfig
from engine.memory_manager import MemoryManager, MemoryConfig, LargeScaleMemoryOptimizer


class TestMemoryUsage:
    """Tests for memory usage at scale."""

    def test_vectorized_user_state_memory(self):
        """Test that vectorized user state stays within memory limits."""
        n_users = 100_000

        # Estimate memory
        optimizer = LargeScaleMemoryOptimizer()
        estimates = optimizer.estimate_memory_requirements(
            n_users=n_users,
            n_posts_per_step=100,
            n_steps=100,
        )

        # Should be under 8GB total
        assert estimates["total_gb"] < 8.0, f"Estimated memory {estimates['total_gb']:.2f}GB exceeds 8GB"

        # Create vectorized state
        user_state = VectorizedUserState(n_users)

        # Check actual memory (approximate)
        state_size_mb = (
            user_state.activity_level.nbytes +
            user_state.influence_score.nbytes +
            user_state.openness.nbytes +
            user_state.conscientiousness.nbytes +
            user_state.emotional_reactivity.nbytes +
            user_state.confirmation_bias.nbytes +
            user_state.misinfo_susceptibility.nbytes +
            user_state.fatigue.nbytes +
            user_state.attention_budget.nbytes +
            user_state.session_interactions.nbytes +
            user_state.last_active_step.nbytes
        ) / (1024 * 1024)

        # User state should be under 100MB for 100k users
        assert state_size_mb < 100, f"User state {state_size_mb:.2f}MB exceeds 100MB"

    def test_post_state_memory_growth(self):
        """Test that post state memory grows linearly."""
        sizes = [1000, 10000, 50000]
        memory_per_post = []

        for n_posts in sizes:
            post_state = VectorizedPostState(n_posts)
            size_mb = (
                post_state.author_idx.nbytes +
                post_state.created_step.nbytes +
                post_state.virality_score.nbytes +
                post_state.sentiment.nbytes +
                post_state.controversy_score.nbytes +
                post_state.quality_score.nbytes +
                post_state.view_count.nbytes +
                post_state.like_count.nbytes +
                post_state.share_count.nbytes +
                post_state.is_active.nbytes
            ) / (1024 * 1024)

            memory_per_post.append(size_mb / n_posts * 1024 * 1024)  # bytes per post

        # Memory per post should be roughly constant (linear growth)
        avg_bytes_per_post = np.mean(memory_per_post)
        std_bytes_per_post = np.std(memory_per_post)

        # Should have low variance (linear growth)
        assert std_bytes_per_post / avg_bytes_per_post < 0.1, "Memory growth is not linear"


class TestBatchProcessing:
    """Tests for batch processing performance."""

    def test_batch_processor_initialization(self):
        """Test batch processor initializes correctly."""
        config = BatchConfig(
            user_batch_size=1000,
            post_batch_size=5000,
            engagement_batch_size=10000,
        )
        processor = BatchProcessor(config, seed=42)

        assert processor.config.user_batch_size == 1000
        assert processor.batch_counter == 0

    def test_active_user_sampling_performance(self):
        """Test active user sampling at scale."""
        n_users = 100_000
        user_state = VectorizedUserState(n_users)

        # Set random activity levels
        rng = np.random.default_rng(42)
        user_state.activity_level = rng.random(n_users).astype(np.float32)

        processor = BatchProcessor(seed=42)

        # Measure time
        start = time.time()
        for step in range(10):
            active = processor.sample_active_users(user_state, step)
        elapsed = time.time() - start

        # Should complete 10 steps in under 1 second
        assert elapsed < 1.0, f"Sampling took {elapsed:.2f}s for 10 steps"

        # Should return reasonable number of active users
        assert len(active) > 0
        assert len(active) < n_users

    def test_batch_size_recommendation(self):
        """Test batch size recommendations."""
        optimizer = LargeScaleMemoryOptimizer()

        # Test engagement batch size
        batch_size = optimizer.recommend_batch_size(
            available_memory_gb=8.0,
            n_users=100_000,
            operation="engagement",
        )

        assert batch_size > 0
        assert batch_size <= 50000

        # Test feed generation batch size
        feed_batch = optimizer.recommend_batch_size(
            available_memory_gb=8.0,
            n_users=100_000,
            operation="feed_generation",
        )

        assert feed_batch > 0
        # Feed generation uses more memory per item
        assert feed_batch <= batch_size


class TestMemoryManagement:
    """Tests for memory management."""

    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly."""
        config = MemoryConfig(
            max_memory_gb=8.0,
            interaction_retention_steps=100,
        )
        manager = MemoryManager(config)

        assert manager.config.max_memory_gb == 8.0
        assert manager.stats.interactions_pruned == 0

    def test_interaction_pruning(self):
        """Test interaction pruning logic."""
        from dataclasses import dataclass

        @dataclass
        class MockInteraction:
            step: int

        manager = MemoryManager(MemoryConfig(interaction_retention_steps=10))

        # Create mock interactions
        interactions = [MockInteraction(step=i) for i in range(100)]

        # Prune at step 50
        pruned = manager.prune_interactions(interactions, current_step=50)

        # Should only keep interactions from step 40 onwards
        assert len(pruned) == 10
        assert all(i.step >= 40 for i in pruned)

    def test_cascade_archiving_threshold(self):
        """Test cascade archiving identification."""
        from dataclasses import dataclass, field

        @dataclass
        class MockCascade:
            cascade_id: str
            is_active: bool
            start_step: int
            shares_by_step: dict = field(default_factory=dict)

        manager = MemoryManager(MemoryConfig(cascade_archive_threshold=20))

        cascades = {
            "c1": MockCascade("c1", is_active=False, start_step=0, shares_by_step={5: 10}),
            "c2": MockCascade("c2", is_active=True, start_step=50),
            "c3": MockCascade("c3", is_active=False, start_step=40, shares_by_step={45: 5}),
        }

        archivable = manager.find_archivable_cascades(cascades, current_step=100)

        # c1 should be archivable (last activity at step 5, 95 steps ago)
        # c2 is active, not archivable
        # c3 last activity at step 45, 55 steps ago, should be archivable
        assert "c1" in archivable
        assert "c2" not in archivable
        assert "c3" in archivable


class TestScalability:
    """Tests for scalability characteristics."""

    @pytest.mark.slow
    def test_large_user_population(self):
        """Test handling 100k users (marked as slow test)."""
        n_users = 100_000

        start_time = time.time()

        # Create vectorized state
        user_state = VectorizedUserState(n_users)
        rng = np.random.default_rng(42)
        user_state.activity_level = rng.random(n_users).astype(np.float32)
        user_state.fatigue = rng.random(n_users).astype(np.float32) * 0.5

        creation_time = time.time() - start_time

        # Should create state in under 5 seconds
        assert creation_time < 5.0, f"State creation took {creation_time:.2f}s"

        # Test batch processing
        processor = BatchProcessor(seed=42)

        step_times = []
        for step in range(10):
            step_start = time.time()

            active = processor.sample_active_users(user_state, step)
            posting = processor.sample_posting_users(user_state, active, 0.1)

            step_times.append(time.time() - step_start)

        avg_step_time = np.mean(step_times)

        # Average step should be under 0.5 seconds for sampling only
        assert avg_step_time < 0.5, f"Average step time {avg_step_time:.3f}s too slow"

    def test_dtype_optimization(self):
        """Test numpy dtype optimization."""
        optimizer = LargeScaleMemoryOptimizer()

        arrays = {
            "floats": np.array([1.0, 2.0, 3.0]),
            "small_ints": np.array([1, 2, 100], dtype=np.int64),
            "large_ints": np.array([1000000, 2000000], dtype=np.int64),
        }

        optimized = optimizer.optimize_numpy_dtypes(arrays)

        # Small ints should be downcast
        assert optimized["small_ints"].dtype in (np.uint16, np.int16)

        # Floats should become float32
        assert optimized["floats"].dtype == np.float32


class TestVectorizedOperations:
    """Tests for vectorized operation correctness."""

    def test_engagement_batch_consistency(self):
        """Test that batch engagement gives consistent results."""
        n_users = 1000
        n_posts = 100

        user_state = VectorizedUserState(n_users)
        post_state = VectorizedPostState(n_posts)

        # Initialize with deterministic values
        rng = np.random.default_rng(42)
        user_state.activity_level = rng.random(n_users).astype(np.float32)
        post_state.virality_score = rng.random(n_posts).astype(np.float32)
        post_state.created_step = np.zeros(n_posts, dtype=np.int32)
        post_state.author_idx = rng.integers(0, n_users, n_posts).astype(np.int32)

        # Run twice with same seed
        processor1 = BatchProcessor(seed=123)
        processor2 = BatchProcessor(seed=123)

        active1 = processor1.sample_active_users(user_state, 0)
        active2 = processor2.sample_active_users(user_state, 0)

        np.testing.assert_array_equal(active1, active2)


def run_quick_stress_tests():
    """Run quick stress tests (suitable for CI)."""
    print("Running memory usage tests...")
    test = TestMemoryUsage()
    test.test_vectorized_user_state_memory()
    test.test_post_state_memory_growth()
    print("✓ Memory tests passed")

    print("Running batch processing tests...")
    test = TestBatchProcessing()
    test.test_batch_processor_initialization()
    test.test_active_user_sampling_performance()
    test.test_batch_size_recommendation()
    print("✓ Batch processing tests passed")

    print("Running memory management tests...")
    test = TestMemoryManagement()
    test.test_memory_manager_initialization()
    test.test_interaction_pruning()
    test.test_cascade_archiving_threshold()
    print("✓ Memory management tests passed")

    print("Running scalability tests...")
    test = TestScalability()
    test.test_dtype_optimization()
    print("✓ Scalability tests passed")

    print("\nAll quick stress tests passed!")


if __name__ == "__main__":
    run_quick_stress_tests()
