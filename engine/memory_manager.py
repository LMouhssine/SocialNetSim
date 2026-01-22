"""Memory management for large-scale simulations.

Handles:
- Pruning old interactions
- Archiving completed cascades
- Memory monitoring and pressure relief
- Incremental state cleanup
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from collections import deque
import gc
import sys

import numpy as np


@dataclass
class MemoryConfig:
    """Configuration for memory management.

    Attributes:
        max_memory_gb: Maximum memory usage in GB
        interaction_retention_steps: Steps to retain interactions
        cascade_archive_threshold: Steps after deactivation to archive
        prune_frequency: Steps between prune operations
        gc_frequency: Steps between garbage collection
        archive_path: Path for archived data (None = discard)
        enable_monitoring: Whether to track memory usage
    """

    max_memory_gb: float = 8.0
    interaction_retention_steps: int = 100
    cascade_archive_threshold: int = 50
    prune_frequency: int = 10
    gc_frequency: int = 50
    archive_path: str | None = None
    enable_monitoring: bool = True


@dataclass
class MemoryStats:
    """Memory usage statistics.

    Attributes:
        current_usage_mb: Current memory usage in MB
        peak_usage_mb: Peak memory usage in MB
        interactions_pruned: Total interactions pruned
        cascades_archived: Total cascades archived
        gc_collections: Number of GC collections triggered
        pressure_events: Number of memory pressure events
    """

    current_usage_mb: float = 0.0
    peak_usage_mb: float = 0.0
    interactions_pruned: int = 0
    cascades_archived: int = 0
    gc_collections: int = 0
    pressure_events: int = 0


@dataclass
class ArchivedCascade:
    """Archived cascade data.

    Attributes:
        cascade_id: Cascade identifier
        post_id: Associated post ID
        total_shares: Final share count
        total_reach: Final reach count
        max_depth: Maximum propagation depth
        peak_velocity: Peak sharing velocity
        start_step: When cascade started
        end_step: When cascade was archived
        depth_distribution: Distribution of depths
        summary_stats: Additional summary statistics
    """

    cascade_id: str
    post_id: str
    total_shares: int
    total_reach: int
    max_depth: int
    peak_velocity: float
    start_step: int
    end_step: int
    depth_distribution: dict[int, int] = field(default_factory=dict)
    summary_stats: dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """Manages memory for large-scale simulations.

    Provides:
    - Automatic pruning of old interactions
    - Cascade archiving and cleanup
    - Memory pressure monitoring
    - Garbage collection scheduling
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        archive_callback: Callable[[ArchivedCascade], None] | None = None,
    ):
        """Initialize memory manager.

        Args:
            config: Memory configuration
            archive_callback: Callback for archived cascades
        """
        self.config = config or MemoryConfig()
        self.archive_callback = archive_callback

        self.stats = MemoryStats()
        self.archived_cascades: list[ArchivedCascade] = []
        self.memory_history: deque[tuple[int, float]] = deque(maxlen=100)

        self._last_prune_step = 0
        self._last_gc_step = 0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        # Get size of all objects tracked by gc
        gc.collect()
        return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / (1024 * 1024)

    def check_memory_pressure(self) -> bool:
        """Check if memory is under pressure.

        Returns:
            True if memory usage exceeds threshold
        """
        if not self.config.enable_monitoring:
            return False

        current_mb = self.get_memory_usage_mb()
        self.stats.current_usage_mb = current_mb
        self.stats.peak_usage_mb = max(self.stats.peak_usage_mb, current_mb)

        max_mb = self.config.max_memory_gb * 1024
        return current_mb > max_mb * 0.9

    def should_prune(self, current_step: int) -> bool:
        """Check if pruning should occur.

        Args:
            current_step: Current simulation step

        Returns:
            True if pruning should happen
        """
        steps_since_prune = current_step - self._last_prune_step
        return steps_since_prune >= self.config.prune_frequency

    def should_gc(self, current_step: int) -> bool:
        """Check if garbage collection should occur.

        Args:
            current_step: Current simulation step

        Returns:
            True if GC should happen
        """
        steps_since_gc = current_step - self._last_gc_step
        return steps_since_gc >= self.config.gc_frequency

    def prune_interactions(
        self,
        interactions: list,
        current_step: int,
    ) -> list:
        """Prune old interactions.

        Args:
            interactions: List of interactions
            current_step: Current simulation step

        Returns:
            Pruned list of interactions
        """
        min_step = current_step - self.config.interaction_retention_steps

        original_count = len(interactions)
        pruned = [i for i in interactions if i.step >= min_step]

        pruned_count = original_count - len(pruned)
        self.stats.interactions_pruned += pruned_count
        self._last_prune_step = current_step

        return pruned

    def prune_interaction_index(
        self,
        interactions_by_step: dict[int, list],
        current_step: int,
    ) -> dict[int, list]:
        """Prune interaction index by step.

        Args:
            interactions_by_step: Index of interactions by step
            current_step: Current simulation step

        Returns:
            Pruned index
        """
        min_step = current_step - self.config.interaction_retention_steps

        steps_to_remove = [s for s in interactions_by_step.keys() if s < min_step]
        for step in steps_to_remove:
            pruned_count = len(interactions_by_step[step])
            del interactions_by_step[step]
            self.stats.interactions_pruned += pruned_count

        self._last_prune_step = current_step
        return interactions_by_step

    def archive_cascade(
        self,
        cascade,
        current_step: int,
    ) -> ArchivedCascade:
        """Archive a cascade for storage.

        Args:
            cascade: Cascade to archive
            current_step: Current simulation step

        Returns:
            Archived cascade data
        """
        archived = ArchivedCascade(
            cascade_id=cascade.cascade_id,
            post_id=cascade.post_id,
            total_shares=cascade.total_shares,
            total_reach=cascade.total_reach,
            max_depth=cascade.max_depth,
            peak_velocity=cascade.peak_velocity,
            start_step=cascade.start_step,
            end_step=current_step,
            depth_distribution=cascade.get_depth_distribution(),
            summary_stats={
                "branching_factor": cascade.get_branching_factor(),
                "duration": current_step - cascade.start_step,
            },
        )

        self.archived_cascades.append(archived)
        self.stats.cascades_archived += 1

        if self.archive_callback:
            self.archive_callback(archived)

        return archived

    def find_archivable_cascades(
        self,
        cascades: dict,
        current_step: int,
    ) -> list[str]:
        """Find cascades that can be archived.

        Args:
            cascades: Dictionary of cascades
            current_step: Current simulation step

        Returns:
            List of cascade IDs to archive
        """
        archivable = []

        for cascade_id, cascade in cascades.items():
            if not cascade.is_active:
                # Check how long since deactivation
                # Use last share step as proxy for deactivation time
                last_activity = cascade.start_step
                if cascade.shares_by_step:
                    last_activity = max(cascade.shares_by_step.keys())

                steps_inactive = current_step - last_activity
                if steps_inactive >= self.config.cascade_archive_threshold:
                    archivable.append(cascade_id)

        return archivable

    def cleanup_cascades(
        self,
        cascades: dict,
        posts: dict,
        current_step: int,
    ) -> tuple[dict, list[ArchivedCascade]]:
        """Archive and remove old cascades.

        Args:
            cascades: Dictionary of cascades
            posts: Dictionary of posts
            current_step: Current simulation step

        Returns:
            Tuple of (updated cascades dict, archived cascades)
        """
        archivable_ids = self.find_archivable_cascades(cascades, current_step)

        archived = []
        for cascade_id in archivable_ids:
            cascade = cascades[cascade_id]
            archived_cascade = self.archive_cascade(cascade, current_step)
            archived.append(archived_cascade)

            # Clear cascade reference from post
            if cascade.post_id in posts:
                posts[cascade.post_id].cascade_id = None

            del cascades[cascade_id]

        return cascades, archived

    def run_gc(self, current_step: int) -> int:
        """Run garbage collection.

        Args:
            current_step: Current simulation step

        Returns:
            Number of objects collected
        """
        collected = gc.collect()
        self.stats.gc_collections += 1
        self._last_gc_step = current_step

        return collected

    def manage_memory(
        self,
        state,
        current_step: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """Comprehensive memory management.

        Checks conditions and performs pruning/archiving/GC as needed.

        Args:
            state: Simulation state object
            current_step: Current simulation step
            force: Force all operations regardless of conditions

        Returns:
            Dictionary of operations performed
        """
        operations = {
            "pruned_interactions": 0,
            "archived_cascades": 0,
            "gc_collected": 0,
            "memory_pressure": False,
        }

        # Check memory pressure
        if self.check_memory_pressure():
            operations["memory_pressure"] = True
            self.stats.pressure_events += 1
            force = True

        # Prune interactions
        if force or self.should_prune(current_step):
            if hasattr(state, "_interactions_by_step"):
                state._interactions_by_step = self.prune_interaction_index(
                    state._interactions_by_step,
                    current_step,
                )

            if hasattr(state, "interactions"):
                original_count = len(state.interactions)
                state.interactions = self.prune_interactions(
                    state.interactions,
                    current_step,
                )
                operations["pruned_interactions"] = original_count - len(
                    state.interactions
                )

        # Archive cascades
        if hasattr(state, "cascades") and hasattr(state, "posts"):
            state.cascades, archived = self.cleanup_cascades(
                state.cascades,
                state.posts,
                current_step,
            )
            operations["archived_cascades"] = len(archived)

        # Run GC
        if force or self.should_gc(current_step):
            operations["gc_collected"] = self.run_gc(current_step)

        # Record memory history
        if self.config.enable_monitoring:
            current_mb = self.get_memory_usage_mb()
            self.memory_history.append((current_step, current_mb))

        return operations

    def get_stats(self) -> dict[str, Any]:
        """Get memory management statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "current_usage_mb": self.stats.current_usage_mb,
            "peak_usage_mb": self.stats.peak_usage_mb,
            "interactions_pruned": self.stats.interactions_pruned,
            "cascades_archived": self.stats.cascades_archived,
            "gc_collections": self.stats.gc_collections,
            "pressure_events": self.stats.pressure_events,
            "archived_cascade_count": len(self.archived_cascades),
        }

    def get_memory_trend(self) -> list[tuple[int, float]]:
        """Get memory usage trend.

        Returns:
            List of (step, memory_mb) tuples
        """
        return list(self.memory_history)


class IncrementalStateCompactor:
    """Compacts state incrementally to reduce memory.

    Provides methods to:
    - Compact user runtime states
    - Summarize interaction histories
    - Reduce cascade detail granularity
    """

    def __init__(self, retention_window: int = 50):
        """Initialize compactor.

        Args:
            retention_window: Steps to retain full detail
        """
        self.retention_window = retention_window

    def compact_user_runtime_state(
        self,
        runtime_state,
        current_step: int,
    ):
        """Compact a user's runtime state.

        Removes old history while preserving summary statistics.

        Args:
            runtime_state: User runtime state to compact
            current_step: Current simulation step
        """
        min_step = current_step - self.retention_window

        # Compact engagement history
        if hasattr(runtime_state, "engagement_history"):
            runtime_state.engagement_history = [
                e for e in runtime_state.engagement_history if e.get("step", 0) >= min_step
            ]

    def compact_cascade_history(
        self,
        cascade,
        current_step: int,
    ):
        """Compact cascade step-by-step history.

        Aggregates old step data into summary.

        Args:
            cascade: Cascade to compact
            current_step: Current simulation step
        """
        min_step = current_step - self.retention_window

        # Keep only recent step data, summarize the rest
        if hasattr(cascade, "shares_by_step"):
            old_total = sum(
                count
                for step, count in cascade.shares_by_step.items()
                if step < min_step
            )

            # Remove old entries
            cascade.shares_by_step = {
                step: count
                for step, count in cascade.shares_by_step.items()
                if step >= min_step
            }

            # Store summary
            if not hasattr(cascade, "archived_share_count"):
                cascade.archived_share_count = 0
            cascade.archived_share_count += old_total

    def compact_state(
        self,
        state,
        current_step: int,
    ) -> dict[str, int]:
        """Compact entire simulation state.

        Args:
            state: Simulation state
            current_step: Current simulation step

        Returns:
            Dictionary of compaction statistics
        """
        stats = {
            "users_compacted": 0,
            "cascades_compacted": 0,
        }

        # Compact user runtime states
        if hasattr(state, "runtime_states"):
            for runtime_state in state.runtime_states.values():
                self.compact_user_runtime_state(runtime_state, current_step)
                stats["users_compacted"] += 1

        # Compact cascades
        if hasattr(state, "cascades"):
            for cascade in state.cascades.values():
                self.compact_cascade_history(cascade, current_step)
                stats["cascades_compacted"] += 1

        return stats


class LargeScaleMemoryOptimizer:
    """Optimizations for 100k+ user simulations.

    Provides:
    - Sparse data structure recommendations
    - Batch processing memory estimation
    - Memory-efficient data layouts
    """

    @staticmethod
    def estimate_memory_requirements(
        n_users: int,
        n_posts_per_step: float,
        n_steps: int,
        interaction_rate: float = 0.1,
    ) -> dict[str, float]:
        """Estimate memory requirements for simulation.

        Args:
            n_users: Number of users
            n_posts_per_step: Average posts per step
            n_steps: Total simulation steps
            interaction_rate: Fraction of users interacting per step

        Returns:
            Dictionary of memory estimates in MB
        """
        # User state (traits, cognitive state, etc.)
        # ~500 bytes per user for numpy arrays
        user_state_mb = n_users * 500 / (1024 * 1024)

        # Network (sparse adjacency)
        # Assume average 150 connections per user
        avg_connections = 150
        network_mb = n_users * avg_connections * 8 / (1024 * 1024)

        # Posts
        total_posts = int(n_posts_per_step * n_steps)
        # ~200 bytes per post
        posts_mb = total_posts * 200 / (1024 * 1024)

        # Interactions (with retention window)
        retention_window = 100
        interactions_per_step = int(n_users * interaction_rate)
        total_interactions = interactions_per_step * min(n_steps, retention_window)
        # ~100 bytes per interaction
        interactions_mb = total_interactions * 100 / (1024 * 1024)

        # Cascades
        # ~1KB per active cascade, assume 100 active at any time
        cascades_mb = 100 * 1024 / (1024 * 1024)

        total_mb = (
            user_state_mb + network_mb + posts_mb + interactions_mb + cascades_mb
        )

        return {
            "user_state_mb": user_state_mb,
            "network_mb": network_mb,
            "posts_mb": posts_mb,
            "interactions_mb": interactions_mb,
            "cascades_mb": cascades_mb,
            "total_mb": total_mb,
            "total_gb": total_mb / 1024,
        }

    @staticmethod
    def recommend_batch_size(
        available_memory_gb: float,
        n_users: int,
        operation: str = "engagement",
    ) -> int:
        """Recommend batch size for given memory constraints.

        Args:
            available_memory_gb: Available memory in GB
            n_users: Total number of users
            operation: Type of operation

        Returns:
            Recommended batch size
        """
        available_mb = available_memory_gb * 1024

        # Reserve 50% for base state
        working_mb = available_mb * 0.5

        if operation == "engagement":
            # ~1KB per user-post pair for engagement calculation
            bytes_per_item = 1024
        elif operation == "feed_generation":
            # ~5KB per user for feed generation
            bytes_per_item = 5 * 1024
        else:
            bytes_per_item = 1024

        max_batch = int(working_mb * 1024 * 1024 / bytes_per_item)

        # Cap at reasonable limits
        return min(max_batch, n_users, 50000)

    @staticmethod
    def optimize_numpy_dtypes(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Optimize numpy array dtypes for memory efficiency.

        Args:
            arrays: Dictionary of named arrays

        Returns:
            Dictionary with optimized arrays
        """
        optimized = {}

        for name, arr in arrays.items():
            if arr.dtype == np.float64:
                # Check if float32 is sufficient
                if np.allclose(arr, arr.astype(np.float32)):
                    optimized[name] = arr.astype(np.float32)
                else:
                    optimized[name] = arr
            elif arr.dtype == np.int64:
                # Check if smaller int type works
                min_val, max_val = arr.min(), arr.max()
                if min_val >= 0 and max_val < 65536:
                    optimized[name] = arr.astype(np.uint16)
                elif min_val >= -32768 and max_val < 32768:
                    optimized[name] = arr.astype(np.int16)
                elif min_val >= 0 and max_val < 4294967296:
                    optimized[name] = arr.astype(np.uint32)
                else:
                    optimized[name] = arr.astype(np.int32)
            else:
                optimized[name] = arr

        return optimized
