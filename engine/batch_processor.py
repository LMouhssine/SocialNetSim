"""Batch processing for high-performance simulation.

Handles batched:
- Activity sampling
- Feed generation
- Engagement calculation
- Cascade processing
"""

from dataclasses import dataclass, field
from typing import Any, Iterator
import math

import numpy as np
from numpy.random import Generator

from .vectorized_ops import (
    VectorizedUserState,
    VectorizedPostState,
    VectorizedEngagement,
    VectorizedFeedRanking,
)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        user_batch_size: Users per batch
        post_batch_size: Posts per batch
        engagement_batch_size: Engagement calculations per batch
        feed_generation_batch_size: Feed generations per batch
        max_active_users_per_step: Maximum active users per step
        prefetch_posts: Number of posts to prefetch for feed
    """

    user_batch_size: int = 1000
    post_batch_size: int = 5000
    engagement_batch_size: int = 10000
    feed_generation_batch_size: int = 500
    max_active_users_per_step: int = 50000
    prefetch_posts: int = 100


@dataclass
class BatchResult:
    """Result from a batch operation.

    Attributes:
        batch_id: Batch identifier
        n_processed: Number of items processed
        results: Operation-specific results
        metrics: Performance metrics
    """

    batch_id: int
    n_processed: int
    results: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class BatchProcessor:
    """Processes simulation operations in batches for efficiency."""

    def __init__(
        self,
        config: BatchConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize batch processor.

        Args:
            config: Batch configuration
            seed: Random seed
        """
        self.config = config or BatchConfig()
        self.rng = np.random.default_rng(seed)

        # Vectorized components
        self.engagement_calc = VectorizedEngagement()
        self.feed_ranker = VectorizedFeedRanking()

        # Batch tracking
        self.batch_counter = 0
        self.total_processed = 0

    def iterate_user_batches(
        self,
        user_state: VectorizedUserState,
        shuffle: bool = True,
    ) -> Iterator[np.ndarray]:
        """Iterate over user batches.

        Args:
            user_state: Vectorized user state
            shuffle: Whether to shuffle users

        Yields:
            Arrays of user indices
        """
        indices = np.arange(user_state.n_users)
        if shuffle:
            self.rng.shuffle(indices)

        batch_size = self.config.user_batch_size
        for i in range(0, len(indices), batch_size):
            yield indices[i : i + batch_size]

    def sample_active_users(
        self,
        user_state: VectorizedUserState,
        current_step: int,
        event_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Sample active users for current step.

        Args:
            user_state: Vectorized user state
            current_step: Current simulation step
            event_multiplier: Event activity boost

        Returns:
            Array of active user indices
        """
        # Base activity probability from activity level
        base_prob = user_state.activity_level.copy()

        # Apply event multiplier
        base_prob *= event_multiplier

        # Fatigue penalty
        base_prob *= (1 - user_state.fatigue * 0.3)

        # Sample
        random_vals = self.rng.random(user_state.n_users)
        active_mask = random_vals < base_prob

        active_indices = np.where(active_mask)[0]

        # Cap at max active users
        if len(active_indices) > self.config.max_active_users_per_step:
            active_indices = self.rng.choice(
                active_indices,
                size=self.config.max_active_users_per_step,
                replace=False,
            )

        return active_indices

    def sample_posting_users(
        self,
        user_state: VectorizedUserState,
        active_users: np.ndarray,
        avg_posts_per_step: float,
    ) -> np.ndarray:
        """Sample users who will post this step.

        Args:
            user_state: Vectorized user state
            active_users: Active user indices
            avg_posts_per_step: Average posts per user per step

        Returns:
            Array of posting user indices
        """
        if len(active_users) == 0:
            return np.array([], dtype=np.int32)

        # Post probability
        post_prob = avg_posts_per_step * user_state.activity_level[active_users]

        # Sample
        random_vals = self.rng.random(len(active_users))
        posting_mask = random_vals < post_prob

        return active_users[posting_mask]

    def process_engagement_batch(
        self,
        user_indices: np.ndarray,
        post_indices: np.ndarray,
        user_state: VectorizedUserState,
        post_state: VectorizedPostState,
        current_step: int,
        event_multiplier: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Process engagement for batch of user-post pairs.

        Args:
            user_indices: User indices (n,)
            post_indices: Post indices (n,)
            user_state: Vectorized user state
            post_state: Vectorized post state
            current_step: Current step
            event_multiplier: Event effect multiplier

        Returns:
            Dictionary with engagement results
        """
        # Calculate probabilities
        probs = self.engagement_calc.compute_engagement_probabilities_batch(
            user_state,
            user_indices,
            post_state,
            post_indices,
            current_step,
            event_multiplier,
        )

        # Sample engagements
        random_vals = self.rng.random((4, len(user_indices)))

        viewed = random_vals[0] < probs["view"]
        liked = viewed & (random_vals[1] < probs["like"])
        shared = viewed & (random_vals[2] < probs["share"])
        commented = viewed & (random_vals[3] < probs["comment"])

        return {
            "user_indices": user_indices,
            "post_indices": post_indices,
            "viewed": viewed,
            "liked": liked,
            "shared": shared,
            "commented": commented,
            "view_prob": probs["view"],
            "like_prob": probs["like"],
            "share_prob": probs["share"],
            "comment_prob": probs["comment"],
        }

    def generate_feeds_batch(
        self,
        user_indices: np.ndarray,
        user_state: VectorizedUserState,
        post_state: VectorizedPostState,
        current_step: int,
        feed_size: int = 20,
        max_post_age: int = 50,
    ) -> list[np.ndarray]:
        """Generate feeds for batch of users.

        Args:
            user_indices: User indices
            user_state: Vectorized user state
            post_state: Vectorized post state
            current_step: Current step
            feed_size: Size of each feed
            max_post_age: Maximum post age to consider

        Returns:
            List of post index arrays (one per user)
        """
        # Get candidate posts (recent and active)
        min_step = max(0, current_step - max_post_age)
        candidate_mask = (
            post_state.is_active &
            (post_state.created_step >= min_step)
        )
        all_candidates = np.where(candidate_mask)[0]

        if len(all_candidates) == 0:
            return [np.array([], dtype=np.int32) for _ in user_indices]

        # For each user, get personalized candidates
        feeds = []
        for user_idx in user_indices:
            # Get posts from followed users
            followed_authors = user_state.following_matrix[user_idx].indices
            followed_posts_mask = np.isin(
                post_state.author_idx[all_candidates],
                followed_authors,
            )

            # Also include viral posts
            viral_mask = (
                (post_state.like_count[all_candidates] > 10) |
                (post_state.share_count[all_candidates] > 5)
            )

            combined_mask = followed_posts_mask | viral_mask

            # Random exploration
            explore_mask = self.rng.random(len(all_candidates)) < 0.2
            combined_mask = combined_mask | explore_mask

            user_candidates = all_candidates[combined_mask]

            if len(user_candidates) == 0:
                # Fall back to all candidates
                user_candidates = all_candidates

            # Rank and select
            ranked = self.feed_ranker.rank_posts_for_user(
                user_idx,
                user_state,
                user_candidates,
                post_state,
                current_step,
                feed_size,
            )
            feeds.append(ranked)

        return feeds

    def process_step_batched(
        self,
        user_state: VectorizedUserState,
        post_state: VectorizedPostState,
        current_step: int,
        avg_posts_per_step: float = 0.1,
        event_multiplier: float = 1.0,
        feed_size: int = 20,
    ) -> dict[str, Any]:
        """Process a complete simulation step using batched operations.

        Args:
            user_state: Vectorized user state
            post_state: Vectorized post state
            current_step: Current step
            avg_posts_per_step: Average posts per user
            event_multiplier: Event multiplier
            feed_size: Feed size

        Returns:
            Step results
        """
        results = {
            "step": current_step,
            "active_users": 0,
            "posting_users": 0,
            "views": 0,
            "likes": 0,
            "shares": 0,
            "comments": 0,
            "engagement_pairs_processed": 0,
        }

        # Sample active users
        active_users = self.sample_active_users(
            user_state, current_step, event_multiplier
        )
        results["active_users"] = len(active_users)

        if len(active_users) == 0:
            return results

        # Sample posting users
        posting_users = self.sample_posting_users(
            user_state, active_users, avg_posts_per_step
        )
        results["posting_users"] = len(posting_users)

        # Generate feeds for active users
        feeds = self.generate_feeds_batch(
            active_users,
            user_state,
            post_state,
            current_step,
            feed_size,
        )

        # Process engagements in batches
        all_user_indices = []
        all_post_indices = []

        for i, user_idx in enumerate(active_users):
            feed = feeds[i]
            if len(feed) > 0:
                all_user_indices.extend([user_idx] * len(feed))
                all_post_indices.extend(feed)

        if all_user_indices:
            user_arr = np.array(all_user_indices, dtype=np.int32)
            post_arr = np.array(all_post_indices, dtype=np.int32)

            # Process in batches
            batch_size = self.config.engagement_batch_size
            for start in range(0, len(user_arr), batch_size):
                end = min(start + batch_size, len(user_arr))

                batch_result = self.process_engagement_batch(
                    user_arr[start:end],
                    post_arr[start:end],
                    user_state,
                    post_state,
                    current_step,
                    event_multiplier,
                )

                results["views"] += batch_result["viewed"].sum()
                results["likes"] += batch_result["liked"].sum()
                results["shares"] += batch_result["shared"].sum()
                results["comments"] += batch_result["commented"].sum()
                results["engagement_pairs_processed"] += end - start

                # Update post counts
                for i, post_idx in enumerate(batch_result["post_indices"]):
                    if batch_result["viewed"][i]:
                        post_state.view_count[post_idx] += 1
                    if batch_result["liked"][i]:
                        post_state.like_count[post_idx] += 1
                    if batch_result["shared"][i]:
                        post_state.share_count[post_idx] += 1

                # Update user state
                engaged_users = batch_result["user_indices"][batch_result["viewed"]]
                user_state.session_interactions[engaged_users] += 1
                user_state.fatigue[engaged_users] += 0.02
                user_state.last_active_step[engaged_users] = current_step

        return results


class ParallelBatchProcessor:
    """Parallel batch processor using multiprocessing.

    Note: This is a placeholder for parallel processing.
    Full implementation would use multiprocessing or joblib.
    """

    def __init__(
        self,
        n_workers: int = 4,
        config: BatchConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize parallel batch processor.

        Args:
            n_workers: Number of worker processes
            config: Batch configuration
            seed: Random seed
        """
        self.n_workers = n_workers
        self.config = config or BatchConfig()
        self.base_seed = seed

        # For now, use single-threaded batch processor
        self.processor = BatchProcessor(config, seed)

    def process_step_parallel(
        self,
        user_state: VectorizedUserState,
        post_state: VectorizedPostState,
        current_step: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Process step with parallel batching.

        Currently falls back to single-threaded.
        Could be extended with multiprocessing.

        Args:
            user_state: Vectorized user state
            post_state: Vectorized post state
            current_step: Current step
            **kwargs: Additional arguments

        Returns:
            Step results
        """
        # For now, delegate to single-threaded processor
        return self.processor.process_step_batched(
            user_state, post_state, current_step, **kwargs
        )

    def partition_users(
        self,
        n_users: int,
    ) -> list[tuple[int, int]]:
        """Partition users across workers.

        Args:
            n_users: Total number of users

        Returns:
            List of (start, end) tuples
        """
        chunk_size = math.ceil(n_users / self.n_workers)
        partitions = []
        for i in range(self.n_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_users)
            if start < n_users:
                partitions.append((start, end))
        return partitions
