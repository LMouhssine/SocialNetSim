"""Vectorized operations for high-performance simulation.

Provides NumPy-based vectorized implementations for:
- User state management
- Engagement calculation
- Feed ranking
- Cascade processing

Optimized for 100k+ users with minimal Python overhead.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


@dataclass
class VectorizedUserState:
    """Vectorized user state arrays for efficient computation.

    All arrays are indexed by user_idx (0 to n_users-1).
    Maintains mapping between user_id and user_idx.

    Attributes:
        n_users: Number of users
        user_ids: Array of user IDs
        user_id_to_idx: Mapping from user_id to array index
        traits: User trait arrays
        fatigue: Current fatigue levels
        attention_budget: Current attention budgets
        opinions: User opinions
        last_active_step: Last active step per user
    """

    n_users: int
    user_ids: np.ndarray
    user_id_to_idx: dict[str, int] = field(default_factory=dict)

    # Trait arrays (immutable during simulation)
    ideology: np.ndarray = field(default=None)
    confirmation_bias: np.ndarray = field(default=None)
    misinfo_susceptibility: np.ndarray = field(default=None)
    emotional_reactivity: np.ndarray = field(default=None)
    activity_level: np.ndarray = field(default=None)

    # Dynamic state arrays
    fatigue: np.ndarray = field(default=None)
    attention_budget: np.ndarray = field(default=None)
    opinions: np.ndarray = field(default=None)
    emotional_valence: np.ndarray = field(default=None)
    emotional_arousal: np.ndarray = field(default=None)
    last_active_step: np.ndarray = field(default=None)
    session_interactions: np.ndarray = field(default=None)

    # Network (sparse adjacency)
    follower_matrix: sparse.csr_matrix = field(default=None)
    following_matrix: sparse.csr_matrix = field(default=None)

    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.ideology is None:
            self.ideology = np.zeros(self.n_users, dtype=np.float32)
        if self.confirmation_bias is None:
            self.confirmation_bias = np.zeros(self.n_users, dtype=np.float32)
        if self.misinfo_susceptibility is None:
            self.misinfo_susceptibility = np.zeros(self.n_users, dtype=np.float32)
        if self.emotional_reactivity is None:
            self.emotional_reactivity = np.zeros(self.n_users, dtype=np.float32)
        if self.activity_level is None:
            self.activity_level = np.ones(self.n_users, dtype=np.float32) * 0.3
        if self.fatigue is None:
            self.fatigue = np.zeros(self.n_users, dtype=np.float32)
        if self.attention_budget is None:
            self.attention_budget = np.ones(self.n_users, dtype=np.float32)
        if self.opinions is None:
            self.opinions = np.zeros(self.n_users, dtype=np.float32)
        if self.emotional_valence is None:
            self.emotional_valence = np.zeros(self.n_users, dtype=np.float32)
        if self.emotional_arousal is None:
            self.emotional_arousal = np.ones(self.n_users, dtype=np.float32) * 0.5
        if self.last_active_step is None:
            self.last_active_step = np.zeros(self.n_users, dtype=np.int32)
        if self.session_interactions is None:
            self.session_interactions = np.zeros(self.n_users, dtype=np.int32)

    @classmethod
    def from_users(cls, users: dict[str, Any]) -> "VectorizedUserState":
        """Create vectorized state from user dictionary.

        Args:
            users: Dictionary of user_id -> User objects

        Returns:
            VectorizedUserState instance
        """
        n_users = len(users)
        user_ids = np.array(list(users.keys()))
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        state = cls(
            n_users=n_users,
            user_ids=user_ids,
            user_id_to_idx=user_id_to_idx,
        )

        # Fill trait arrays
        for uid, user in users.items():
            idx = user_id_to_idx[uid]
            state.ideology[idx] = user.traits.ideology
            state.confirmation_bias[idx] = user.traits.confirmation_bias
            state.misinfo_susceptibility[idx] = user.traits.misinfo_susceptibility
            state.emotional_reactivity[idx] = user.traits.emotional_reactivity
            state.activity_level[idx] = user.traits.activity_level
            state.opinions[idx] = user.traits.ideology  # Initialize opinion from ideology

        # Build follower/following sparse matrices
        rows, cols = [], []
        for uid, user in users.items():
            i = user_id_to_idx[uid]
            for follower_id in user.followers:
                if follower_id in user_id_to_idx:
                    j = user_id_to_idx[follower_id]
                    rows.append(j)  # follower
                    cols.append(i)  # followed

        data = np.ones(len(rows), dtype=np.float32)
        state.following_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_users),
            dtype=np.float32,
        )
        state.follower_matrix = state.following_matrix.T.tocsr()

        return state

    def get_idx(self, user_id: str) -> int | None:
        """Get array index for user ID."""
        return self.user_id_to_idx.get(user_id)

    def get_user_id(self, idx: int) -> str:
        """Get user ID for array index."""
        return self.user_ids[idx]

    def recover_fatigue(self, recovery_rate: float) -> None:
        """Recover fatigue for all users."""
        self.fatigue = np.maximum(0.0, self.fatigue - recovery_rate)

    def recover_attention(self, recovery_rate: float) -> None:
        """Recover attention for all users."""
        self.attention_budget = np.minimum(1.0, self.attention_budget + recovery_rate)

    def reset_sessions(self) -> None:
        """Reset session interaction counts."""
        self.session_interactions.fill(0)

    def decay_emotions(self, decay_rate: float) -> None:
        """Decay emotional states toward neutral."""
        self.emotional_valence *= (1 - decay_rate)
        self.emotional_arousal = 0.5 + (self.emotional_arousal - 0.5) * (1 - decay_rate)


@dataclass
class VectorizedPostState:
    """Vectorized post state arrays.

    Attributes:
        n_posts: Number of posts
        post_ids: Array of post IDs
        post_id_to_idx: Mapping from post_id to index
        author_idx: Author user indices
        created_step: Creation step per post
        quality_score: Quality scores
        controversy_score: Controversy scores
        ideology_score: Ideology scores
        emotional_intensity: Emotional intensity
        view_count: View counts
        like_count: Like counts
        share_count: Share counts
        is_active: Active status
    """

    n_posts: int
    post_ids: np.ndarray
    post_id_to_idx: dict[str, int] = field(default_factory=dict)
    author_idx: np.ndarray = field(default=None)
    created_step: np.ndarray = field(default=None)
    quality_score: np.ndarray = field(default=None)
    controversy_score: np.ndarray = field(default=None)
    ideology_score: np.ndarray = field(default=None)
    emotional_intensity: np.ndarray = field(default=None)
    is_misinformation: np.ndarray = field(default=None)
    view_count: np.ndarray = field(default=None)
    like_count: np.ndarray = field(default=None)
    share_count: np.ndarray = field(default=None)
    is_active: np.ndarray = field(default=None)

    def __post_init__(self):
        """Initialize arrays."""
        if self.author_idx is None:
            self.author_idx = np.zeros(self.n_posts, dtype=np.int32)
        if self.created_step is None:
            self.created_step = np.zeros(self.n_posts, dtype=np.int32)
        if self.quality_score is None:
            self.quality_score = np.ones(self.n_posts, dtype=np.float32) * 0.5
        if self.controversy_score is None:
            self.controversy_score = np.zeros(self.n_posts, dtype=np.float32)
        if self.ideology_score is None:
            self.ideology_score = np.zeros(self.n_posts, dtype=np.float32)
        if self.emotional_intensity is None:
            self.emotional_intensity = np.zeros(self.n_posts, dtype=np.float32)
        if self.is_misinformation is None:
            self.is_misinformation = np.zeros(self.n_posts, dtype=np.bool_)
        if self.view_count is None:
            self.view_count = np.zeros(self.n_posts, dtype=np.int32)
        if self.like_count is None:
            self.like_count = np.zeros(self.n_posts, dtype=np.int32)
        if self.share_count is None:
            self.share_count = np.zeros(self.n_posts, dtype=np.int32)
        if self.is_active is None:
            self.is_active = np.ones(self.n_posts, dtype=np.bool_)


class VectorizedEngagement:
    """Vectorized engagement calculation.

    Computes engagement probabilities for batches of user-post pairs
    using NumPy operations.
    """

    def __init__(
        self,
        base_view_rate: float = 0.3,
        base_like_rate: float = 0.1,
        base_share_rate: float = 0.02,
        base_comment_rate: float = 0.03,
        interest_weight: float = 0.4,
        ideology_weight: float = 0.3,
        quality_weight: float = 0.2,
        social_weight: float = 0.3,
    ):
        """Initialize vectorized engagement calculator.

        Args:
            base_view_rate: Base probability of viewing
            base_like_rate: Base probability of liking
            base_share_rate: Base probability of sharing
            base_comment_rate: Base probability of commenting
            interest_weight: Weight for interest match
            ideology_weight: Weight for ideology alignment
            quality_weight: Weight for content quality
            social_weight: Weight for social factors
        """
        self.base_view_rate = base_view_rate
        self.base_like_rate = base_like_rate
        self.base_share_rate = base_share_rate
        self.base_comment_rate = base_comment_rate
        self.interest_weight = interest_weight
        self.ideology_weight = ideology_weight
        self.quality_weight = quality_weight
        self.social_weight = social_weight

    def compute_content_match_batch(
        self,
        user_ideology: np.ndarray,
        user_confirmation_bias: np.ndarray,
        post_ideology: np.ndarray,
    ) -> np.ndarray:
        """Compute content match for batch of user-post pairs.

        Args:
            user_ideology: User ideology values (n,)
            user_confirmation_bias: User confirmation bias (n,)
            post_ideology: Post ideology values (n,)

        Returns:
            Content match scores (n,)
        """
        # Ideology alignment
        ideology_diff = np.abs(user_ideology - post_ideology)
        ideology_match = 1 - (ideology_diff / 2)

        # Weighted by confirmation bias
        ideology_contribution = (
            user_confirmation_bias * self.ideology_weight * ideology_match
        )

        # Base interest match (simplified without topic tracking)
        interest_contribution = self.interest_weight * 0.5

        return 0.5 + interest_contribution + ideology_contribution

    def compute_quality_factor_batch(
        self,
        user_emotional_reactivity: np.ndarray,
        post_quality: np.ndarray,
        post_emotional_intensity: np.ndarray,
        post_controversy: np.ndarray,
    ) -> np.ndarray:
        """Compute quality factor for batch.

        Args:
            user_emotional_reactivity: User reactivity (n,)
            post_quality: Post quality scores (n,)
            post_emotional_intensity: Post emotional intensity (n,)
            post_controversy: Post controversy scores (n,)

        Returns:
            Quality factor (n,)
        """
        emotional_factor = 1 + post_emotional_intensity * user_emotional_reactivity
        controversy_factor = 1 + post_controversy * 0.3
        quality_factor = post_quality * emotional_factor * controversy_factor * self.quality_weight

        return 0.5 + quality_factor

    def compute_temporal_factor_batch(
        self,
        user_fatigue: np.ndarray,
        user_activity_level: np.ndarray,
        post_age: np.ndarray,
        freshness_decay: float = 0.1,
    ) -> np.ndarray:
        """Compute temporal factor for batch.

        Args:
            user_fatigue: User fatigue levels (n,)
            user_activity_level: User activity levels (n,)
            post_age: Post ages in steps (n,)
            freshness_decay: Decay rate for freshness

        Returns:
            Temporal factor (n,)
        """
        freshness = np.exp(-post_age * freshness_decay)
        fatigue_factor = 1 - user_fatigue * 0.5
        activity_factor = 0.5 + user_activity_level * 0.5

        return freshness * fatigue_factor * activity_factor

    def compute_engagement_probabilities_batch(
        self,
        user_state: VectorizedUserState,
        user_indices: np.ndarray,
        post_state: VectorizedPostState,
        post_indices: np.ndarray,
        current_step: int,
        event_multiplier: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Compute engagement probabilities for batch of user-post pairs.

        Args:
            user_state: Vectorized user state
            user_indices: User indices (n,)
            post_state: Vectorized post state
            post_indices: Post indices (n,)
            current_step: Current simulation step
            event_multiplier: Event effect multiplier

        Returns:
            Dictionary with view/like/share/comment probability arrays
        """
        n = len(user_indices)

        # Gather user features
        user_ideology = user_state.ideology[user_indices]
        user_confirmation_bias = user_state.confirmation_bias[user_indices]
        user_emotional_reactivity = user_state.emotional_reactivity[user_indices]
        user_fatigue = user_state.fatigue[user_indices]
        user_activity_level = user_state.activity_level[user_indices]
        user_misinfo_suscept = user_state.misinfo_susceptibility[user_indices]

        # Gather post features
        post_ideology = post_state.ideology_score[post_indices]
        post_quality = post_state.quality_score[post_indices]
        post_emotional = post_state.emotional_intensity[post_indices]
        post_controversy = post_state.controversy_score[post_indices]
        post_age = current_step - post_state.created_step[post_indices]
        post_misinfo = post_state.is_misinformation[post_indices]

        # Compute factors
        content_match = self.compute_content_match_batch(
            user_ideology, user_confirmation_bias, post_ideology
        )
        quality_factor = self.compute_quality_factor_batch(
            user_emotional_reactivity, post_quality, post_emotional, post_controversy
        )
        temporal_factor = self.compute_temporal_factor_batch(
            user_fatigue, user_activity_level, post_age
        )

        # Combined factor
        combined = content_match * quality_factor * temporal_factor * event_multiplier

        # Base probabilities
        view_prob = np.minimum(0.95, self.base_view_rate * combined)
        like_prob = np.minimum(0.8, self.base_like_rate * combined)
        share_prob = np.minimum(0.5, self.base_share_rate * combined)
        comment_prob = np.minimum(0.4, self.base_comment_rate * combined)

        # Misinformation boost
        misinfo_factor = 1 + post_misinfo * user_misinfo_suscept * 0.5
        like_prob *= misinfo_factor
        share_prob *= misinfo_factor

        return {
            "view": view_prob,
            "like": like_prob,
            "share": share_prob,
            "comment": comment_prob,
        }


class VectorizedFeedRanking:
    """Vectorized feed ranking using sparse matrix operations."""

    def __init__(
        self,
        recency_weight: float = 0.3,
        velocity_weight: float = 0.3,
        relevance_weight: float = 0.4,
        controversy_weight: float = 0.1,
    ):
        """Initialize vectorized feed ranking.

        Args:
            recency_weight: Weight for recency
            velocity_weight: Weight for velocity
            relevance_weight: Weight for relevance
            controversy_weight: Weight for controversy
        """
        self.recency_weight = recency_weight
        self.velocity_weight = velocity_weight
        self.relevance_weight = relevance_weight
        self.controversy_weight = controversy_weight

    def rank_posts_for_user(
        self,
        user_idx: int,
        user_state: VectorizedUserState,
        candidate_post_indices: np.ndarray,
        post_state: VectorizedPostState,
        current_step: int,
        feed_size: int = 20,
    ) -> np.ndarray:
        """Rank posts for a single user.

        Args:
            user_idx: User index
            user_state: Vectorized user state
            candidate_post_indices: Candidate post indices
            post_state: Vectorized post state
            current_step: Current step
            feed_size: Number of posts to return

        Returns:
            Sorted post indices
        """
        n_candidates = len(candidate_post_indices)
        if n_candidates == 0:
            return np.array([], dtype=np.int32)

        # Recency scores
        ages = current_step - post_state.created_step[candidate_post_indices]
        recency_scores = 1.0 / (1.0 + ages * 0.1)

        # Quality/relevance scores
        quality = post_state.quality_score[candidate_post_indices]
        emotional = post_state.emotional_intensity[candidate_post_indices]
        relevance_scores = quality * 0.5 + emotional * 0.3

        # Ideology alignment
        user_ideology = user_state.ideology[user_idx]
        post_ideology = post_state.ideology_score[candidate_post_indices]
        ideology_match = 1 - np.abs(user_ideology - post_ideology) / 2

        # Controversy boost (capped)
        controversy = np.minimum(
            post_state.controversy_score[candidate_post_indices] * self.controversy_weight,
            0.3,
        )

        # Combined score
        scores = (
            self.recency_weight * recency_scores +
            self.relevance_weight * relevance_scores +
            self.relevance_weight * ideology_match * user_state.confirmation_bias[user_idx] +
            controversy
        )

        # Sort and take top feed_size
        sorted_indices = np.argsort(-scores)[:feed_size]
        return candidate_post_indices[sorted_indices]

    def rank_posts_batch(
        self,
        user_indices: np.ndarray,
        user_state: VectorizedUserState,
        candidate_posts_per_user: list[np.ndarray],
        post_state: VectorizedPostState,
        current_step: int,
        feed_size: int = 20,
    ) -> list[np.ndarray]:
        """Rank posts for batch of users.

        Args:
            user_indices: User indices
            user_state: Vectorized user state
            candidate_posts_per_user: List of candidate post arrays per user
            post_state: Vectorized post state
            current_step: Current step
            feed_size: Feed size

        Returns:
            List of ranked post index arrays
        """
        results = []
        for i, user_idx in enumerate(user_indices):
            ranked = self.rank_posts_for_user(
                user_idx,
                user_state,
                candidate_posts_per_user[i],
                post_state,
                current_step,
                feed_size,
            )
            results.append(ranked)
        return results


def compute_social_influence_vectorized(
    following_matrix: sparse.csr_matrix,
    user_opinions: np.ndarray,
    influence_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute social influence on opinions using sparse matrix multiplication.

    Args:
        following_matrix: Sparse following adjacency matrix (n x n)
        user_opinions: Current user opinions (n,)
        influence_weights: Optional influence weights (n,)

    Returns:
        Influence values for each user (n,)
    """
    if influence_weights is not None:
        weighted_opinions = user_opinions * influence_weights
    else:
        weighted_opinions = user_opinions

    # Sum of followed users' opinions
    influence_sum = following_matrix @ weighted_opinions

    # Normalize by number of followed users
    follow_counts = np.array(following_matrix.sum(axis=1)).flatten()
    follow_counts = np.maximum(follow_counts, 1)  # Avoid division by zero

    return influence_sum / follow_counts
