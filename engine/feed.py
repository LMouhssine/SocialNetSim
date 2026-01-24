"""Feed ranking algorithms."""

from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import FeedConfig, SimulationConfig
from models import User, Post
from models.enums import FeedAlgorithm
from .state import SimulationState
from .ranking_optimization import FeedOptimizer, FeedOptimizationConfig, create_user_context_vector


class EnhancedFeedRanker:
    """Ranks posts for user feeds using various algorithms.

    Supports:
    - Chronological: Sort by creation time
    - Engagement: Rank by predicted engagement
    - Diverse: Balance relevance with diversity
    - Interest: Personalized by user interests

    Enhanced features:
    - Controversy amplification (capped boost)
    - Social proximity (2-hop connections)
    - Bandit-based weight optimization
    """

    def __init__(
        self,
        config: FeedConfig | SimulationConfig,
        seed: int | None = None,
        controversy_amplification_weight: float = 0.1,
        controversy_cap: float = 0.3,
        social_proximity_weight: float = 0.2,
        use_bandit_optimization: bool = False,
        bandit_type: str = "thompson",
    ):
        """Initialize feed ranker.

        Args:
            config: Feed configuration
            seed: Random seed
            controversy_amplification_weight: Weight for controversy boost
            controversy_cap: Maximum controversy boost
            social_proximity_weight: Weight for social proximity
            use_bandit_optimization: Whether to use bandit optimization
            bandit_type: Type of bandit ("thompson" or "linucb")
        """
        if isinstance(config, SimulationConfig):
            if seed is None:
                seed = config.seed
            config = config.feed

        self.config = config
        self.rng = np.random.default_rng(seed)
        self.algorithm = FeedAlgorithm(config.algorithm)

        # Enhanced ranking parameters
        self.controversy_weight = controversy_amplification_weight
        self.controversy_cap = controversy_cap
        self.social_proximity_weight = social_proximity_weight

        # Bandit optimization
        self.use_bandit = use_bandit_optimization
        if self.use_bandit:
            from .ranking_optimization import BanditType
            opt_config = FeedOptimizationConfig(
                enabled=True,
                bandit_type=BanditType(bandit_type),
            )
            self.optimizer = FeedOptimizer(opt_config, seed=seed)
        else:
            self.optimizer = None

    def _normalize_algorithm(self, algorithm: FeedAlgorithm | str) -> FeedAlgorithm:
        """Normalize algorithm input to FeedAlgorithm enum."""
        if isinstance(algorithm, FeedAlgorithm):
            return algorithm
        try:
            return FeedAlgorithm(str(algorithm))
        except Exception:
            return FeedAlgorithm.ENGAGEMENT

    def rank_feed(
        self,
        user: User,
        candidate_posts: list[Post],
        state: SimulationState,
        override_algorithm: FeedAlgorithm | None = None,
    ) -> list[Post]:
        """Rank posts for a user's feed.

        Args:
            user: User viewing the feed
            candidate_posts: Posts to rank
            state: Current simulation state
            override_algorithm: Optional algorithm override

        Returns:
            Ranked list of posts
        """
        algorithm = self._normalize_algorithm(override_algorithm or self.algorithm)

        # Filter out posts user has seen (with penalty instead of removal)
        user_state = state.get_user_state(user.user_id)

        if algorithm == FeedAlgorithm.CHRONOLOGICAL:
            scored_posts = self._rank_chronological(candidate_posts, state)
        elif algorithm == FeedAlgorithm.ENGAGEMENT:
            scored_posts = self._rank_engagement(user, candidate_posts, state)
        elif algorithm == FeedAlgorithm.DIVERSE:
            scored_posts = self._rank_diverse(user, candidate_posts, state)
        elif algorithm == FeedAlgorithm.INTEREST:
            scored_posts = self._rank_interest(user, candidate_posts, state)
        else:
            scored_posts = self._rank_engagement(user, candidate_posts, state)

        # Apply seen penalty
        if user_state:
            for post, score in scored_posts:
                if user_state.has_seen(post.post_id):
                    score *= (1 - self.config.seen_penalty)

        # Sort by score and take top N
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        ranked_posts = [post for post, _ in scored_posts[:self.config.feed_size]]

        return ranked_posts

    def rank_posts(
        self,
        posts: list[Post],
        user: User | None = None,
        current_step: int = 0,
        limit: int | None = None,
        seen_post_ids: set[str] | None = None,
    ) -> list[Post]:
        """Backward-compatible ranking API (no SimulationState required)."""
        seen_post_ids = seen_post_ids or set()
        filtered = [p for p in posts if p.post_id not in seen_post_ids]

        algorithm = self._normalize_algorithm(self.algorithm)

        if algorithm == FeedAlgorithm.CHRONOLOGICAL:
            ranked = sorted(filtered, key=lambda p: p.created_step, reverse=True)
        elif algorithm == FeedAlgorithm.ENGAGEMENT:
            ranked = sorted(filtered, key=lambda p: p.total_engagement, reverse=True)
        else:
            ranked = list(filtered)

        if limit is not None:
            ranked = ranked[:limit]

        return ranked

    def _rank_chronological(
        self,
        posts: list[Post],
        state: SimulationState,
    ) -> list[tuple[Post, float]]:
        """Rank posts by creation time (newest first).

        Args:
            posts: Posts to rank
            state: Simulation state

        Returns:
            List of (post, score) tuples
        """
        scored = []
        max_step = state.current_step

        for post in posts:
            # Newer posts get higher scores
            age = max_step - post.created_step
            score = 1.0 / (1.0 + age * 0.1)  # Decay with age
            scored.append((post, score))

        return scored

    def _rank_engagement(
        self,
        user: User,
        posts: list[Post],
        state: SimulationState,
    ) -> list[tuple[Post, float]]:
        """Rank posts by predicted engagement with enhanced features.

        Score = w1*velocity + w2*predicted_engagement + w3*author_relevance + w4*recency
                + controversy_boost + social_proximity_boost

        Args:
            user: Viewing user
            posts: Posts to rank
            state: Simulation state

        Returns:
            List of (post, score) tuples
        """
        scored = []

        # Get dynamic weights from bandit optimizer if enabled
        if self.use_bandit and self.optimizer:
            weights = self.optimizer.select_weights()
            velocity_weight = weights.get("velocity", self.config.velocity_weight)
            relevance_weight = weights.get("relevance", self.config.relevance_weight)
            recency_weight = weights.get("recency", self.config.recency_weight)
        else:
            velocity_weight = self.config.velocity_weight
            relevance_weight = self.config.relevance_weight
            recency_weight = self.config.recency_weight

        for post in posts:
            # Recency factor
            age = state.current_step - post.created_step
            recency = 1.0 / (1.0 + age * recency_weight * 0.5)

            # Velocity (engagement momentum)
            velocity = post.get_velocity(state.current_step)
            velocity_score = min(1.0, velocity / 10.0)  # Normalize

            # Author relevance (following, similarity)
            author = state.users.get(post.author_id)
            if author:
                is_following = post.author_id in user.following
                author_relevance = 0.3 if is_following else 0.0
                author_relevance += author.influence_score * 0.3
            else:
                author_relevance = 0.0

            # Predicted engagement (content quality + virality)
            predicted_engagement = (
                post.content.quality_score * 0.3 +
                post.virality_score * 0.4 +
                post.content.emotional_intensity * 0.3
            )

            # ENHANCED: Controversy amplification (capped)
            controversy_boost = self._calculate_controversy_boost(post)

            # ENHANCED: Social proximity (2-hop connections)
            social_proximity_boost = self._calculate_social_proximity(
                user, post, state
            )

            # Combine with weights
            score = (
                velocity_weight * velocity_score +
                relevance_weight * predicted_engagement +
                recency_weight * recency +
                0.2 * author_relevance +
                controversy_boost +
                social_proximity_boost
            )

            # Small random factor to add variety
            score += self.rng.uniform(0, 0.05)

            scored.append((post, score))

        return scored

    def _calculate_controversy_boost(self, post: Post) -> float:
        """Calculate controversy amplification boost (capped).

        Controversial content gets a boost, but capped to prevent
        excessive promotion of divisive content.

        Args:
            post: Post to evaluate

        Returns:
            Controversy boost value
        """
        raw_boost = post.content.controversy_score * self.controversy_weight

        # Apply cap to prevent runaway controversy amplification
        return min(raw_boost, self.controversy_cap)

    def _calculate_social_proximity(
        self,
        user: User,
        post: Post,
        state: SimulationState,
    ) -> float:
        """Calculate social proximity boost based on 2-hop connections.

        Boosts posts that friends or friends-of-friends engaged with.

        Args:
            user: Viewing user
            post: Post to evaluate
            state: Simulation state

        Returns:
            Social proximity boost value
        """
        if not self.social_proximity_weight:
            return 0.0

        # Get post interactions
        interactions = state.get_interactions_for_post(post.post_id)
        if not interactions:
            return 0.0

        # 1-hop: Direct friends who engaged
        direct_friend_engagements = 0
        # 2-hop: Friends of friends who engaged
        friend_of_friend_engagements = 0

        engaged_users = {i.user_id for i in interactions}

        for engaged_user_id in engaged_users:
            if engaged_user_id in user.following:
                direct_friend_engagements += 1
            else:
                # Check if any friend follows this user (2-hop)
                for friend_id in user.following:
                    friend = state.users.get(friend_id)
                    if friend and engaged_user_id in friend.following:
                        friend_of_friend_engagements += 1
                        break

        # Weight direct friends more than 2-hop
        proximity_score = (
            direct_friend_engagements * 0.3 +
            friend_of_friend_engagements * 0.1
        )

        # Normalize and apply weight
        normalized = min(1.0, proximity_score / 5.0)
        return normalized * self.social_proximity_weight

    def record_engagement_feedback(
        self,
        views: int,
        likes: int,
        shares: int,
        comments: int,
    ) -> None:
        """Record engagement feedback for bandit optimization.

        Args:
            views: Number of views
            likes: Number of likes
            shares: Number of shares
            comments: Number of comments
        """
        if self.use_bandit and self.optimizer:
            reward = self.optimizer.calculate_engagement_reward(
                views, likes, shares, comments
            )
            self.optimizer.record_reward(reward)

    def get_optimization_stats(self) -> dict[str, Any] | None:
        """Get bandit optimization statistics.

        Returns:
            Statistics dictionary or None if optimization disabled
        """
        if self.use_bandit and self.optimizer:
            return self.optimizer.get_statistics()
        return None

    def _rank_diverse(
        self,
        user: User,
        posts: list[Post],
        state: SimulationState,
    ) -> list[tuple[Post, float]]:
        """Rank posts balancing relevance with diversity.

        Args:
            user: Viewing user
            posts: Posts to rank
            state: Simulation state

        Returns:
            List of (post, score) tuples
        """
        # Start with engagement-based scores
        base_scores = self._rank_engagement(user, posts, state)
        base_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity penalty for similar consecutive posts
        scored = []
        selected_topics: set[str] = set()
        selected_authors: set[str] = set()

        for post, base_score in base_scores:
            diversity_penalty = 0.0

            # Penalize repeated topics
            common_topics = post.content.topics & selected_topics
            if common_topics:
                diversity_penalty += len(common_topics) * self.config.diversity_penalty

            # Penalize repeated authors
            if post.author_id in selected_authors:
                diversity_penalty += self.config.diversity_penalty * 2

            # Apply penalty
            score = base_score * (1 - diversity_penalty)

            scored.append((post, score))

            # Track seen topics and authors
            selected_topics.update(post.content.topics)
            selected_authors.add(post.author_id)

        return scored

    def _rank_interest(
        self,
        user: User,
        posts: list[Post],
        state: SimulationState,
    ) -> list[tuple[Post, float]]:
        """Rank posts by user interest match.

        Args:
            user: Viewing user
            posts: Posts to rank
            state: Simulation state

        Returns:
            List of (post, score) tuples
        """
        scored = []

        for post in posts:
            # Calculate interest match
            interest_score = self._calculate_interest_match(user, post)

            # Ideology match (weighted by user's confirmation bias)
            ideology_diff = abs(user.traits.ideology - post.content.ideology_score)
            ideology_match = 1 - (ideology_diff / 2)
            # Higher confirmation bias = stronger preference for aligned content
            ideology_weight = 0.2 + user.traits.confirmation_bias * 0.3
            ideology_score = ideology_match * ideology_weight

            # Recency
            age = state.current_step - post.created_step
            recency = 1.0 / (1.0 + age * 0.1)

            # Author relationship
            author_boost = 0.2 if post.author_id in user.following else 0.0

            # Combine
            score = (
                0.4 * interest_score +
                ideology_score +
                0.2 * recency +
                author_boost +
                0.1 * post.content.quality_score
            )

            scored.append((post, score))

        return scored

    def _calculate_interest_match(self, user: User, post: Post) -> float:
        """Calculate how well a post matches user interests.

        Args:
            user: User
            post: Post

        Returns:
            Match score (0-1)
        """
        if not user.interests or not post.content.topics:
            return 0.3  # Default neutral score

        common_topics = user.interests & post.content.topics
        if not common_topics:
            return 0.1

        # Weight by user's interest strength and post's topic weight
        match_score = sum(
            user.get_interest_weight(t) * post.content.get_topic_weight(t)
            for t in common_topics
        )

        # Normalize
        max_possible = min(len(user.interests), len(post.content.topics))
        return min(1.0, match_score / max(1, max_possible))

    def get_candidate_posts(
        self,
        user: User,
        state: SimulationState,
        include_from_following: bool = True,
        include_viral: bool = True,
        max_age_steps: int = 50,
    ) -> list[Post]:
        """Get candidate posts for a user's feed.

        Args:
            user: User viewing feed
            state: Simulation state
            include_from_following: Include posts from followed users
            include_viral: Include viral posts from non-followed users
            max_age_steps: Maximum age of posts to consider

        Returns:
            List of candidate posts
        """
        candidates = set()
        min_step = max(0, state.current_step - max_age_steps)

        # Posts from followed users
        if include_from_following:
            for followed_id in user.following:
                for post in state.get_posts_by_author(followed_id):
                    if post.created_step >= min_step and post.is_visible():
                        candidates.add(post.post_id)

        # Viral posts (high engagement regardless of following)
        if include_viral:
            for post in state.get_active_posts():
                if post.created_step >= min_step:
                    # Include if above engagement threshold
                    if post.total_engagement > 10 or post.virality_score > 0.5:
                        candidates.add(post.post_id)

        # Recent posts (exploration)
        recent_posts = state.get_recent_posts(n_steps=10)
        for post in recent_posts:
            if post.is_visible() and self.rng.random() < 0.3:
                candidates.add(post.post_id)

        return [state.posts[pid] for pid in candidates if pid in state.posts]

    def apply_event_effects(
        self,
        scores: list[tuple[Post, float]],
        state: SimulationState,
    ) -> list[tuple[Post, float]]:
        """Apply active event effects to feed scores.

        Args:
            scores: Current (post, score) tuples
            state: Simulation state

        Returns:
            Modified scores
        """
        effect = state.get_combined_event_effect()

        modified = []
        for post, score in scores:
            # Apply topic boosts
            for topic in post.content.topics:
                topic_mult = effect.get_topic_multiplier(topic)
                if topic_mult > 1.0:
                    score *= topic_mult

            # Apply ideology activation
            if effect.ideology_activation > 0:
                if abs(post.content.ideology_score) > 0.3:
                    score *= (1 + effect.ideology_activation * 0.5)

            modified.append((post, score))

        return modified


# Backward compatibility alias
FeedRanker = EnhancedFeedRanker
