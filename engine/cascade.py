"""Viral cascade mechanics."""

from typing import Any
import uuid

import numpy as np
from numpy.random import Generator

from config.schemas import CascadeConfig
from models import User, Post, Cascade, Interaction
from models.enums import InteractionType
from .state import SimulationState


class CascadeEngine:
    """Manages viral cascade spreading mechanics.

    Implements:
    - Cascade initialization and tracking
    - Exposure-based spreading
    - Threshold model for share decisions
    - Cascade decay over time
    """

    def __init__(
        self,
        config: CascadeConfig,
        seed: int | None = None,
    ):
        """Initialize cascade engine.

        Args:
            config: Cascade configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.cascade_counter = 0

    def initialize_cascade(
        self,
        post: Post,
        author: User,
        state: SimulationState,
    ) -> Cascade:
        """Initialize a new cascade for a post.

        Args:
            post: Post that may go viral
            author: Post author
            state: Simulation state

        Returns:
            New Cascade object
        """
        self.cascade_counter += 1
        cascade_id = f"cascade_{self.cascade_counter:08d}"

        cascade = Cascade(
            cascade_id=cascade_id,
            post_id=post.post_id,
        )
        cascade.initialize(author.user_id, state.current_step)

        # Link post to cascade
        post.cascade_id = cascade_id

        return cascade

    def process_cascade_spread(
        self,
        cascade: Cascade,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        network_followers: dict[str, list[str]],
    ) -> list[tuple[User, str]]:
        """Process cascade spreading for one step.

        Determines which users get exposed and potentially share.

        Args:
            cascade: Cascade to process
            post: Associated post
            state: Simulation state
            users: All users
            network_followers: Mapping of user_id to list of follower IDs

        Returns:
            List of (user, source_user_id) tuples for users who were exposed
        """
        if not cascade.is_active:
            return []

        exposures = []

        # Get recent sharers (users who shared in recent steps)
        recent_sharers = self._get_recent_sharers(cascade, state, lookback=3)

        for sharer_id in recent_sharers:
            # Get followers of sharer
            followers = network_followers.get(sharer_id, [])

            for follower_id in followers:
                if follower_id not in cascade.reached_users:
                    follower = users.get(follower_id)
                    if follower and follower.is_active():
                        # Calculate exposure probability
                        exposure_prob = self._calculate_exposure_probability(
                            follower, post, cascade, state
                        )

                        if self.rng.random() < exposure_prob:
                            cascade.record_reach(follower_id)
                            exposures.append((follower, sharer_id))

        # Update cascade velocity
        cascade.update_peak_velocity(state.current_step)

        # Check if cascade should deactivate
        if self._should_deactivate(cascade, state):
            cascade.deactivate()

        return exposures

    def _get_recent_sharers(
        self,
        cascade: Cascade,
        state: SimulationState,
        lookback: int = 3,
    ) -> list[str]:
        """Get users who shared in recent steps.

        Args:
            cascade: Cascade to check
            state: Simulation state
            lookback: Number of steps to look back

        Returns:
            List of user IDs who shared recently
        """
        recent_sharers = []
        min_step = max(cascade.start_step, state.current_step - lookback)

        for step in range(min_step, state.current_step + 1):
            count = cascade.shares_by_step.get(step, 0)
            if count > 0:
                # Find users who shared at this step from interactions
                post_interactions = state.get_interactions_for_post(cascade.post_id)
                for interaction in post_interactions:
                    if (interaction.interaction_type == InteractionType.SHARE and
                            interaction.step == step):
                        recent_sharers.append(interaction.user_id)

        return recent_sharers

    def _calculate_exposure_probability(
        self,
        user: User,
        post: Post,
        cascade: Cascade,
        state: SimulationState,
    ) -> float:
        """Calculate probability of user being exposed to cascading content.

        exposure_factor = base_rate * (1 + log(shares) * 0.1) *
                         (1 + virality_potential * 0.5) * decay

        Args:
            user: User who might be exposed
            post: Post spreading
            cascade: Cascade object
            state: Simulation state

        Returns:
            Exposure probability
        """
        # Base spread rate
        base_rate = self.config.base_spread_rate

        # Share velocity boost
        velocity = cascade.get_velocity(state.current_step)
        velocity_factor = 1 + np.log1p(velocity) * self.config.share_velocity_multiplier

        # Virality boost
        virality_factor = 1 + post.virality_score * self.config.virality_boost

        # Time decay
        age = state.current_step - cascade.start_step
        decay = np.exp(-age * self.config.decay_rate)

        # Event effect
        event_effect = state.get_combined_event_effect()
        event_factor = event_effect.engagement_multiplier

        exposure_prob = base_rate * velocity_factor * virality_factor * decay * event_factor

        return min(0.9, exposure_prob)

    def should_user_share(
        self,
        user: User,
        post: Post,
        cascade: Cascade,
        state: SimulationState,
        base_share_prob: float,
    ) -> bool:
        """Determine if a user should share content using threshold model.

        User shares if >= N friends already shared, where N is their threshold.

        Args:
            user: User considering sharing
            post: Post to potentially share
            cascade: Associated cascade
            state: Simulation state
            base_share_prob: Base share probability from engagement model

        Returns:
            True if user should share
        """
        # Get user's threshold (based on traits)
        threshold = self._get_user_threshold(user)

        # Count friends who shared
        friends_who_shared = self._count_friends_who_shared(
            user, cascade, state
        )

        # Threshold model: share if enough friends shared
        threshold_met = friends_who_shared >= threshold

        # Combine with probabilistic model
        if threshold_met:
            # Boost share probability if threshold met
            boosted_prob = min(0.8, base_share_prob * 2)
            return self.rng.random() < boosted_prob
        else:
            # Can still share probabilistically, but lower chance
            reduced_prob = base_share_prob * 0.3
            return self.rng.random() < reduced_prob

    def _get_user_threshold(self, user: User) -> int:
        """Get user's sharing threshold based on traits.

        Args:
            user: User to get threshold for

        Returns:
            Threshold value
        """
        # Lower threshold for more susceptible/reactive users
        base_threshold = self.config.threshold_max

        # Emotional reactivity lowers threshold
        threshold = base_threshold - user.traits.emotional_reactivity * 2

        # Misinfo susceptibility lowers threshold for misinfo
        threshold -= user.traits.misinfo_susceptibility * 1

        return max(self.config.threshold_min, int(threshold))

    def _count_friends_who_shared(
        self,
        user: User,
        cascade: Cascade,
        state: SimulationState,
    ) -> int:
        """Count how many friends shared this content.

        Args:
            user: User to check friends for
            cascade: Cascade being spread
            state: Simulation state

        Returns:
            Count of friends who shared
        """
        count = 0
        post_interactions = state.get_interactions_for_post(cascade.post_id)

        for interaction in post_interactions:
            if (interaction.interaction_type == InteractionType.SHARE and
                    interaction.user_id in user.following):
                count += 1

        return count

    def record_share(
        self,
        cascade: Cascade,
        user: User,
        source_user_id: str,
        state: SimulationState,
    ) -> None:
        """Record a share in the cascade.

        Args:
            cascade: Cascade being shared
            user: User who shared
            source_user_id: User who exposed them
            state: Simulation state
        """
        cascade.record_share(user.user_id, source_user_id, state.current_step)

    def _should_deactivate(
        self,
        cascade: Cascade,
        state: SimulationState,
    ) -> bool:
        """Determine if cascade should be deactivated.

        Args:
            cascade: Cascade to check
            state: Simulation state

        Returns:
            True if cascade should deactivate
        """
        # Deactivate if velocity is very low for extended period
        velocity = cascade.get_velocity(state.current_step, window=5)
        if velocity < 0.1:
            # Check how long it's been low
            age = state.current_step - cascade.start_step
            if age > 20:
                return True

        # Deactivate if cascade is very old
        if state.current_step - cascade.start_step > 100:
            return True

        return False

    def get_cascade_statistics(
        self,
        cascade: Cascade,
        state: SimulationState,
    ) -> dict[str, Any]:
        """Get statistics for a cascade.

        Args:
            cascade: Cascade to analyze
            state: Simulation state

        Returns:
            Dictionary of statistics
        """
        return {
            "cascade_id": cascade.cascade_id,
            "post_id": cascade.post_id,
            "total_shares": cascade.total_shares,
            "total_reach": cascade.total_reach,
            "max_depth": cascade.max_depth,
            "peak_velocity": cascade.peak_velocity,
            "current_velocity": cascade.get_velocity(state.current_step),
            "branching_factor": cascade.get_branching_factor(),
            "depth_distribution": cascade.get_depth_distribution(),
            "is_active": cascade.is_active,
            "age": state.current_step - cascade.start_step,
        }

    def get_viral_cascades(
        self,
        state: SimulationState,
        min_shares: int = 10,
    ) -> list[Cascade]:
        """Get cascades that have gone viral.

        Args:
            state: Simulation state
            min_shares: Minimum shares to be considered viral

        Returns:
            List of viral cascades
        """
        return [
            cascade for cascade in state.cascades.values()
            if cascade.total_shares >= min_shares
        ]
