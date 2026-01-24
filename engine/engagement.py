"""User engagement model."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import EngagementConfig
from models import User, Post, Interaction, UserCognitiveState, InteractionMemory
from models.enums import InteractionType
from .state import SimulationState
from .decision_model import UtilityBasedDecisionModel, DecisionType, DecisionConfig


@dataclass
class EngagementProbabilities:
    """Calculated engagement probabilities for a user-post pair."""

    view: float = 0.0
    like: float = 0.0
    share: float = 0.0
    comment: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "view": self.view,
            "like": self.like,
            "share": self.share,
            "comment": self.comment,
        }


class EngagementModel:
    """Models user engagement with content.

    Engagement probability = base_rate * content_match * quality_factor *
                            social_factor * temporal_factor

    Where:
    - content_match: topic interest + ideology alignment (weighted by confirmation_bias)
    - quality_factor: amplified by emotional_reactivity
    - social_factor: author influence + friend engagement count
    - temporal_factor: post freshness decay and user fatigue
    """

    def __init__(
        self,
        config: EngagementConfig,
        seed: int | None = None,
        use_utility_model: bool = True,
        attention_recovery_rate: float = 0.1,
        emotional_decay_rate: float = 0.1,
    ):
        """Initialize engagement model.

        Args:
            config: Engagement configuration
            seed: Random seed
            use_utility_model: Whether to use utility-based decisions
            attention_recovery_rate: Rate of attention recovery per step
            emotional_decay_rate: Rate of emotional state decay
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.interaction_counter = 0
        self.use_utility_model = use_utility_model
        self.attention_recovery_rate = attention_recovery_rate
        self.emotional_decay_rate = emotional_decay_rate

        # Initialize utility-based decision model
        if self.use_utility_model:
            self.decision_model = UtilityBasedDecisionModel(
                config=DecisionConfig(
                    emotional_decay_per_step=emotional_decay_rate,
                ),
                seed=seed,
            )
        else:
            self.decision_model = None

    def calculate_engagement_probability(
        self,
        user: User,
        post: Post,
        state: SimulationState | int | None = None,
        users: dict[str, User] | None = None,
        cognitive_state: UserCognitiveState | None = None,
        current_step: int | None = None,
    ) -> EngagementProbabilities | float:
        """Calculate engagement probabilities for a user-post pair.

        Backward compatibility:
        - If ``state`` is omitted, returns a scalar probability (view probability).
        - ``current_step`` can be provided for legacy callers.
        """
        # Handle legacy signature where state was an int current_step
        if isinstance(state, int) and current_step is None:
            current_step = state
            state = None

        if isinstance(state, SimulationState):
            if users is None:
                users = state.users
            # Use utility-based model if enabled
            if self.use_utility_model and self.decision_model:
                return self._calculate_utility_based_probabilities(
                    user, post, state, users, cognitive_state
                )

            # Fall back to original probabilistic model
            return self._calculate_legacy_probabilities(user, post, state, users)

        # Legacy path: no SimulationState provided -> return scalar probability
        if current_step is None:
            current_step = 0

        users_map = users or {user.user_id: user}
        dummy_state = SimulationState(users_map)
        dummy_state.current_step = current_step

        probs = self._calculate_legacy_probabilities(user, post, dummy_state, users_map)
        return probs.view

    def decide_interaction_type(
        self,
        user: User,
        post: Post,
        current_step: int = 0,
    ) -> InteractionType | None:
        """Backward-compatible interaction decision."""
        prob = self.calculate_engagement_probability(
            user=user,
            post=post,
            current_step=current_step,
        )
        if isinstance(prob, EngagementProbabilities):
            engage_prob = prob.view
        else:
            engage_prob = prob

        if self.rng.random() > engage_prob:
            return None

        roll = self.rng.random()
        if roll < 0.6:
            return InteractionType.VIEW
        if roll < 0.8:
            return InteractionType.LIKE
        if roll < 0.9:
            return InteractionType.COMMENT
        return InteractionType.SHARE

    def _calculate_utility_based_probabilities(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        cognitive_state: UserCognitiveState | None = None,
    ) -> EngagementProbabilities:
        """Calculate probabilities using utility-based model.

        Args:
            user: User viewing the post
            post: Post being viewed
            state: Simulation state
            users: Dictionary of all users
            cognitive_state: User's cognitive state

        Returns:
            EngagementProbabilities object
        """
        # Get utility from decision model
        utility = self.decision_model.compute_engagement_utility(
            user, post, state, users, cognitive_state
        )

        # Get event effect
        event_effect = state.get_combined_event_effect()
        event_multiplier = event_effect.engagement_multiplier

        # Convert utility to probabilities
        # Utility (0-1) modulates base rates
        utility_factor = 0.5 + utility  # Range: 0.5 to 1.5

        view_prob = min(0.95, self.config.base_view_rate * utility_factor * event_multiplier)
        like_prob = min(0.8, self.config.base_like_rate * utility_factor * event_multiplier)
        share_prob = min(0.5, self.config.base_share_rate * utility_factor * event_multiplier * 0.7)
        comment_prob = min(0.4, self.config.base_comment_rate * utility_factor * event_multiplier)

        # Misinformation susceptibility effect
        if post.content.is_misinformation:
            misinfo_factor = 1 + user.traits.misinfo_susceptibility * event_effect.misinfo_boost * 0.5
            like_prob *= misinfo_factor
            share_prob *= misinfo_factor

        # Cognitive state effects
        if cognitive_state is not None:
            # Depleted attention reduces engagement
            attention_factor = 0.5 + 0.5 * cognitive_state.attention_budget
            view_prob *= attention_factor
            like_prob *= attention_factor
            share_prob *= attention_factor
            comment_prob *= attention_factor

            # High arousal increases engagement
            arousal_factor = 0.8 + 0.4 * cognitive_state.emotional_arousal
            like_prob *= arousal_factor
            share_prob *= arousal_factor

        return EngagementProbabilities(
            view=view_prob,
            like=like_prob,
            share=share_prob,
            comment=comment_prob,
        )

    def _calculate_legacy_probabilities(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
    ) -> EngagementProbabilities:
        """Calculate probabilities using original model (backward compatibility).

        Args:
            user: User viewing the post
            post: Post being viewed
            state: Simulation state
            users: Dictionary of all users

        Returns:
            EngagementProbabilities object
        """
        # Content match factor
        content_match = self._calculate_content_match(user, post)

        # Quality factor (amplified by emotional reactivity)
        quality_factor = self._calculate_quality_factor(user, post)

        # Social factor
        social_factor = self._calculate_social_factor(user, post, state, users)

        # Temporal factor
        temporal_factor = self._calculate_temporal_factor(user, post, state)

        # Get event effect
        event_effect = state.get_combined_event_effect()
        event_multiplier = event_effect.engagement_multiplier

        # Calculate final probabilities
        combined_factor = content_match * quality_factor * social_factor * temporal_factor * event_multiplier

        view_prob = min(0.95, self.config.base_view_rate * combined_factor)
        like_prob = min(0.8, self.config.base_like_rate * combined_factor)
        share_prob = min(0.5, self.config.base_share_rate * combined_factor)
        comment_prob = min(0.4, self.config.base_comment_rate * combined_factor)

        # Misinformation boost (susceptible users more likely to engage with misinfo)
        if post.content.is_misinformation:
            misinfo_factor = 1 + user.traits.misinfo_susceptibility * event_effect.misinfo_boost * 0.5
            like_prob *= misinfo_factor
            share_prob *= misinfo_factor

        return EngagementProbabilities(
            view=view_prob,
            like=like_prob,
            share=share_prob,
            comment=comment_prob,
        )

    def _calculate_content_match(self, user: User, post: Post) -> float:
        """Calculate content match factor.

        Considers topic interest and ideology alignment.
        """
        # Topic interest match
        if user.interests and post.content.topics:
            common_topics = user.interests & post.content.topics
            if common_topics:
                interest_match = sum(
                    user.get_interest_weight(t) * post.content.get_topic_weight(t)
                    for t in common_topics
                ) / len(common_topics)
            else:
                interest_match = 0.2  # Some baseline for non-matched content
        else:
            interest_match = 0.3

        # Ideology alignment (weighted by confirmation bias)
        ideology_diff = abs(user.traits.ideology - post.content.ideology_score)
        ideology_match = 1 - (ideology_diff / 2)

        # Confirmation bias amplifies ideology preference
        ideology_weight = user.traits.confirmation_bias * self.config.ideology_weight

        content_match = (
            self.config.interest_weight * interest_match +
            ideology_weight * ideology_match
        )

        # Normalize to reasonable range
        return 0.5 + content_match

    def _calculate_quality_factor(self, user: User, post: Post) -> float:
        """Calculate quality factor.

        Quality and emotional content, amplified by user's emotional reactivity.
        """
        # Base quality
        quality = post.content.quality_score

        # Emotional intensity impact (reactive users respond more to emotional content)
        emotional_intensity = post.content.emotional_intensity
        emotional_factor = 1 + emotional_intensity * user.traits.emotional_reactivity

        # Controversy can drive engagement
        controversy_factor = 1 + post.content.controversy_score * 0.3

        quality_factor = quality * emotional_factor * controversy_factor * self.config.quality_weight

        return 0.5 + quality_factor

    def _calculate_social_factor(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
    ) -> float:
        """Calculate social factor.

        Author influence and friend engagement count.
        """
        # Author influence
        author = users.get(post.author_id)
        if author:
            author_influence = author.influence_score
            # Boost if user follows author
            if post.author_id in user.following:
                author_influence += 0.2
        else:
            author_influence = 0.3

        # Friend engagement (how many friends engaged with this post)
        post_interactions = state.get_interactions_for_post(post.post_id)
        friend_engagements = sum(
            1 for i in post_interactions
            if i.user_id in user.following and i.interaction_type != InteractionType.VIEW
        )

        # Log scale for friend engagement
        friend_factor = np.log1p(friend_engagements) / 3  # Normalize

        social_factor = (
            author_influence * 0.6 +
            friend_factor * 0.4
        ) * self.config.social_weight

        return 0.5 + social_factor

    def _calculate_temporal_factor(
        self,
        user: User,
        post: Post,
        state: SimulationState,
    ) -> float:
        """Calculate temporal factor.

        Post freshness and user fatigue.
        """
        # Post freshness (decay with age)
        age = state.current_step - post.created_step
        freshness = np.exp(-age * self.config.freshness_decay)

        # User fatigue
        user_state = state.get_user_state(user.user_id)
        if user_state:
            fatigue_factor = 1 - user_state.fatigue * 0.5
        else:
            fatigue_factor = 1.0

        # User activity level baseline
        activity_factor = 0.5 + user.traits.activity_level * 0.5

        return freshness * fatigue_factor * activity_factor

    def process_engagement(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        source_user_id: str | None = None,
        cognitive_state: UserCognitiveState | None = None,
    ) -> list[Interaction]:
        """Process user engagement with a post.

        Args:
            user: User engaging
            post: Post being engaged with
            state: Simulation state
            users: All users
            source_user_id: User who exposed this user to the post
            cognitive_state: User's cognitive state

        Returns:
            List of generated interactions
        """
        interactions = []
        probs = self.calculate_engagement_probability(
            user, post, state, users, cognitive_state
        )

        # Always attempt view first
        if self.rng.random() < probs.view:
            view_interaction = self._create_interaction(
                user, post, InteractionType.VIEW, state, source_user_id
            )
            interactions.append(view_interaction)
            post.record_view()

            # Update cognitive state if available
            if cognitive_state is not None:
                self._update_cognitive_state_for_view(
                    cognitive_state, post, state.current_step
                )

            # Only consider other engagements if viewed
            if self.rng.random() < probs.like:
                like_interaction = self._create_interaction(
                    user, post, InteractionType.LIKE, state, source_user_id
                )
                interactions.append(like_interaction)
                post.record_like()

                if cognitive_state is not None:
                    self._update_cognitive_state_for_engagement(
                        cognitive_state, post, "like", state.current_step
                    )

            if self.rng.random() < probs.share:
                share_interaction = self._create_interaction(
                    user, post, InteractionType.SHARE, state, source_user_id
                )
                interactions.append(share_interaction)
                post.record_share()

                if cognitive_state is not None:
                    self._update_cognitive_state_for_engagement(
                        cognitive_state, post, "share", state.current_step
                    )

            if self.rng.random() < probs.comment:
                comment_interaction = self._create_interaction(
                    user, post, InteractionType.COMMENT, state, source_user_id
                )
                interactions.append(comment_interaction)
                post.record_comment()

                if cognitive_state is not None:
                    self._update_cognitive_state_for_engagement(
                        cognitive_state, post, "comment", state.current_step
                    )

        return interactions

    def _update_cognitive_state_for_view(
        self,
        cognitive_state: UserCognitiveState,
        post: Post,
        step: int,
    ) -> None:
        """Update cognitive state after viewing content.

        Args:
            cognitive_state: User's cognitive state
            post: Viewed post
            step: Current step
        """
        # Deplete attention (viewing has low cost)
        cognitive_state.deplete_attention(0.02)

        # Update emotional state based on content
        content_valence = 0.0
        if post.content.sentiment.value == "positive":
            content_valence = 0.3
        elif post.content.sentiment.value == "negative":
            content_valence = -0.3

        arousal_delta = post.content.emotional_intensity * 0.1

        cognitive_state.update_emotional_state(
            valence_delta=content_valence * 0.1,
            arousal_delta=arousal_delta,
            decay_rate=self.emotional_decay_rate,
        )

    def _update_cognitive_state_for_engagement(
        self,
        cognitive_state: UserCognitiveState,
        post: Post,
        engagement_type: str,
        step: int,
    ) -> None:
        """Update cognitive state after engaging with content.

        Args:
            cognitive_state: User's cognitive state
            post: Engaged post
            engagement_type: Type of engagement
            step: Current step
        """
        # Deplete attention based on engagement type
        attention_costs = {"like": 0.03, "share": 0.08, "comment": 0.1}
        cognitive_state.deplete_attention(attention_costs.get(engagement_type, 0.05))

        # Calculate emotional impact
        emotional_impact = 0.0
        if engagement_type in ("like", "share"):
            emotional_impact = 0.2  # Positive engagement
        elif engagement_type == "comment":
            # Comments can be positive or negative based on content
            emotional_impact = 0.1 if post.content.controversy_score < 0.5 else -0.1

        # Add to interaction memory
        memory = InteractionMemory(
            post_id=post.post_id,
            author_id=post.author_id,
            interaction_type=engagement_type,
            step=step,
            emotional_impact=emotional_impact,
            topics=tuple(post.content.topics),
        )
        cognitive_state.add_interaction_memory(memory)

        # Update opinion based on content ideology
        cognitive_state.update_opinion(
            influence=post.content.ideology_score,
            influence_weight=0.05 if engagement_type == "share" else 0.02,
        )

    def recover_user_cognitive_state(
        self,
        cognitive_state: UserCognitiveState,
    ) -> None:
        """Recover user cognitive state between steps.

        Called at the beginning of each step.

        Args:
            cognitive_state: User's cognitive state to recover
        """
        cognitive_state.recover_attention(self.attention_recovery_rate)
        cognitive_state.update_emotional_state(0, 0, self.emotional_decay_rate)

    def _create_interaction(
        self,
        user: User,
        post: Post,
        interaction_type: InteractionType,
        state: SimulationState,
        source_user_id: str | None = None,
    ) -> Interaction:
        """Create an interaction object.

        Args:
            user: User interacting
            post: Post being interacted with
            interaction_type: Type of interaction
            state: Simulation state
            source_user_id: Source user for cascade tracking

        Returns:
            Interaction object
        """
        self.interaction_counter += 1
        interaction_id = f"int_{self.interaction_counter:010d}"

        return Interaction(
            interaction_id=interaction_id,
            user_id=user.user_id,
            post_id=post.post_id,
            interaction_type=interaction_type,
            step=state.current_step,
            cascade_id=post.cascade_id,
            source_user_id=source_user_id,
        )

    def should_user_be_active(
        self,
        user: User,
        state: SimulationState,
    ) -> bool:
        """Determine if a user should be active in current step.

        Args:
            user: User to check
            state: Simulation state

        Returns:
            True if user should be active
        """
        if not user.is_active():
            return False

        # Base probability from activity level
        base_prob = user.traits.activity_level

        # Event boost
        event_effect = state.get_combined_event_effect()
        base_prob *= event_effect.activity_boost

        # Fatigue penalty
        user_state = state.get_user_state(user.user_id)
        if user_state:
            base_prob *= (1 - user_state.fatigue * 0.3)

        return self.rng.random() < base_prob

    def should_user_post(
        self,
        user: User,
        state: SimulationState,
        avg_posts_per_step: float,
    ) -> bool:
        """Determine if a user should create a post.

        Args:
            user: User to check
            state: Simulation state
            avg_posts_per_step: Average posts per user per step

        Returns:
            True if user should create a post
        """
        if not user.is_active():
            return False

        # Base probability
        base_prob = avg_posts_per_step * user.traits.activity_level

        # Daily limit check
        user_state = state.get_user_state(user.user_id)
        if user_state and user_state.daily_post_count >= 5:
            base_prob *= 0.1  # Heavily reduce if already posted a lot

        # Event boost
        event_effect = state.get_combined_event_effect()
        base_prob *= event_effect.activity_boost

        # Influence boost (influential users post more)
        base_prob *= (1 + user.influence_score * 0.5)

        return self.rng.random() < base_prob
