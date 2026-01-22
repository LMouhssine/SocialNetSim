"""Utility-based decision model for user behavior.

Implements a rational agent model where users make decisions
by maximizing expected utility across multiple dimensions:
- Information gain (topic novelty)
- Social utility (author relationship)
- Emotional resonance (content vs. user state)
- Cognitive load cost (attention depletion)
- Ideological alignment
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import numpy as np
from numpy.random import Generator

from models import User, Post
from engine.state import SimulationState


class DecisionType(Enum):
    """Types of user decisions."""

    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    POST = "post"
    FOLLOW = "follow"


@dataclass
class UtilityWeights:
    """Weights for utility components.

    Attributes:
        information_gain: Weight for novelty/information value
        social_utility: Weight for social relationship factors
        emotional_resonance: Weight for emotional content match
        cognitive_cost: Weight for attention/effort cost
        ideological_alignment: Weight for belief consistency
        quality: Weight for content quality
    """

    information_gain: float = 0.2
    social_utility: float = 0.25
    emotional_resonance: float = 0.15
    cognitive_cost: float = 0.15
    ideological_alignment: float = 0.15
    quality: float = 0.1

    def to_array(self) -> np.ndarray:
        """Convert weights to numpy array."""
        return np.array([
            self.information_gain,
            self.social_utility,
            self.emotional_resonance,
            self.cognitive_cost,
            self.ideological_alignment,
            self.quality,
        ])

    def normalize(self) -> "UtilityWeights":
        """Return normalized weights summing to 1."""
        total = (
            self.information_gain +
            self.social_utility +
            self.emotional_resonance +
            self.cognitive_cost +
            self.ideological_alignment +
            self.quality
        )
        if total == 0:
            return self
        return UtilityWeights(
            information_gain=self.information_gain / total,
            social_utility=self.social_utility / total,
            emotional_resonance=self.emotional_resonance / total,
            cognitive_cost=self.cognitive_cost / total,
            ideological_alignment=self.ideological_alignment / total,
            quality=self.quality / total,
        )


@dataclass
class UtilityComponents:
    """Individual utility component values.

    Attributes:
        information_gain: Novelty value (0-1)
        social_utility: Social relationship value (0-1)
        emotional_resonance: Emotional match value (0-1)
        cognitive_cost: Cost of engagement (0-1, higher = more costly)
        ideological_alignment: Belief alignment value (0-1)
        quality: Content quality value (0-1)
    """

    information_gain: float = 0.5
    social_utility: float = 0.5
    emotional_resonance: float = 0.5
    cognitive_cost: float = 0.3
    ideological_alignment: float = 0.5
    quality: float = 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.information_gain,
            self.social_utility,
            self.emotional_resonance,
            1.0 - self.cognitive_cost,  # Invert cost to utility
            self.ideological_alignment,
            self.quality,
        ])

    def compute_total_utility(self, weights: UtilityWeights) -> float:
        """Compute weighted total utility.

        Args:
            weights: Utility weights

        Returns:
            Total utility value
        """
        components = self.to_array()
        weight_array = weights.normalize().to_array()
        return float(np.dot(components, weight_array))


@dataclass
class DecisionConfig:
    """Configuration for utility-based decision model.

    Attributes:
        base_weights: Default utility weights
        noise_scale: Scale of decision noise (bounded rationality)
        attention_cost_per_engagement: Attention cost per interaction
        emotional_decay_per_step: How quickly emotional state decays
        memory_influence_decay: Decay rate for memory influence
        confirmation_bias_amplifier: How much confirmation bias affects ideology weight
    """

    base_weights: UtilityWeights = field(default_factory=UtilityWeights)
    noise_scale: float = 0.1
    attention_cost_per_engagement: float = 0.05
    emotional_decay_per_step: float = 0.1
    memory_influence_decay: float = 0.95
    confirmation_bias_amplifier: float = 1.5


class UtilityBasedDecisionModel:
    """Models user decisions based on utility maximization.

    Users are modeled as bounded rational agents who:
    1. Evaluate content along multiple utility dimensions
    2. Weight dimensions based on personality traits
    3. Make noisy decisions (not perfectly rational)
    4. Update internal states based on decisions
    """

    def __init__(
        self,
        config: DecisionConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize decision model.

        Args:
            config: Decision model configuration
            seed: Random seed
        """
        self.config = config or DecisionConfig()
        self.rng = np.random.default_rng(seed)

    def calculate_utility_components(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        cognitive_state: Any | None = None,
    ) -> UtilityComponents:
        """Calculate utility components for a user-post pair.

        Args:
            user: User evaluating the post
            post: Post being evaluated
            state: Simulation state
            users: All users
            cognitive_state: User's cognitive state (if available)

        Returns:
            UtilityComponents for this user-post pair
        """
        # Information gain: novelty of topics
        information_gain = self._calculate_information_gain(user, post, state)

        # Social utility: relationship with author
        social_utility = self._calculate_social_utility(user, post, state, users)

        # Emotional resonance: match between content and user emotional state
        emotional_resonance = self._calculate_emotional_resonance(
            user, post, cognitive_state
        )

        # Cognitive cost: based on attention budget and content complexity
        cognitive_cost = self._calculate_cognitive_cost(user, post, state, cognitive_state)

        # Ideological alignment: weighted by confirmation bias
        ideological_alignment = self._calculate_ideological_alignment(user, post)

        # Quality: content quality score
        quality = post.content.quality_score

        return UtilityComponents(
            information_gain=information_gain,
            social_utility=social_utility,
            emotional_resonance=emotional_resonance,
            cognitive_cost=cognitive_cost,
            ideological_alignment=ideological_alignment,
            quality=quality,
        )

    def _calculate_information_gain(
        self,
        user: User,
        post: Post,
        state: SimulationState,
    ) -> float:
        """Calculate information gain utility.

        Higher for novel topics the user hasn't seen much.

        Args:
            user: User
            post: Post
            state: Simulation state

        Returns:
            Information gain value (0-1)
        """
        user_state = state.get_user_state(user.user_id)
        if not user_state or not post.content.topics:
            return 0.5

        # Average novelty across post topics
        novelties = []
        for topic in post.content.topics:
            weight = post.content.get_topic_weight(topic)
            novelty = user_state.get_topic_novelty(topic)
            novelties.append(novelty * weight)

        if not novelties:
            return 0.5

        return float(np.mean(novelties))

    def _calculate_social_utility(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
    ) -> float:
        """Calculate social utility based on author relationship.

        Args:
            user: User
            post: Post
            state: Simulation state
            users: All users

        Returns:
            Social utility value (0-1)
        """
        author = users.get(post.author_id)
        if not author:
            return 0.3

        social_score = 0.0

        # Following relationship
        if post.author_id in user.following:
            social_score += 0.4

        # Past interactions with author
        user_state = state.get_user_state(user.user_id)
        if user_state:
            author_interactions = user_state.author_interaction_counts.get(
                post.author_id, 0
            )
            # Diminishing returns from repeated interactions
            social_score += 0.3 * (1 - np.exp(-author_interactions / 10))

        # Author influence and credibility
        social_score += 0.15 * author.influence_score
        social_score += 0.15 * author.credibility_score

        return min(1.0, social_score)

    def _calculate_emotional_resonance(
        self,
        user: User,
        post: Post,
        cognitive_state: Any | None = None,
    ) -> float:
        """Calculate emotional resonance between content and user state.

        Args:
            user: User
            post: Post
            cognitive_state: User's cognitive state

        Returns:
            Emotional resonance value (0-1)
        """
        # Base emotional intensity of content
        content_intensity = post.content.emotional_intensity

        # User's emotional reactivity
        reactivity = user.traits.emotional_reactivity

        # Resonance is higher when emotional content meets reactive user
        base_resonance = content_intensity * reactivity

        # If we have cognitive state, consider emotional alignment
        if cognitive_state is not None:
            # Match between content emotions and user emotional state
            user_arousal = getattr(cognitive_state, "emotional_arousal", 0.5)
            arousal_match = 1 - abs(content_intensity - user_arousal)
            base_resonance = 0.5 * base_resonance + 0.5 * arousal_match

        return np.clip(base_resonance, 0.0, 1.0)

    def _calculate_cognitive_cost(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        cognitive_state: Any | None = None,
    ) -> float:
        """Calculate cognitive cost of engaging with content.

        Higher cost when:
        - User is fatigued
        - Content is complex
        - Attention budget is depleted

        Args:
            user: User
            post: Post
            state: Simulation state
            cognitive_state: User's cognitive state

        Returns:
            Cognitive cost value (0-1)
        """
        # Base fatigue from state
        user_state = state.get_user_state(user.user_id)
        fatigue = user_state.fatigue if user_state else 0.0

        # Content complexity (inversely related to quality for simplicity)
        complexity = 0.5 + 0.5 * (1 - post.content.quality_score)

        # Session interaction count adds to cost
        session_interactions = user_state.session_interactions if user_state else 0
        session_cost = min(0.3, session_interactions * self.config.attention_cost_per_engagement)

        # Attention budget from cognitive state
        if cognitive_state is not None:
            attention_budget = getattr(cognitive_state, "attention_budget", 1.0)
            attention_cost = 1 - attention_budget
        else:
            attention_cost = 0.0

        # Combined cost
        cost = 0.3 * fatigue + 0.2 * complexity + 0.25 * session_cost + 0.25 * attention_cost

        return np.clip(cost, 0.0, 1.0)

    def _calculate_ideological_alignment(
        self,
        user: User,
        post: Post,
    ) -> float:
        """Calculate ideological alignment between user and content.

        Amplified by confirmation bias.

        Args:
            user: User
            post: Post

        Returns:
            Ideological alignment value (0-1)
        """
        # Base alignment (1 when identical, 0 when opposite)
        ideology_diff = abs(user.traits.ideology - post.content.ideology_score)
        base_alignment = 1 - (ideology_diff / 2)  # Normalize to 0-1

        # Confirmation bias effect
        # High confirmation bias makes alignment more important (amplified)
        confirmation_factor = (
            1 + user.traits.confirmation_bias * self.config.confirmation_bias_amplifier
        )

        # Apply confirmation bias
        if base_alignment > 0.5:
            # Aligned content: boost based on confirmation bias
            alignment = 0.5 + (base_alignment - 0.5) * confirmation_factor
        else:
            # Misaligned content: penalty based on confirmation bias
            alignment = 0.5 - (0.5 - base_alignment) * confirmation_factor

        return np.clip(alignment, 0.0, 1.0)

    def get_user_weights(
        self,
        user: User,
        cognitive_state: Any | None = None,
    ) -> UtilityWeights:
        """Get personalized utility weights for a user.

        Weights are adjusted based on user traits.

        Args:
            user: User
            cognitive_state: User's cognitive state

        Returns:
            Personalized UtilityWeights
        """
        base = self.config.base_weights

        # Adjust based on traits
        # High confirmation bias → higher ideology weight
        ideology_weight = base.ideological_alignment * (
            1 + user.traits.confirmation_bias * 0.5
        )

        # High emotional reactivity → higher emotional weight
        emotional_weight = base.emotional_resonance * (
            1 + user.traits.emotional_reactivity * 0.5
        )

        # Low activity level → higher cognitive cost sensitivity
        activity_factor = 1 + (1 - user.traits.activity_level) * 0.3
        cognitive_weight = base.cognitive_cost * activity_factor

        # If cognitive state shows depleted attention, increase cognitive cost weight
        if cognitive_state is not None:
            attention_budget = getattr(cognitive_state, "attention_budget", 1.0)
            if attention_budget < 0.5:
                cognitive_weight *= 1.5

        return UtilityWeights(
            information_gain=base.information_gain,
            social_utility=base.social_utility,
            emotional_resonance=emotional_weight,
            cognitive_cost=cognitive_weight,
            ideological_alignment=ideology_weight,
            quality=base.quality,
        )

    def compute_engagement_utility(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        cognitive_state: Any | None = None,
    ) -> float:
        """Compute total utility for engaging with a post.

        Args:
            user: User
            post: Post
            state: Simulation state
            users: All users
            cognitive_state: User's cognitive state

        Returns:
            Total utility value (0-1)
        """
        components = self.calculate_utility_components(
            user, post, state, users, cognitive_state
        )
        weights = self.get_user_weights(user, cognitive_state)
        return components.compute_total_utility(weights)

    def make_decision(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        decision_type: DecisionType,
        base_probability: float,
        cognitive_state: Any | None = None,
    ) -> tuple[bool, float]:
        """Make a utility-based decision with noise.

        Args:
            user: User making decision
            post: Post being considered
            state: Simulation state
            users: All users
            decision_type: Type of decision
            base_probability: Base probability for this action type
            cognitive_state: User's cognitive state

        Returns:
            Tuple of (decision made, probability used)
        """
        # Calculate utility
        utility = self.compute_engagement_utility(
            user, post, state, users, cognitive_state
        )

        # Convert utility to probability with base rate
        # Utility modulates the base probability
        probability = base_probability * (0.5 + utility)

        # Apply decision-specific modifiers
        probability = self._apply_decision_modifiers(
            probability, decision_type, user, post, state, cognitive_state
        )

        # Add noise (bounded rationality)
        noise = self.rng.normal(0, self.config.noise_scale)
        noisy_probability = np.clip(probability + noise, 0.0, 0.95)

        # Make decision
        decision = self.rng.random() < noisy_probability

        return decision, noisy_probability

    def _apply_decision_modifiers(
        self,
        probability: float,
        decision_type: DecisionType,
        user: User,
        post: Post,
        state: SimulationState,
        cognitive_state: Any | None = None,
    ) -> float:
        """Apply decision-type specific modifiers.

        Args:
            probability: Base probability
            decision_type: Type of decision
            user: User
            post: Post
            state: Simulation state
            cognitive_state: User's cognitive state

        Returns:
            Modified probability
        """
        if decision_type == DecisionType.SHARE:
            # Sharing requires higher utility threshold
            # Only share if content is worth endorsing
            probability *= 0.5

            # Misinformation susceptibility affects sharing
            if post.content.is_misinformation:
                probability *= (1 + user.traits.misinfo_susceptibility * 0.5)

        elif decision_type == DecisionType.COMMENT:
            # Comments require more engagement
            # Controversy increases comment probability
            probability *= (1 + post.content.controversy_score * 0.3)

        elif decision_type == DecisionType.LIKE:
            # Liking is low-cost, slightly higher probability
            probability *= 1.2

        return np.clip(probability, 0.0, 0.95)

    def get_decision_explanation(
        self,
        user: User,
        post: Post,
        state: SimulationState,
        users: dict[str, User],
        cognitive_state: Any | None = None,
    ) -> dict[str, Any]:
        """Get detailed explanation of decision factors.

        Useful for analysis and debugging.

        Args:
            user: User
            post: Post
            state: Simulation state
            users: All users
            cognitive_state: User's cognitive state

        Returns:
            Dictionary with detailed decision factors
        """
        components = self.calculate_utility_components(
            user, post, state, users, cognitive_state
        )
        weights = self.get_user_weights(user, cognitive_state)
        utility = components.compute_total_utility(weights)

        return {
            "total_utility": utility,
            "components": {
                "information_gain": components.information_gain,
                "social_utility": components.social_utility,
                "emotional_resonance": components.emotional_resonance,
                "cognitive_cost": components.cognitive_cost,
                "ideological_alignment": components.ideological_alignment,
                "quality": components.quality,
            },
            "weights": {
                "information_gain": weights.information_gain,
                "social_utility": weights.social_utility,
                "emotional_resonance": weights.emotional_resonance,
                "cognitive_cost": weights.cognitive_cost,
                "ideological_alignment": weights.ideological_alignment,
                "quality": weights.quality,
            },
            "user_traits": {
                "ideology": user.traits.ideology,
                "confirmation_bias": user.traits.confirmation_bias,
                "emotional_reactivity": user.traits.emotional_reactivity,
                "activity_level": user.traits.activity_level,
            },
            "post_attributes": {
                "ideology_score": post.content.ideology_score,
                "quality_score": post.content.quality_score,
                "emotional_intensity": post.content.emotional_intensity,
                "controversy_score": post.content.controversy_score,
            },
        }
