"""Information diffusion model with SIR-like dynamics.

Implements:
- Susceptible-Infected-Recovered style spreading
- Content fatigue and saturation
- Backlash dynamics from overexposure
- Immunity and recovery
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import math

import numpy as np
from numpy.random import Generator


class DiffusionState(Enum):
    """User state in diffusion process."""

    SUSCEPTIBLE = "susceptible"  # Can be exposed
    EXPOSED = "exposed"          # Has seen, may share
    INFECTED = "infected"        # Has shared
    RECOVERED = "recovered"      # Won't share again (temporary)
    IMMUNE = "immune"            # Won't be exposed


@dataclass
class UserDiffusionState:
    """Diffusion state for a single user.

    Attributes:
        user_id: User identifier
        state: Current diffusion state
        exposure_count: How many times exposed
        exposure_step: Step when first exposed
        infection_step: Step when shared (if infected)
        recovery_step: Step when recovered (if recovered)
        fatigue_level: Content fatigue (0-1)
        sentiment: Sentiment toward content (-1 to 1)
    """

    user_id: str
    state: DiffusionState = DiffusionState.SUSCEPTIBLE
    exposure_count: int = 0
    exposure_step: int | None = None
    infection_step: int | None = None
    recovery_step: int | None = None
    fatigue_level: float = 0.0
    sentiment: float = 0.0

    def expose(self, step: int) -> None:
        """Record exposure."""
        if self.state == DiffusionState.SUSCEPTIBLE:
            self.state = DiffusionState.EXPOSED
            self.exposure_step = step
        self.exposure_count += 1

    def infect(self, step: int) -> None:
        """Record infection (share)."""
        self.state = DiffusionState.INFECTED
        self.infection_step = step

    def recover(self, step: int) -> None:
        """Record recovery."""
        self.state = DiffusionState.RECOVERED
        self.recovery_step = step

    def add_fatigue(self, amount: float) -> None:
        """Add content fatigue."""
        self.fatigue_level = min(1.0, self.fatigue_level + amount)

    def decay_fatigue(self, decay_rate: float) -> None:
        """Decay fatigue over time."""
        self.fatigue_level = max(0.0, self.fatigue_level - decay_rate)

    def update_sentiment(self, delta: float) -> None:
        """Update sentiment toward content."""
        self.sentiment = np.clip(self.sentiment + delta, -1.0, 1.0)


@dataclass
class DiffusionConfig:
    """Configuration for information diffusion model.

    Attributes:
        saturation_constant: Share count for 50% saturation
        fatigue_rate: Fatigue increase per exposure
        fatigue_decay: Fatigue decay per step
        backlash_threshold: Exposure count triggering backlash
        backlash_strength: Strength of negative sentiment from backlash
        immunity_duration: Steps of immunity after recovery
        recovery_probability: Base probability of recovery per step
        exposure_decay: Decay of exposure probability over time
    """

    saturation_constant: float = 100.0
    fatigue_rate: float = 0.1
    fatigue_decay: float = 0.02
    backlash_threshold: int = 5
    backlash_strength: float = 0.3
    immunity_duration: int = 20
    recovery_probability: float = 0.1
    exposure_decay: float = 0.05


class InformationDiffusionModel:
    """Models information diffusion with fatigue and saturation.

    Features:
    - SIR-like state transitions
    - Content saturation (diminishing returns)
    - User fatigue from repeated exposure
    - Backlash from overexposure
    - Temporary immunity after sharing
    """

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize diffusion model.

        Args:
            config: Diffusion configuration
            seed: Random seed
        """
        self.config = config or DiffusionConfig()
        self.rng = np.random.default_rng(seed)

        # User states per cascade
        self.user_states: dict[str, dict[str, UserDiffusionState]] = {}

        # Global metrics
        self.total_exposures = 0
        self.total_infections = 0

    def initialize_cascade(self, cascade_id: str, initial_users: list[str]) -> None:
        """Initialize diffusion tracking for a cascade.

        Args:
            cascade_id: Cascade identifier
            initial_users: Users initially infected (sharers)
        """
        self.user_states[cascade_id] = {}

        for user_id in initial_users:
            state = UserDiffusionState(user_id=user_id)
            state.infect(step=0)
            self.user_states[cascade_id][user_id] = state

    def get_user_state(
        self,
        cascade_id: str,
        user_id: str,
    ) -> UserDiffusionState:
        """Get user's diffusion state for a cascade.

        Args:
            cascade_id: Cascade identifier
            user_id: User identifier

        Returns:
            User's diffusion state
        """
        if cascade_id not in self.user_states:
            self.user_states[cascade_id] = {}

        if user_id not in self.user_states[cascade_id]:
            self.user_states[cascade_id][user_id] = UserDiffusionState(user_id=user_id)

        return self.user_states[cascade_id][user_id]

    def calculate_saturation(self, share_count: int) -> float:
        """Calculate market saturation factor.

        saturation = 1 - exp(-share_count / saturation_constant)

        Returns value in [0, 1] where:
        - 0 = no saturation (content is fresh)
        - 1 = fully saturated (everyone who will share has shared)

        Args:
            share_count: Current share count

        Returns:
            Saturation factor
        """
        return 1 - math.exp(-share_count / self.config.saturation_constant)

    def calculate_exposure_probability(
        self,
        user_state: UserDiffusionState,
        cascade_age: int,
        share_count: int,
        base_probability: float,
    ) -> float:
        """Calculate probability of user being exposed.

        Factors:
        - Base probability
        - Cascade age decay
        - Saturation
        - User fatigue

        Args:
            user_state: User's diffusion state
            cascade_age: Age of cascade in steps
            share_count: Current share count
            base_probability: Base exposure probability

        Returns:
            Adjusted exposure probability
        """
        # Already infected or immune
        if user_state.state in (DiffusionState.INFECTED, DiffusionState.IMMUNE):
            return 0.0

        # Time decay
        time_factor = math.exp(-cascade_age * self.config.exposure_decay)

        # Saturation reduces fresh exposure
        saturation = self.calculate_saturation(share_count)
        saturation_factor = 1 - saturation * 0.5

        # User fatigue reduces receptivity
        fatigue_factor = 1 - user_state.fatigue_level * 0.5

        probability = (
            base_probability *
            time_factor *
            saturation_factor *
            fatigue_factor
        )

        return min(0.9, probability)

    def calculate_infection_probability(
        self,
        user_state: UserDiffusionState,
        base_share_prob: float,
        friend_count_shared: int,
    ) -> float:
        """Calculate probability of sharing (infection).

        Factors:
        - Base share probability
        - Social proof (friends who shared)
        - User sentiment (backlash reduces)
        - Fatigue

        Args:
            user_state: User's diffusion state
            base_share_prob: Base share probability
            friend_count_shared: Number of friends who shared

        Returns:
            Adjusted share probability
        """
        if user_state.state != DiffusionState.EXPOSED:
            return 0.0

        # Social proof boost
        social_factor = 1 + math.log1p(friend_count_shared) * 0.3

        # Sentiment affects sharing (negative sentiment = less likely)
        sentiment_factor = 1 + user_state.sentiment * 0.3

        # Fatigue reduces engagement
        fatigue_factor = 1 - user_state.fatigue_level * 0.5

        probability = (
            base_share_prob *
            social_factor *
            sentiment_factor *
            fatigue_factor
        )

        return min(0.8, max(0.0, probability))

    def process_exposure(
        self,
        cascade_id: str,
        user_id: str,
        step: int,
        cascade_age: int,
        share_count: int,
        base_probability: float,
    ) -> bool:
        """Process potential exposure of a user.

        Args:
            cascade_id: Cascade identifier
            user_id: User identifier
            step: Current step
            cascade_age: Age of cascade
            share_count: Current share count
            base_probability: Base exposure probability

        Returns:
            True if user was exposed
        """
        user_state = self.get_user_state(cascade_id, user_id)

        # Check if exposure happens
        prob = self.calculate_exposure_probability(
            user_state, cascade_age, share_count, base_probability
        )

        if self.rng.random() < prob:
            user_state.expose(step)
            user_state.add_fatigue(self.config.fatigue_rate)
            self.total_exposures += 1

            # Check for backlash
            if user_state.exposure_count >= self.config.backlash_threshold:
                user_state.update_sentiment(-self.config.backlash_strength)

            return True

        return False

    def process_infection(
        self,
        cascade_id: str,
        user_id: str,
        step: int,
        base_share_prob: float,
        friend_count_shared: int,
    ) -> bool:
        """Process potential infection (share) by a user.

        Args:
            cascade_id: Cascade identifier
            user_id: User identifier
            step: Current step
            base_share_prob: Base share probability
            friend_count_shared: Friends who shared

        Returns:
            True if user shared (became infected)
        """
        user_state = self.get_user_state(cascade_id, user_id)

        prob = self.calculate_infection_probability(
            user_state, base_share_prob, friend_count_shared
        )

        if self.rng.random() < prob:
            user_state.infect(step)
            self.total_infections += 1
            return True

        return False

    def process_recovery(
        self,
        cascade_id: str,
        user_id: str,
        step: int,
    ) -> bool:
        """Process potential recovery of an infected user.

        Args:
            cascade_id: Cascade identifier
            user_id: User identifier
            step: Current step

        Returns:
            True if user recovered
        """
        user_state = self.get_user_state(cascade_id, user_id)

        if user_state.state != DiffusionState.INFECTED:
            return False

        # Time since infection
        time_infected = step - (user_state.infection_step or step)

        # Recovery probability increases over time
        recovery_prob = self.config.recovery_probability * (1 + time_infected * 0.1)

        if self.rng.random() < recovery_prob:
            user_state.recover(step)
            return True

        return False

    def check_immunity_expiration(
        self,
        cascade_id: str,
        user_id: str,
        step: int,
    ) -> None:
        """Check and handle immunity expiration.

        Args:
            cascade_id: Cascade identifier
            user_id: User identifier
            step: Current step
        """
        user_state = self.get_user_state(cascade_id, user_id)

        if user_state.state != DiffusionState.RECOVERED:
            return

        if user_state.recovery_step is None:
            return

        time_recovered = step - user_state.recovery_step
        if time_recovered >= self.config.immunity_duration:
            # Return to susceptible but with lower susceptibility
            user_state.state = DiffusionState.IMMUNE

    def step_cascade(
        self,
        cascade_id: str,
        step: int,
    ) -> None:
        """Process one step for a cascade.

        - Decay fatigue
        - Check immunity expiration
        - Process recoveries

        Args:
            cascade_id: Cascade identifier
            step: Current step
        """
        if cascade_id not in self.user_states:
            return

        for user_state in self.user_states[cascade_id].values():
            # Decay fatigue
            user_state.decay_fatigue(self.config.fatigue_decay)

            # Check immunity
            self.check_immunity_expiration(cascade_id, user_state.user_id, step)

            # Process recovery
            self.process_recovery(cascade_id, user_state.user_id, step)

    def get_cascade_statistics(self, cascade_id: str) -> dict[str, Any]:
        """Get diffusion statistics for a cascade.

        Args:
            cascade_id: Cascade identifier

        Returns:
            Dictionary of statistics
        """
        if cascade_id not in self.user_states:
            return {"error": "Unknown cascade"}

        states = self.user_states[cascade_id]

        state_counts = {state: 0 for state in DiffusionState}
        for user_state in states.values():
            state_counts[user_state.state] += 1

        avg_fatigue = np.mean([s.fatigue_level for s in states.values()]) if states else 0
        avg_sentiment = np.mean([s.sentiment for s in states.values()]) if states else 0

        return {
            "total_users": len(states),
            "state_counts": {s.value: c for s, c in state_counts.items()},
            "avg_fatigue": avg_fatigue,
            "avg_sentiment": avg_sentiment,
        }

    def get_effective_reproduction_number(
        self,
        cascade_id: str,
        share_count: int,
    ) -> float:
        """Calculate effective reproduction number (R_t).

        R_t decreases as saturation increases and backlash grows.

        Args:
            cascade_id: Cascade identifier
            share_count: Current share count

        Returns:
            Effective R value
        """
        # Base R from branching ratio assumption
        base_r = 0.8

        # Saturation reduces R
        saturation = self.calculate_saturation(share_count)
        saturation_factor = 1 - saturation

        # Backlash reduces R
        if cascade_id in self.user_states:
            states = self.user_states[cascade_id]
            avg_sentiment = np.mean([s.sentiment for s in states.values()]) if states else 0
            sentiment_factor = max(0.1, 1 + avg_sentiment)
        else:
            sentiment_factor = 1.0

        return base_r * saturation_factor * sentiment_factor


@dataclass
class DelayedEffect:
    """A delayed effect to be applied later.

    Attributes:
        trigger_step: Step when effect triggers
        effect_type: Type of effect
        target_id: Target user or cascade
        magnitude: Effect magnitude
        metadata: Additional data
    """

    trigger_step: int
    effect_type: str
    target_id: str
    magnitude: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DelayedEffectsManager:
    """Manages delayed effects in diffusion (e.g., delayed sharing)."""

    def __init__(self):
        """Initialize delayed effects manager."""
        self.pending_effects: list[DelayedEffect] = []

    def schedule_effect(
        self,
        delay: int,
        current_step: int,
        effect_type: str,
        target_id: str,
        magnitude: float,
        **metadata,
    ) -> None:
        """Schedule a delayed effect.

        Args:
            delay: Steps until effect triggers
            current_step: Current simulation step
            effect_type: Type of effect
            target_id: Target ID
            magnitude: Effect magnitude
            **metadata: Additional metadata
        """
        effect = DelayedEffect(
            trigger_step=current_step + delay,
            effect_type=effect_type,
            target_id=target_id,
            magnitude=magnitude,
            metadata=metadata,
        )
        self.pending_effects.append(effect)

    def get_triggered_effects(self, step: int) -> list[DelayedEffect]:
        """Get effects that trigger at given step.

        Args:
            step: Current step

        Returns:
            List of triggered effects
        """
        triggered = [e for e in self.pending_effects if e.trigger_step == step]
        self.pending_effects = [e for e in self.pending_effects if e.trigger_step > step]
        return triggered

    def clear(self) -> None:
        """Clear all pending effects."""
        self.pending_effects = []
