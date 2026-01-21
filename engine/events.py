"""Random event system."""

from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import EventConfig
from models import Event
from models.enums import EventType
from generator.topic_generator import Topic
from .state import SimulationState


class EventEngine:
    """Manages random events in the simulation.

    Event types:
    - Political shock: Major political event causing increased engagement
    - Misinformation wave: Increased spread of misinformation
    - Viral trend: A topic goes viral
    - Algorithm change: Platform algorithm temporarily changes
    - External event: News or external factors affecting engagement
    """

    def __init__(
        self,
        config: EventConfig,
        topics: dict[str, Topic],
        seed: int | None = None,
    ):
        """Initialize event engine.

        Args:
            config: Event configuration
            topics: Available topics
            seed: Random seed
        """
        self.config = config
        self.topics = topics
        self.rng = np.random.default_rng(seed)
        self.event_counter = 0

        # Normalize event type probabilities
        total_prob = (
            config.political_shock_prob +
            config.misinfo_wave_prob +
            config.viral_trend_prob +
            config.algorithm_change_prob +
            config.external_event_prob
        )
        self.event_probs = {
            EventType.POLITICAL_SHOCK: config.political_shock_prob / total_prob,
            EventType.MISINFO_WAVE: config.misinfo_wave_prob / total_prob,
            EventType.VIRAL_TREND: config.viral_trend_prob / total_prob,
            EventType.ALGORITHM_CHANGE: config.algorithm_change_prob / total_prob,
            EventType.EXTERNAL_EVENT: config.external_event_prob / total_prob,
        }

    def process_step(self, state: SimulationState) -> Event | None:
        """Process events for current step.

        May generate a new event based on probability.

        Args:
            state: Simulation state

        Returns:
            New event if generated, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check if new event should occur
        if self.rng.random() < self.config.event_probability:
            # Select event type
            event_type = self._select_event_type()

            # Generate event
            event = self._generate_event(event_type, state)
            return event

        return None

    def _select_event_type(self) -> EventType:
        """Select event type based on configured probabilities.

        Returns:
            Selected EventType
        """
        types = list(self.event_probs.keys())
        probs = list(self.event_probs.values())

        # Use index-based selection to avoid numpy string conversion
        selected_idx = self.rng.choice(len(types), p=probs)
        return types[selected_idx]

    def _generate_event(
        self,
        event_type: EventType,
        state: SimulationState,
    ) -> Event:
        """Generate an event of the specified type.

        Args:
            event_type: Type of event to generate
            state: Simulation state

        Returns:
            Generated Event
        """
        self.event_counter += 1
        event_id = f"event_{self.event_counter:06d}"

        # Generate duration
        duration = self.rng.integers(
            self.config.min_duration,
            self.config.max_duration + 1,
        )

        # Select affected topics
        affected_topics = self._select_affected_topics(event_type)

        # Generate engagement multiplier
        min_mult, max_mult = self.config.engagement_multiplier_range
        engagement_multiplier = float(self.rng.uniform(min_mult, max_mult))

        # Create event based on type
        if event_type == EventType.POLITICAL_SHOCK:
            event = Event.create_political_shock(
                event_id=event_id,
                start_step=state.current_step,
                duration=duration,
                topics=affected_topics,
                engagement_multiplier=engagement_multiplier,
            )
        elif event_type == EventType.MISINFO_WAVE:
            misinfo_boost = float(self.rng.uniform(1.5, 3.0))
            event = Event.create_misinfo_wave(
                event_id=event_id,
                start_step=state.current_step,
                duration=duration,
                topics=affected_topics,
                misinfo_boost=misinfo_boost,
            )
        elif event_type == EventType.VIRAL_TREND:
            topic_boost = float(self.rng.uniform(2.0, 5.0))
            event = Event.create_viral_trend(
                event_id=event_id,
                start_step=state.current_step,
                duration=duration,
                topics=affected_topics,
                topic_boost=topic_boost,
            )
        elif event_type == EventType.ALGORITHM_CHANGE:
            feed_weights = self._generate_algorithm_change()
            event = Event.create_algorithm_change(
                event_id=event_id,
                start_step=state.current_step,
                duration=duration,
                feed_weights=feed_weights,
            )
        else:  # EXTERNAL_EVENT
            event = Event.create_external_event(
                event_id=event_id,
                start_step=state.current_step,
                duration=duration,
                topics=affected_topics,
                engagement_multiplier=engagement_multiplier,
            )

        return event

    def _select_affected_topics(
        self,
        event_type: EventType,
        n_topics: int | None = None,
    ) -> set[str]:
        """Select topics affected by an event.

        Args:
            event_type: Type of event
            n_topics: Number of topics (random if None)

        Returns:
            Set of affected topic IDs
        """
        topic_list = list(self.topics.values())

        if n_topics is None:
            n_topics = self.rng.integers(1, 4)

        # Different events prefer different topic types
        if event_type == EventType.POLITICAL_SHOCK:
            # Prefer political and controversial topics
            weights = np.array([
                (2.0 if t.category == "politics" else 1.0) *
                (1 + t.controversy_score)
                for t in topic_list
            ])
        elif event_type == EventType.VIRAL_TREND:
            # Prefer popular topics
            weights = np.array([t.popularity for t in topic_list])
        elif event_type == EventType.MISINFO_WAVE:
            # Prefer health, politics, controversial
            weights = np.array([
                (2.0 if t.category in ("health", "politics", "science") else 1.0) *
                (1 + t.controversy_score * 0.5)
                for t in topic_list
            ])
        else:
            # Random weights
            weights = np.array([t.popularity + 0.1 for t in topic_list])

        weights = weights / weights.sum()

        selected_indices = self.rng.choice(
            len(topic_list),
            size=min(n_topics, len(topic_list)),
            replace=False,
            p=weights,
        )

        return {topic_list[i].topic_id for i in selected_indices}

    def _generate_algorithm_change(self) -> dict[str, float]:
        """Generate algorithm weight changes.

        Returns:
            Dictionary of weight overrides
        """
        # Randomly modify feed algorithm weights
        changes = {}

        # Possible changes
        possible_changes = [
            ("recency_weight", self.rng.uniform(0.1, 0.6)),
            ("velocity_weight", self.rng.uniform(0.1, 0.6)),
            ("relevance_weight", self.rng.uniform(0.2, 0.6)),
            ("diversity_penalty", self.rng.uniform(0.0, 0.3)),
        ]

        # Select 1-2 changes
        n_changes = self.rng.integers(1, 3)
        selected = self.rng.choice(len(possible_changes), size=n_changes, replace=False)

        for idx in selected:
            key, value = possible_changes[idx]
            changes[key] = float(value)

        return changes

    def get_active_events(self, state: SimulationState) -> list[Event]:
        """Get currently active events.

        Args:
            state: Simulation state

        Returns:
            List of active events
        """
        return [
            event for event in state.events.values()
            if event.is_occurring(state.current_step)
        ]

    def get_event_history(
        self,
        state: SimulationState,
        n_recent: int | None = None,
    ) -> list[Event]:
        """Get event history.

        Args:
            state: Simulation state
            n_recent: Number of recent events (None for all)

        Returns:
            List of events
        """
        events = sorted(
            state.events.values(),
            key=lambda e: e.start_step,
            reverse=True,
        )

        if n_recent:
            return events[:n_recent]
        return events

    def get_event_statistics(
        self,
        state: SimulationState,
    ) -> dict[str, Any]:
        """Get event statistics.

        Args:
            state: Simulation state

        Returns:
            Dictionary of statistics
        """
        events = list(state.events.values())

        if not events:
            return {"total_events": 0}

        # Count by type
        type_counts = {}
        for event in events:
            event_type = str(event.event_type)
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Average duration
        avg_duration = np.mean([e.duration for e in events])

        return {
            "total_events": len(events),
            "active_events": len(self.get_active_events(state)),
            "events_by_type": type_counts,
            "average_duration": float(avg_duration),
        }
