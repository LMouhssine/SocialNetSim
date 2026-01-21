"""Event models for simulation events."""

from dataclasses import dataclass, field
from typing import Any

from .enums import EventType


@dataclass
class EventEffect:
    """Effects that an event has on the simulation.

    Attributes:
        engagement_multiplier: Multiplier for engagement rates
        topic_boost: Topics that get boosted engagement (topic_id -> multiplier)
        ideology_activation: Boost for ideologically charged content
        misinfo_boost: Multiplier for misinformation spread
        activity_boost: Multiplier for user activity
        feed_weight_override: Temporary feed algorithm weight changes
        moderation_change: Changes to moderation settings
    """

    engagement_multiplier: float = 1.0
    topic_boost: dict[str, float] = field(default_factory=dict)
    ideology_activation: float = 0.0
    misinfo_boost: float = 1.0
    activity_boost: float = 1.0
    feed_weight_override: dict[str, float] = field(default_factory=dict)
    moderation_change: dict[str, Any] = field(default_factory=dict)

    def apply_engagement_modifier(self, base_rate: float) -> float:
        """Apply engagement multiplier to a base rate."""
        return base_rate * self.engagement_multiplier * self.activity_boost

    def get_topic_multiplier(self, topic_id: str) -> float:
        """Get multiplier for a specific topic."""
        return self.topic_boost.get(topic_id, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert effect to dictionary."""
        return {
            "engagement_multiplier": self.engagement_multiplier,
            "topic_boost": self.topic_boost,
            "ideology_activation": self.ideology_activation,
            "misinfo_boost": self.misinfo_boost,
            "activity_boost": self.activity_boost,
            "feed_weight_override": self.feed_weight_override,
            "moderation_change": self.moderation_change,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventEffect":
        """Create effect from dictionary."""
        return cls(
            engagement_multiplier=data.get("engagement_multiplier", 1.0),
            topic_boost=data.get("topic_boost", {}),
            ideology_activation=data.get("ideology_activation", 0.0),
            misinfo_boost=data.get("misinfo_boost", 1.0),
            activity_boost=data.get("activity_boost", 1.0),
            feed_weight_override=data.get("feed_weight_override", {}),
            moderation_change=data.get("moderation_change", {}),
        )

    @classmethod
    def combine(cls, effects: list["EventEffect"]) -> "EventEffect":
        """Combine multiple effects into one."""
        if not effects:
            return cls()

        combined = cls(
            engagement_multiplier=1.0,
            activity_boost=1.0,
            misinfo_boost=1.0,
        )

        for effect in effects:
            combined.engagement_multiplier *= effect.engagement_multiplier
            combined.activity_boost *= effect.activity_boost
            combined.misinfo_boost *= effect.misinfo_boost
            combined.ideology_activation = max(
                combined.ideology_activation, effect.ideology_activation
            )

            # Merge topic boosts (take max)
            for topic, boost in effect.topic_boost.items():
                if topic in combined.topic_boost:
                    combined.topic_boost[topic] = max(combined.topic_boost[topic], boost)
                else:
                    combined.topic_boost[topic] = boost

            # Merge feed overrides (later overwrites)
            combined.feed_weight_override.update(effect.feed_weight_override)
            combined.moderation_change.update(effect.moderation_change)

        return combined


@dataclass
class Event:
    """Represents a random event in the simulation.

    Attributes:
        event_id: Unique identifier
        event_type: Type of event
        name: Human-readable name
        description: Event description
        start_step: Step when event starts
        duration: Number of steps event lasts
        effect: Effects on simulation
        affected_topics: Topics particularly affected
        is_active: Whether event is currently active
        metadata: Additional event data
    """

    event_id: str
    event_type: EventType
    name: str = ""
    description: str = ""
    start_step: int = 0
    duration: int = 10
    effect: EventEffect = field(default_factory=EventEffect)
    affected_topics: set[str] = field(default_factory=set)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_step(self) -> int:
        """Calculate end step."""
        return self.start_step + self.duration

    def is_occurring(self, step: int) -> bool:
        """Check if event is occurring at given step."""
        return self.is_active and self.start_step <= step < self.end_step

    def get_remaining_duration(self, current_step: int) -> int:
        """Get remaining duration at current step."""
        if not self.is_occurring(current_step):
            return 0
        return self.end_step - current_step

    def get_intensity(self, current_step: int) -> float:
        """Get event intensity at current step (can decay over time)."""
        if not self.is_occurring(current_step):
            return 0.0

        # Linear decay over duration
        elapsed = current_step - self.start_step
        remaining_ratio = 1 - (elapsed / self.duration)
        return max(0.0, remaining_ratio)

    def deactivate(self) -> None:
        """Deactivate the event."""
        self.is_active = False

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": str(self.event_type),
            "name": self.name,
            "description": self.description,
            "start_step": self.start_step,
            "duration": self.duration,
            "effect": self.effect.to_dict(),
            "affected_topics": list(self.affected_topics),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            start_step=data.get("start_step", 0),
            duration=data.get("duration", 10),
            effect=EventEffect.from_dict(data.get("effect", {})),
            affected_topics=set(data.get("affected_topics", [])),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create_political_shock(
        cls,
        event_id: str,
        start_step: int,
        duration: int,
        topics: set[str],
        engagement_multiplier: float = 2.0,
    ) -> "Event":
        """Factory method for political shock events."""
        effect = EventEffect(
            engagement_multiplier=engagement_multiplier,
            topic_boost={t: 2.0 for t in topics},
            ideology_activation=0.5,
            activity_boost=1.5,
        )
        return cls(
            event_id=event_id,
            event_type=EventType.POLITICAL_SHOCK,
            name="Political Shock",
            description="A major political event causing increased engagement",
            start_step=start_step,
            duration=duration,
            effect=effect,
            affected_topics=topics,
        )

    @classmethod
    def create_misinfo_wave(
        cls,
        event_id: str,
        start_step: int,
        duration: int,
        topics: set[str],
        misinfo_boost: float = 2.0,
    ) -> "Event":
        """Factory method for misinformation wave events."""
        effect = EventEffect(
            engagement_multiplier=1.5,
            topic_boost={t: 1.5 for t in topics},
            misinfo_boost=misinfo_boost,
            activity_boost=1.3,
        )
        return cls(
            event_id=event_id,
            event_type=EventType.MISINFO_WAVE,
            name="Misinformation Wave",
            description="Increased spread of misinformation",
            start_step=start_step,
            duration=duration,
            effect=effect,
            affected_topics=topics,
        )

    @classmethod
    def create_viral_trend(
        cls,
        event_id: str,
        start_step: int,
        duration: int,
        topics: set[str],
        topic_boost: float = 3.0,
    ) -> "Event":
        """Factory method for viral trend events."""
        effect = EventEffect(
            engagement_multiplier=1.8,
            topic_boost={t: topic_boost for t in topics},
            activity_boost=1.2,
        )
        return cls(
            event_id=event_id,
            event_type=EventType.VIRAL_TREND,
            name="Viral Trend",
            description="A topic going viral",
            start_step=start_step,
            duration=duration,
            effect=effect,
            affected_topics=topics,
        )

    @classmethod
    def create_algorithm_change(
        cls,
        event_id: str,
        start_step: int,
        duration: int,
        feed_weights: dict[str, float],
    ) -> "Event":
        """Factory method for algorithm change events."""
        effect = EventEffect(
            feed_weight_override=feed_weights,
        )
        return cls(
            event_id=event_id,
            event_type=EventType.ALGORITHM_CHANGE,
            name="Algorithm Change",
            description="Platform algorithm temporarily changed",
            start_step=start_step,
            duration=duration,
            effect=effect,
        )

    @classmethod
    def create_external_event(
        cls,
        event_id: str,
        start_step: int,
        duration: int,
        topics: set[str],
        engagement_multiplier: float = 1.5,
    ) -> "Event":
        """Factory method for external events (news, etc.)."""
        effect = EventEffect(
            engagement_multiplier=engagement_multiplier,
            topic_boost={t: 2.0 for t in topics},
            activity_boost=1.4,
        )
        return cls(
            event_id=event_id,
            event_type=EventType.EXTERNAL_EVENT,
            name="External Event",
            description="External event affecting network activity",
            start_step=start_step,
            duration=duration,
            effect=effect,
            affected_topics=topics,
        )
