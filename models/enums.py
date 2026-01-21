"""Enumerations for SocialNetSim."""

from enum import Enum, auto


class InteractionType(str, Enum):
    """Types of user interactions with content."""

    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"

    def __str__(self) -> str:
        return self.value


class EventType(str, Enum):
    """Types of random events that can occur in simulation."""

    POLITICAL_SHOCK = "political_shock"
    MISINFO_WAVE = "misinfo_wave"
    VIRAL_TREND = "viral_trend"
    ALGORITHM_CHANGE = "algorithm_change"
    EXTERNAL_EVENT = "external_event"

    def __str__(self) -> str:
        return self.value


class FeedAlgorithm(str, Enum):
    """Feed ranking algorithm types."""

    CHRONOLOGICAL = "chronological"
    ENGAGEMENT = "engagement"
    DIVERSE = "diverse"
    INTEREST = "interest"

    def __str__(self) -> str:
        return self.value


class UserState(str, Enum):
    """User activity states."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    CHURNED = "churned"
    SUSPENDED = "suspended"

    def __str__(self) -> str:
        return self.value


class PostState(str, Enum):
    """Post visibility states."""

    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    REMOVED = "removed"
    EXPIRED = "expired"

    def __str__(self) -> str:
        return self.value


class Sentiment(str, Enum):
    """Content sentiment types."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

    def __str__(self) -> str:
        return self.value


class EmotionType(str, Enum):
    """Primary emotion types for content."""

    JOY = "joy"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

    def __str__(self) -> str:
        return self.value
