"""Post and content models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .enums import PostState, Sentiment, EmotionType


@dataclass
class PostContent:
    """Content characteristics of a post.

    Attributes:
        topics: Set of topic IDs the post relates to
        topic_weights: Weight/relevance of each topic (0-1)
        sentiment: Overall sentiment of the content
        emotions: Emotional content mapping (emotion -> intensity)
        quality_score: Content quality measure (0-1)
        controversy_score: How controversial the content is (0-1)
        ideology_score: Political/ideological leaning (-1 to 1)
        is_misinformation: Whether the content is misinformation
        text_length: Simulated text length (word count proxy)
    """

    topics: set[str] = field(default_factory=set)
    topic_weights: dict[str, float] = field(default_factory=dict)
    sentiment: Sentiment = Sentiment.NEUTRAL
    emotions: dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.5
    controversy_score: float = 0.0
    ideology_score: float = 0.0
    is_misinformation: bool = False
    text_length: int = 100

    def __post_init__(self) -> None:
        """Validate content attributes."""
        self.quality_score = np.clip(self.quality_score, 0.0, 1.0)
        self.controversy_score = np.clip(self.controversy_score, 0.0, 1.0)
        self.ideology_score = np.clip(self.ideology_score, -1.0, 1.0)
        self.text_length = max(1, self.text_length)

    @property
    def emotional_intensity(self) -> float:
        """Calculate overall emotional intensity."""
        if not self.emotions:
            return 0.0
        return sum(self.emotions.values()) / len(self.emotions)

    @property
    def primary_emotion(self) -> str | None:
        """Get the dominant emotion."""
        if not self.emotions:
            return None
        return max(self.emotions, key=self.emotions.get)

    def add_topic(self, topic_id: str, weight: float = 0.5) -> None:
        """Add a topic with given weight."""
        self.topics.add(topic_id)
        self.topic_weights[topic_id] = np.clip(weight, 0.0, 1.0)

    def get_topic_weight(self, topic_id: str) -> float:
        """Get topic relevance weight."""
        return self.topic_weights.get(topic_id, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert content to dictionary."""
        return {
            "topics": list(self.topics),
            "topic_weights": self.topic_weights,
            "sentiment": str(self.sentiment),
            "emotions": self.emotions,
            "quality_score": self.quality_score,
            "controversy_score": self.controversy_score,
            "ideology_score": self.ideology_score,
            "is_misinformation": self.is_misinformation,
            "text_length": self.text_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostContent":
        """Create content from dictionary."""
        return cls(
            topics=set(data.get("topics", [])),
            topic_weights=data.get("topic_weights", {}),
            sentiment=Sentiment(data.get("sentiment", "neutral")),
            emotions=data.get("emotions", {}),
            quality_score=data.get("quality_score", 0.5),
            controversy_score=data.get("controversy_score", 0.0),
            ideology_score=data.get("ideology_score", 0.0),
            is_misinformation=data.get("is_misinformation", False),
            text_length=data.get("text_length", 100),
        )


@dataclass
class Post:
    """Represents a post in the social network.

    Attributes:
        post_id: Unique identifier
        author_id: User ID of the creator
        content: Post content characteristics
        created_step: Simulation step when created
        state: Current post state (active, suppressed, etc.)
        view_count: Number of views
        like_count: Number of likes
        share_count: Number of shares
        comment_count: Number of comments
        cascade_id: ID of associated viral cascade (if any)
        original_post_id: ID of original post if this is a share
        virality_score: Computed virality potential (0-1)
        moderation_score: Moderation confidence score (0-1)
        metadata: Additional arbitrary data
    """

    post_id: str
    author_id: str
    content: PostContent = field(default_factory=PostContent)
    created_step: int = 0
    state: PostState = PostState.ACTIVE
    view_count: int = 0
    like_count: int = 0
    share_count: int = 0
    comment_count: int = 0
    cascade_id: str | None = None
    original_post_id: str | None = None
    virality_score: float = 0.0
    moderation_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_engagement(self) -> int:
        """Total engagement count."""
        return self.like_count + self.share_count + self.comment_count

    @property
    def engagement_rate(self) -> float:
        """Engagement rate as ratio of views."""
        if self.view_count == 0:
            return 0.0
        return self.total_engagement / self.view_count

    @property
    def share_rate(self) -> float:
        """Share rate as ratio of views."""
        if self.view_count == 0:
            return 0.0
        return self.share_count / self.view_count

    @property
    def is_share(self) -> bool:
        """Check if this post is a share of another post."""
        return self.original_post_id is not None

    @property
    def is_misinformation(self) -> bool:
        """Check if post contains misinformation."""
        return self.content.is_misinformation

    @property
    def topics(self) -> list[str]:
        """Return list of topics associated with the post."""
        return list(self.content.topics)

    @property
    def quality_score(self) -> float:
        """Pass-through to content quality score."""
        return self.content.quality_score

    @property
    def controversy_score(self) -> float:
        """Pass-through to content controversy score."""
        return self.content.controversy_score

    @property
    def ideology_score(self) -> float:
        """Pass-through to content ideology score."""
        return self.content.ideology_score

    @property
    def emotional_intensity(self) -> float:
        """Pass-through to content emotional intensity."""
        return self.content.emotional_intensity

    @property
    def sentiment(self) -> float:
        """Numeric sentiment for downstream models."""
        return self._sentiment_to_value(self.content.sentiment)

    @staticmethod
    def _sentiment_to_value(sentiment: Sentiment) -> float:
        """Map Sentiment enum to [-1, 1] scale."""
        mapping = {
            Sentiment.POSITIVE: 1.0,
            Sentiment.NEGATIVE: -1.0,
            Sentiment.NEUTRAL: 0.0,
            Sentiment.MIXED: 0.0,
        }
        return mapping.get(sentiment, 0.0)

    def record_view(self) -> None:
        """Record a view."""
        self.view_count += 1

    def record_like(self) -> None:
        """Record a like."""
        self.like_count += 1

    def record_share(self) -> None:
        """Record a share."""
        self.share_count += 1

    def record_comment(self) -> None:
        """Record a comment."""
        self.comment_count += 1

    def suppress(self) -> None:
        """Suppress the post (reduce visibility)."""
        self.state = PostState.SUPPRESSED

    def remove(self) -> None:
        """Remove the post."""
        self.state = PostState.REMOVED

    def is_active(self) -> bool:
        """Check if post is active."""
        return self.state == PostState.ACTIVE

    def is_visible(self) -> bool:
        """Check if post can be shown (active or suppressed)."""
        return self.state in (PostState.ACTIVE, PostState.SUPPRESSED)

    def get_velocity(self, current_step: int, window: int = 5) -> float:
        """Calculate engagement velocity (engagement per step in recent window).

        This is a simplified calculation - actual velocity would need
        interaction timestamps.
        """
        age = max(1, current_step - self.created_step)
        if age <= window:
            return self.total_engagement / age
        # Decay for older posts
        return self.total_engagement / age * 0.5

    def calculate_virality_potential(self) -> float:
        """Calculate virality potential based on content attributes."""
        # Factors that increase virality
        emotional_factor = self.content.emotional_intensity * 0.3
        controversy_factor = self.content.controversy_score * 0.25
        quality_factor = self.content.quality_score * 0.2

        # Early engagement signals
        early_engagement = min(1.0, self.total_engagement / 100) * 0.25

        self.virality_score = np.clip(
            emotional_factor + controversy_factor + quality_factor + early_engagement,
            0.0,
            1.0,
        )
        return self.virality_score

    def to_dict(self) -> dict[str, Any]:
        """Convert post to dictionary for serialization."""
        return {
            "post_id": self.post_id,
            "author_id": self.author_id,
            "content": self.content.to_dict(),
            "created_step": self.created_step,
            "state": str(self.state),
            "view_count": self.view_count,
            "like_count": self.like_count,
            "share_count": self.share_count,
            "comment_count": self.comment_count,
            "cascade_id": self.cascade_id,
            "original_post_id": self.original_post_id,
            "virality_score": self.virality_score,
            "moderation_score": self.moderation_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Post":
        """Create post from dictionary."""
        return cls(
            post_id=data["post_id"],
            author_id=data["author_id"],
            content=PostContent.from_dict(data.get("content", {})),
            created_step=data.get("created_step", 0),
            state=PostState(data.get("state", "active")),
            view_count=data.get("view_count", 0),
            like_count=data.get("like_count", 0),
            share_count=data.get("share_count", 0),
            comment_count=data.get("comment_count", 0),
            cascade_id=data.get("cascade_id"),
            original_post_id=data.get("original_post_id"),
            virality_score=data.get("virality_score", 0.0),
            moderation_score=data.get("moderation_score", 0.0),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        return hash(self.post_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Post):
            return NotImplemented
        return self.post_id == other.post_id
