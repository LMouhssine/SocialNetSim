"""Content/post generation."""

from typing import Any
import uuid

import numpy as np
from numpy.random import Generator

from config.schemas import ContentConfig
from models import Post, PostContent, User, Sentiment
from models.enums import EmotionType
from .distributions import sample_from_config
from .topic_generator import Topic


class ContentGenerator:
    """Generates synthetic post content matching user characteristics."""

    def __init__(
        self,
        config: ContentConfig,
        topics: dict[str, Topic],
        seed: int | None = None,
    ):
        """Initialize content generator.

        Args:
            config: Content generation configuration
            topics: Available topics
            seed: Random seed for reproducibility
        """
        self.config = config
        self.topics = topics
        self.rng = np.random.default_rng(seed)
        self.post_counter = 0

    def generate_post(
        self,
        author: User,
        step: int,
        forced_topics: set[str] | None = None,
        force_misinfo: bool | None = None,
    ) -> Post:
        """Generate a post for a user.

        Args:
            author: User creating the post
            step: Current simulation step
            forced_topics: Override topic selection
            force_misinfo: Override misinformation status

        Returns:
            Generated Post
        """
        post_id = self._generate_post_id()

        # Generate content
        content = self._generate_content(author, forced_topics, force_misinfo)

        # Calculate initial virality potential
        virality_score = self._calculate_virality_potential(content, author)

        post = Post(
            post_id=post_id,
            author_id=author.user_id,
            content=content,
            created_step=step,
            virality_score=virality_score,
        )

        return post

    def _generate_post_id(self) -> str:
        """Generate unique post ID."""
        self.post_counter += 1
        return f"post_{self.post_counter:08d}"

    def _generate_content(
        self,
        author: User,
        forced_topics: set[str] | None = None,
        force_misinfo: bool | None = None,
    ) -> PostContent:
        """Generate post content based on author characteristics.

        Args:
            author: Post author
            forced_topics: Override topic selection
            force_misinfo: Override misinformation status

        Returns:
            Generated PostContent
        """
        # Select topics
        if forced_topics:
            topics = forced_topics
            topic_weights = {t: 0.8 for t in topics}
        else:
            topics, topic_weights = self._select_topics(author)

        # Generate quality score
        quality = float(sample_from_config(self.rng, self.config.quality_distribution))

        # Author credibility affects quality
        quality = quality * (0.5 + 0.5 * author.credibility_score)
        quality = np.clip(quality, 0.0, 1.0)

        # Generate sentiment based on topics and author
        sentiment = self._generate_sentiment(author, topics)

        # Generate emotions
        emotions = self._generate_emotions(author, sentiment, topics)

        # Calculate controversy based on topics
        controversy = self._calculate_controversy(topics)

        # Generate ideology score based on author and topics
        ideology = self._generate_ideology_score(author, topics)

        # Determine if misinformation
        if force_misinfo is not None:
            is_misinfo = force_misinfo
        else:
            is_misinfo = self._determine_misinformation(author, controversy)

        # Generate text length (simulated)
        text_length = self._generate_text_length(author)

        return PostContent(
            topics=topics,
            topic_weights=topic_weights,
            sentiment=sentiment,
            emotions=emotions,
            quality_score=quality,
            controversy_score=controversy,
            ideology_score=ideology,
            is_misinformation=is_misinfo,
            text_length=text_length,
        )

    def _select_topics(
        self,
        author: User,
    ) -> tuple[set[str], dict[str, float]]:
        """Select topics for a post based on author interests.

        Args:
            author: Post author

        Returns:
            Tuple of (topic set, topic weights)
        """
        # Number of topics (1-3, usually 1-2)
        n_topics = int(self.rng.choice([1, 1, 1, 2, 2, 3], p=[0.3, 0.2, 0.15, 0.2, 0.1, 0.05]))

        if author.interests:
            # Weight by author's interest strength
            interest_list = list(author.interests)
            weights = np.array([
                author.get_interest_weight(t) for t in interest_list
            ])
            weights = weights / weights.sum()

            selected_indices = self.rng.choice(
                len(interest_list),
                size=min(n_topics, len(interest_list)),
                replace=False,
                p=weights,
            )
            selected_topics = {interest_list[i] for i in selected_indices}
        else:
            # Random selection weighted by popularity
            topic_list = list(self.topics.values())
            weights = np.array([t.popularity for t in topic_list])
            weights = weights / weights.sum()

            selected_indices = self.rng.choice(
                len(topic_list),
                size=min(n_topics, len(topic_list)),
                replace=False,
                p=weights,
            )
            selected_topics = {topic_list[i].topic_id for i in selected_indices}

        # Generate weights for selected topics
        topic_weights = {}
        for topic_id in selected_topics:
            base_weight = author.get_interest_weight(topic_id) if topic_id in author.interests else 0.5
            topic_weights[topic_id] = min(1.0, base_weight * self.rng.uniform(0.8, 1.2))

        return selected_topics, topic_weights

    def _generate_sentiment(
        self,
        author: User,
        topics: set[str],
    ) -> Sentiment:
        """Generate sentiment based on author and topics.

        Args:
            author: Post author
            topics: Post topics

        Returns:
            Sentiment enum value
        """
        # Emotional users tend to have more extreme sentiment
        emotional_factor = author.traits.emotional_reactivity

        # Check if topics are controversial
        controversy = self._calculate_controversy(topics)

        # Base probabilities
        if controversy > 0.5 or emotional_factor > 0.6:
            # More likely to be non-neutral
            probs = [0.25, 0.35, 0.20, 0.20]  # positive, negative, neutral, mixed
        else:
            probs = [0.30, 0.20, 0.35, 0.15]

        sentiment_values = [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.MIXED]
        # Use index-based selection to avoid numpy string conversion
        selected_idx = self.rng.choice(len(sentiment_values), p=probs)
        return sentiment_values[selected_idx]

    def _generate_emotions(
        self,
        author: User,
        sentiment: Sentiment,
        topics: set[str],
    ) -> dict[str, float]:
        """Generate emotional content mapping.

        Args:
            author: Post author
            sentiment: Post sentiment
            topics: Post topics

        Returns:
            Dictionary mapping emotion types to intensities
        """
        emotions = {}
        emotional_factor = author.traits.emotional_reactivity

        # Sentiment influences primary emotion
        if sentiment == Sentiment.POSITIVE:
            emotions[EmotionType.JOY.value] = float(self.rng.beta(3, 2) * emotional_factor * 0.8 + 0.2)
        elif sentiment == Sentiment.NEGATIVE:
            # Randomly select negative emotion (use index to avoid numpy str conversion)
            neg_emotions = [EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR]
            neg_emotion = neg_emotions[self.rng.integers(0, len(neg_emotions))]
            emotions[neg_emotion.value] = float(self.rng.beta(3, 2) * emotional_factor * 0.8 + 0.2)
        elif sentiment == Sentiment.MIXED:
            emotions[EmotionType.SURPRISE.value] = float(self.rng.beta(2, 3) * 0.6)
            secondary_emotions = [EmotionType.JOY, EmotionType.SADNESS]
            secondary = secondary_emotions[self.rng.integers(0, len(secondary_emotions))]
            emotions[secondary.value] = float(self.rng.beta(2, 3) * 0.4)

        # Controversial topics increase anger
        controversy = self._calculate_controversy(topics)
        if controversy > 0.5:
            anger_boost = controversy * emotional_factor * 0.5
            emotions[EmotionType.ANGER.value] = emotions.get(EmotionType.ANGER.value, 0) + anger_boost
            emotions[EmotionType.ANGER.value] = min(1.0, emotions[EmotionType.ANGER.value])

        # If no emotions set, add neutral
        if not emotions:
            emotions[EmotionType.NEUTRAL.value] = 0.5

        return emotions

    def _calculate_controversy(self, topics: set[str]) -> float:
        """Calculate controversy score based on topics.

        Args:
            topics: Set of topic IDs

        Returns:
            Controversy score (0-1)
        """
        if not topics:
            return 0.0

        controversy_scores = []
        for topic_id in topics:
            topic = self.topics.get(topic_id)
            if topic:
                controversy_scores.append(topic.controversy_score)

        return max(controversy_scores) if controversy_scores else 0.0

    def _generate_ideology_score(
        self,
        author: User,
        topics: set[str],
    ) -> float:
        """Generate ideology score for post.

        Args:
            author: Post author
            topics: Post topics

        Returns:
            Ideology score (-1 to 1)
        """
        # Start with author's ideology
        base_ideology = author.traits.ideology

        # Topics can influence ideology expression
        topic_ideology = 0.0
        if topics:
            for topic_id in topics:
                topic = self.topics.get(topic_id)
                if topic:
                    topic_ideology += topic.ideology_bias
            topic_ideology /= len(topics)

        # Blend author ideology with topic bias
        ideology = 0.7 * base_ideology + 0.3 * topic_ideology

        # Add some noise
        noise = self.rng.normal(0, 0.1)
        ideology += noise

        return float(np.clip(ideology, -1.0, 1.0))

    def _determine_misinformation(
        self,
        author: User,
        controversy: float,
    ) -> bool:
        """Determine if post contains misinformation.

        Args:
            author: Post author
            controversy: Controversy score

        Returns:
            True if post is misinformation
        """
        base_rate = self.config.misinformation_rate

        # Factors that increase misinfo likelihood
        # Low credibility authors more likely to spread misinfo
        credibility_factor = 1 + (1 - author.credibility_score) * 0.5

        # Controversial topics more likely to have misinfo
        controversy_factor = 1 + controversy * 0.5

        # Susceptible users more likely to create/spread misinfo
        susceptibility_factor = 1 + author.traits.misinfo_susceptibility * 0.3

        adjusted_rate = base_rate * credibility_factor * controversy_factor * susceptibility_factor
        adjusted_rate = min(0.5, adjusted_rate)  # Cap at 50%

        return self.rng.random() < adjusted_rate

    def _generate_text_length(self, author: User) -> int:
        """Generate simulated text length.

        Args:
            author: Post author

        Returns:
            Word count proxy
        """
        # More active users tend to write shorter posts
        base_length = 100

        if author.traits.activity_level > 0.7:
            # Very active = shorter posts
            length = int(self.rng.exponential(50) + 20)
        elif author.traits.activity_level > 0.4:
            # Moderate = medium posts
            length = int(self.rng.exponential(80) + 40)
        else:
            # Less active = longer, more considered posts
            length = int(self.rng.exponential(120) + 60)

        return max(10, min(500, length))

    def _calculate_virality_potential(
        self,
        content: PostContent,
        author: User,
    ) -> float:
        """Calculate initial virality potential.

        Args:
            content: Post content
            author: Post author

        Returns:
            Virality potential (0-1)
        """
        # Factors contributing to virality
        emotional_factor = content.emotional_intensity * 0.25
        controversy_factor = content.controversy_score * 0.2
        quality_factor = content.quality_score * 0.15
        author_influence = author.influence_score * 0.25

        # Misinformation often has higher engagement (unfortunately)
        misinfo_factor = 0.15 if content.is_misinformation else 0.0

        virality = (
            emotional_factor +
            controversy_factor +
            quality_factor +
            author_influence +
            misinfo_factor
        )

        return float(np.clip(virality, 0.0, 1.0))

    def generate_share_post(
        self,
        original_post: Post,
        sharer: User,
        step: int,
    ) -> Post:
        """Generate a share of an existing post.

        Args:
            original_post: Post being shared
            sharer: User sharing the post
            step: Current simulation step

        Returns:
            New share post
        """
        post_id = self._generate_post_id()

        # Share inherits content from original
        # but may have slightly modified attributes based on sharer
        content = PostContent(
            topics=original_post.content.topics.copy(),
            topic_weights=original_post.content.topic_weights.copy(),
            sentiment=original_post.content.sentiment,
            emotions=original_post.content.emotions.copy(),
            quality_score=original_post.content.quality_score,
            controversy_score=original_post.content.controversy_score,
            ideology_score=original_post.content.ideology_score,
            is_misinformation=original_post.content.is_misinformation,
            text_length=original_post.content.text_length,
        )

        return Post(
            post_id=post_id,
            author_id=sharer.user_id,
            content=content,
            created_step=step,
            original_post_id=original_post.post_id,
            cascade_id=original_post.cascade_id,
            virality_score=original_post.virality_score,
        )
