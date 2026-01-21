"""Feature engineering for AI models."""

from typing import Any

import numpy as np
import pandas as pd

from models import User, Post, Interaction
from models.enums import InteractionType
from engine.state import SimulationState


class FeatureExtractor:
    """Extracts features from simulation data for ML models."""

    def __init__(self):
        """Initialize feature extractor."""
        pass

    def extract_post_features(
        self,
        post: Post,
        author: User | None = None,
        state: SimulationState | None = None,
    ) -> dict[str, float]:
        """Extract features from a post for virality/engagement prediction.

        Features:
        - Content features: quality, controversy, sentiment, emotions
        - Author features: influence, credibility, followers
        - Early signals: early engagement counts, velocity

        Args:
            post: Post to extract features from
            author: Post author (optional)
            state: Simulation state for additional context

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Content features
        features["text_length"] = post.content.text_length
        features["quality_score"] = post.content.quality_score
        features["controversy_score"] = post.content.controversy_score
        features["emotional_intensity"] = post.content.emotional_intensity
        features["ideology_score_abs"] = abs(post.content.ideology_score)
        features["num_topics"] = len(post.content.topics)
        features["is_misinformation"] = float(post.content.is_misinformation)

        # Sentiment encoding (one-hot style)
        sentiment = str(post.content.sentiment)
        features["sentiment_positive"] = float(sentiment == "positive")
        features["sentiment_negative"] = float(sentiment == "negative")
        features["sentiment_neutral"] = float(sentiment == "neutral")
        features["sentiment_mixed"] = float(sentiment == "mixed")

        # Author features
        if author:
            features["author_influence"] = author.influence_score
            features["author_credibility"] = author.credibility_score
            features["author_followers"] = len(author.followers)
            features["author_following"] = len(author.following)
            features["author_total_posts"] = author.total_posts
            features["author_activity_level"] = author.traits.activity_level
            features["author_ideology_abs"] = abs(author.traits.ideology)
        else:
            # Default values
            features["author_influence"] = 0.0
            features["author_credibility"] = 0.5
            features["author_followers"] = 0
            features["author_following"] = 0
            features["author_total_posts"] = 0
            features["author_activity_level"] = 0.3
            features["author_ideology_abs"] = 0.0

        # Early engagement signals (if state available)
        if state:
            current_step = state.current_step
            age = max(1, current_step - post.created_step)

            features["post_age"] = age
            features["view_count"] = post.view_count
            features["like_count"] = post.like_count
            features["share_count"] = post.share_count
            features["comment_count"] = post.comment_count
            features["total_engagement"] = post.total_engagement
            features["engagement_rate"] = post.engagement_rate
            features["velocity"] = post.get_velocity(current_step)

            # Normalized by age
            features["likes_per_step"] = post.like_count / age
            features["shares_per_step"] = post.share_count / age
        else:
            features["post_age"] = 0
            features["view_count"] = 0
            features["like_count"] = 0
            features["share_count"] = 0
            features["comment_count"] = 0
            features["total_engagement"] = 0
            features["engagement_rate"] = 0.0
            features["velocity"] = 0.0
            features["likes_per_step"] = 0.0
            features["shares_per_step"] = 0.0

        return features

    def extract_user_features(
        self,
        user: User,
        state: SimulationState | None = None,
    ) -> dict[str, float]:
        """Extract features from a user for churn/behavior prediction.

        Features:
        - Demographic/trait features
        - Activity features
        - Network features
        - Historical behavior

        Args:
            user: User to extract features from
            state: Simulation state for additional context

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Trait features
        features["ideology"] = user.traits.ideology
        features["ideology_abs"] = abs(user.traits.ideology)
        features["confirmation_bias"] = user.traits.confirmation_bias
        features["misinfo_susceptibility"] = user.traits.misinfo_susceptibility
        features["emotional_reactivity"] = user.traits.emotional_reactivity
        features["activity_level"] = user.traits.activity_level

        # Network features
        features["follower_count"] = len(user.followers)
        features["following_count"] = len(user.following)
        features["follower_ratio"] = (
            len(user.followers) / max(1, len(user.following))
        )
        features["influence_score"] = user.influence_score
        features["credibility_score"] = user.credibility_score

        # Activity features
        features["total_posts"] = user.total_posts
        features["total_interactions"] = user.total_interactions

        # Interest features
        features["num_interests"] = len(user.interests)
        features["avg_interest_weight"] = (
            np.mean(list(user.interest_weights.values()))
            if user.interest_weights else 0.0
        )

        # State-dependent features
        if state:
            user_state = state.get_user_state(user.user_id)
            if user_state:
                features["fatigue"] = user_state.fatigue
                features["steps_since_active"] = (
                    state.current_step - user_state.last_active_step
                )
                features["session_interactions"] = user_state.session_interactions
                features["seen_posts_count"] = len(user_state.seen_posts)
            else:
                features["fatigue"] = 0.0
                features["steps_since_active"] = 0
                features["session_interactions"] = 0
                features["seen_posts_count"] = 0

            # Historical engagement rate
            user_interactions = state.get_user_interactions(user.user_id)
            if user_interactions:
                engagements = sum(
                    1 for i in user_interactions
                    if i.interaction_type != InteractionType.VIEW
                )
                views = sum(
                    1 for i in user_interactions
                    if i.interaction_type == InteractionType.VIEW
                )
                features["historical_engagement_rate"] = (
                    engagements / max(1, views)
                )
            else:
                features["historical_engagement_rate"] = 0.0
        else:
            features["fatigue"] = 0.0
            features["steps_since_active"] = 0
            features["session_interactions"] = 0
            features["seen_posts_count"] = 0
            features["historical_engagement_rate"] = 0.0

        return features

    def extract_interaction_features(
        self,
        user: User,
        post: Post,
        author: User | None = None,
    ) -> dict[str, float]:
        """Extract features for user-post interaction prediction.

        Args:
            user: User considering interaction
            post: Post to potentially interact with
            author: Post author

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # User features (subset)
        features["user_activity_level"] = user.traits.activity_level
        features["user_emotional_reactivity"] = user.traits.emotional_reactivity
        features["user_confirmation_bias"] = user.traits.confirmation_bias

        # Post features (subset)
        features["post_quality"] = post.content.quality_score
        features["post_controversy"] = post.content.controversy_score
        features["post_emotional_intensity"] = post.content.emotional_intensity

        # Matching features
        # Topic overlap
        common_topics = user.interests & post.content.topics
        features["topic_overlap"] = len(common_topics)
        features["topic_overlap_ratio"] = (
            len(common_topics) / max(1, len(post.content.topics))
        )

        # Interest-weighted match
        if common_topics:
            match_score = sum(
                user.get_interest_weight(t) * post.content.get_topic_weight(t)
                for t in common_topics
            )
            features["interest_match_score"] = match_score / len(common_topics)
        else:
            features["interest_match_score"] = 0.0

        # Ideology alignment
        ideology_diff = abs(user.traits.ideology - post.content.ideology_score)
        features["ideology_diff"] = ideology_diff
        features["ideology_alignment"] = 1 - (ideology_diff / 2)

        # Author relationship
        if author:
            features["is_following_author"] = float(author.user_id in user.following)
            features["author_influence"] = author.influence_score
        else:
            features["is_following_author"] = 0.0
            features["author_influence"] = 0.0

        return features

    def create_post_dataset(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
        target_column: str = "total_engagement",
    ) -> pd.DataFrame:
        """Create a dataset of post features for training.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state
            target_column: Name of target column

        Returns:
            DataFrame with features and target
        """
        records = []

        for post in posts:
            author = users.get(post.author_id)
            features = self.extract_post_features(post, author, state)

            # Add target
            if target_column == "total_engagement":
                features["target"] = post.total_engagement
            elif target_column == "is_viral":
                features["target"] = float(post.total_engagement > 50)
            elif target_column == "is_misinformation":
                features["target"] = float(post.content.is_misinformation)
            elif target_column == "share_count":
                features["target"] = post.share_count

            features["post_id"] = post.post_id
            records.append(features)

        return pd.DataFrame(records)

    def create_user_dataset(
        self,
        users: dict[str, User],
        state: SimulationState,
        churn_threshold_steps: int = 10,
    ) -> pd.DataFrame:
        """Create a dataset of user features for churn prediction.

        Args:
            users: Dictionary of users
            state: Simulation state
            churn_threshold_steps: Steps of inactivity to consider as churn

        Returns:
            DataFrame with features and churn target
        """
        records = []

        for user in users.values():
            features = self.extract_user_features(user, state)

            # Churn target (inactive for threshold steps)
            user_state = state.get_user_state(user.user_id)
            if user_state:
                steps_inactive = state.current_step - user_state.last_active_step
                features["target"] = float(steps_inactive >= churn_threshold_steps)
            else:
                features["target"] = 0.0

            features["user_id"] = user.user_id
            records.append(features)

        return pd.DataFrame(records)

    @staticmethod
    def get_feature_names(feature_type: str) -> list[str]:
        """Get list of feature names for a feature type.

        Args:
            feature_type: "post", "user", or "interaction"

        Returns:
            List of feature names
        """
        if feature_type == "post":
            return [
                "text_length", "quality_score", "controversy_score",
                "emotional_intensity", "ideology_score_abs", "num_topics",
                "is_misinformation", "sentiment_positive", "sentiment_negative",
                "sentiment_neutral", "sentiment_mixed", "author_influence",
                "author_credibility", "author_followers", "author_following",
                "author_total_posts", "author_activity_level", "author_ideology_abs",
                "post_age", "view_count", "like_count", "share_count",
                "comment_count", "total_engagement", "engagement_rate",
                "velocity", "likes_per_step", "shares_per_step",
            ]
        elif feature_type == "user":
            return [
                "ideology", "ideology_abs", "confirmation_bias",
                "misinfo_susceptibility", "emotional_reactivity", "activity_level",
                "follower_count", "following_count", "follower_ratio",
                "influence_score", "credibility_score", "total_posts",
                "total_interactions", "num_interests", "avg_interest_weight",
                "fatigue", "steps_since_active", "session_interactions",
                "seen_posts_count", "historical_engagement_rate",
            ]
        elif feature_type == "interaction":
            return [
                "user_activity_level", "user_emotional_reactivity",
                "user_confirmation_bias", "post_quality", "post_controversy",
                "post_emotional_intensity", "topic_overlap", "topic_overlap_ratio",
                "interest_match_score", "ideology_diff", "ideology_alignment",
                "is_following_author", "author_influence",
            ]
        else:
            return []
