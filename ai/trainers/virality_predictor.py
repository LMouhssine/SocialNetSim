"""Virality prediction model."""

from typing import Any

import numpy as np
import pandas as pd

from models import Post, User
from engine.state import SimulationState
from ai.features import FeatureExtractor
from .base_trainer import BaseTrainer


class ViralityPredictor(BaseTrainer):
    """Predicts whether a post will go viral.

    Uses early engagement signals and content features to predict
    if a post will achieve high engagement.
    """

    def __init__(
        self,
        viral_threshold: int = 50,
        model_type: str = "xgboost",
        **model_params,
    ):
        """Initialize virality predictor.

        Args:
            viral_threshold: Engagement count to be considered viral
            model_type: Type of model to use
            **model_params: Model parameters
        """
        super().__init__(
            model_type=model_type,
            task_type="classification",
            **model_params,
        )
        self.viral_threshold = viral_threshold
        self.feature_extractor = FeatureExtractor()

    def get_feature_columns(self) -> list[str]:
        """Get feature columns for virality prediction."""
        return [
            # Content features
            "text_length",
            "quality_score",
            "controversy_score",
            "emotional_intensity",
            "ideology_score_abs",
            "num_topics",
            "is_misinformation",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_mixed",
            # Author features
            "author_influence",
            "author_credibility",
            "author_followers",
            "author_following",
            "author_total_posts",
            "author_activity_level",
            "author_ideology_abs",
            # Early signals (for already posted content)
            "view_count",
            "like_count",
            "share_count",
            "comment_count",
            "velocity",
            "likes_per_step",
            "shares_per_step",
        ]

    def create_training_data(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
        min_age: int = 5,
    ) -> pd.DataFrame:
        """Create training dataset from simulation data.

        Args:
            posts: List of posts to use
            users: Dictionary of users
            state: Simulation state
            min_age: Minimum post age to include (for mature engagement data)

        Returns:
            DataFrame with features and target
        """
        records = []

        for post in posts:
            # Only include posts old enough to have meaningful engagement
            age = state.current_step - post.created_step
            if age < min_age:
                continue

            author = users.get(post.author_id)
            features = self.feature_extractor.extract_post_features(post, author, state)

            # Binary target: is viral?
            features["target"] = float(post.total_engagement >= self.viral_threshold)
            features["post_id"] = post.post_id

            records.append(features)

        return pd.DataFrame(records)

    def train_from_simulation(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
        min_age: int = 5,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Train model from simulation data.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state
            min_age: Minimum post age
            test_size: Test set fraction

        Returns:
            Training metrics
        """
        df = self.create_training_data(posts, users, state, min_age)

        if len(df) < 20:
            raise ValueError("Not enough data for training (need at least 20 posts)")

        return self.train(df, target_column="target", test_size=test_size)

    def predict_virality(
        self,
        post: Post,
        author: User | None = None,
        state: SimulationState | None = None,
    ) -> tuple[bool, float]:
        """Predict if a post will go viral.

        Args:
            post: Post to evaluate
            author: Post author
            state: Simulation state

        Returns:
            Tuple of (is_viral_prediction, probability)
        """
        features = self.feature_extractor.extract_post_features(post, author, state)
        df = pd.DataFrame([features])

        prediction = self.predict(df)[0]
        probas = self.predict_proba(df)
        proba = probas[0] if probas.ndim == 1 else probas[0, 1]

        return bool(prediction), float(proba)

    def predict_batch(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
    ) -> pd.DataFrame:
        """Predict virality for multiple posts.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state

        Returns:
            DataFrame with predictions
        """
        records = []
        for post in posts:
            author = users.get(post.author_id)
            features = self.feature_extractor.extract_post_features(post, author, state)
            features["post_id"] = post.post_id
            records.append(features)

        df = pd.DataFrame(records)

        predictions = self.predict(df)
        probas = self.predict_proba(df)
        probabilities = probas if probas.ndim == 1 else probas[:, 1]

        df["viral_prediction"] = predictions
        df["viral_probability"] = probabilities

        return df[["post_id", "viral_prediction", "viral_probability"]]

    def get_viral_features(self) -> dict[str, float]:
        """Get features most associated with virality.

        Returns:
            Dictionary of feature importance for top features
        """
        importance = self.get_feature_importance()

        # Sort by importance
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return dict(sorted_importance[:10])

    def analyze_post(
        self,
        post: Post,
        author: User | None = None,
        state: SimulationState | None = None,
    ) -> dict[str, Any]:
        """Get detailed analysis of why a post might go viral.

        Args:
            post: Post to analyze
            author: Post author
            state: Simulation state

        Returns:
            Analysis dictionary
        """
        features = self.feature_extractor.extract_post_features(post, author, state)
        df = pd.DataFrame([features])

        is_viral, probability = self.predict_virality(post, author, state)

        # Get feature importance
        importance = self.get_feature_importance()

        # Calculate feature contributions
        contributions = {}
        for feature, value in features.items():
            if feature in importance and isinstance(value, (int, float)):
                contributions[feature] = value * importance.get(feature, 0)

        # Sort contributions
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "post_id": post.post_id,
            "viral_prediction": is_viral,
            "viral_probability": probability,
            "viral_threshold": self.viral_threshold,
            "top_contributing_features": dict(sorted_contributions[:5]),
            "feature_values": {
                k: v for k, v in features.items()
                if k in self.feature_names
            },
        }
