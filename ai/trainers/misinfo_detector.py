"""Misinformation detection model."""

from typing import Any

import numpy as np
import pandas as pd

from models import Post, User
from engine.state import SimulationState
from ai.features import FeatureExtractor
from .base_trainer import BaseTrainer


class MisinfoDetector(BaseTrainer):
    """Detects potential misinformation in posts.

    Uses content features and author characteristics to identify
    posts that may contain misinformation.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        **model_params,
    ):
        """Initialize misinformation detector.

        Args:
            model_type: Type of model to use
            **model_params: Model parameters
        """
        super().__init__(
            model_type=model_type,
            task_type="classification",
            **model_params,
        )
        self.feature_extractor = FeatureExtractor()

    def get_feature_columns(self) -> list[str]:
        """Get feature columns for misinformation detection."""
        return [
            # Content features (most important)
            "text_length",
            "quality_score",
            "controversy_score",
            "emotional_intensity",
            "ideology_score_abs",
            "num_topics",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_mixed",
            # Author features
            "author_credibility",
            "author_influence",
            "author_followers",
            "author_total_posts",
            "author_ideology_abs",
            # Engagement patterns (misinfo often has unusual patterns)
            "share_count",
            "like_count",
            "comment_count",
            "velocity",
            "engagement_rate",
        ]

    def create_training_data(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
    ) -> pd.DataFrame:
        """Create training dataset from simulation data.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state

        Returns:
            DataFrame with features and target
        """
        records = []

        for post in posts:
            author = users.get(post.author_id)
            features = self.feature_extractor.extract_post_features(post, author, state)

            # Target: is_misinformation (ground truth from simulation)
            features["target"] = float(post.content.is_misinformation)
            features["post_id"] = post.post_id

            records.append(features)

        return pd.DataFrame(records)

    def train_from_simulation(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Train model from simulation data.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state
            test_size: Test set fraction

        Returns:
            Training metrics
        """
        df = self.create_training_data(posts, users, state)

        if len(df) < 20:
            raise ValueError("Not enough data for training")

        # Check for class balance
        misinfo_rate = df["target"].mean()
        if misinfo_rate < 0.01 or misinfo_rate > 0.99:
            raise ValueError(
                f"Highly imbalanced data: {misinfo_rate:.1%} misinformation rate"
            )

        return self.train(df, target_column="target", test_size=test_size)

    def detect_misinfo(
        self,
        post: Post,
        author: User | None = None,
        state: SimulationState | None = None,
    ) -> tuple[bool, float]:
        """Detect if a post contains misinformation.

        Args:
            post: Post to evaluate
            author: Post author
            state: Simulation state

        Returns:
            Tuple of (is_misinfo_prediction, confidence)
        """
        features = self.feature_extractor.extract_post_features(post, author, state)
        df = pd.DataFrame([features])

        prediction = self.predict(df)[0]
        proba = self.predict_proba(df)[0, 1]

        return bool(prediction), float(proba)

    def detect_batch(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
    ) -> pd.DataFrame:
        """Detect misinformation in multiple posts.

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
            features["actual_is_misinfo"] = post.content.is_misinformation
            records.append(features)

        df = pd.DataFrame(records)

        predictions = self.predict(df)
        probabilities = self.predict_proba(df)[:, 1]

        df["misinfo_prediction"] = predictions
        df["misinfo_probability"] = probabilities

        return df[["post_id", "actual_is_misinfo", "misinfo_prediction", "misinfo_probability"]]

    def get_misinfo_indicators(self) -> dict[str, float]:
        """Get features most indicative of misinformation.

        Returns:
            Dictionary of feature importance
        """
        importance = self.get_feature_importance()

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
        """Get detailed misinformation analysis of a post.

        Args:
            post: Post to analyze
            author: Post author
            state: Simulation state

        Returns:
            Analysis dictionary
        """
        features = self.feature_extractor.extract_post_features(post, author, state)

        is_misinfo, confidence = self.detect_misinfo(post, author, state)

        # Get feature importance
        importance = self.get_feature_importance()

        # Calculate indicator contributions
        indicators = {}
        for feature, value in features.items():
            if feature in importance and isinstance(value, (int, float)):
                indicators[feature] = value * importance.get(feature, 0)

        sorted_indicators = sorted(
            indicators.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Risk assessment
        if confidence >= 0.8:
            risk_level = "high"
            recommendation = "flag_for_review"
        elif confidence >= 0.6:
            risk_level = "moderate"
            recommendation = "add_context_label"
        elif confidence >= 0.4:
            risk_level = "low"
            recommendation = "monitor"
        else:
            risk_level = "minimal"
            recommendation = "none"

        return {
            "post_id": post.post_id,
            "misinfo_prediction": is_misinfo,
            "misinfo_confidence": confidence,
            "actual_is_misinfo": post.content.is_misinformation,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "top_indicators": dict(sorted_indicators[:5]),
            "content_quality": post.content.quality_score,
            "controversy_score": post.content.controversy_score,
            "author_credibility": author.credibility_score if author else None,
        }

    def evaluate_detection_performance(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
    ) -> dict[str, Any]:
        """Evaluate detection performance on a set of posts.

        Args:
            posts: List of posts to evaluate
            users: Dictionary of users
            state: Simulation state

        Returns:
            Performance metrics
        """
        df = self.detect_batch(posts, users, state)

        actual = df["actual_is_misinfo"].values
        predicted = df["misinfo_prediction"].values
        proba = df["misinfo_probability"].values

        # Calculate confusion matrix values
        tp = ((predicted == 1) & (actual == 1)).sum()
        tn = ((predicted == 0) & (actual == 0)).sum()
        fp = ((predicted == 1) & (actual == 0)).sum()
        fn = ((predicted == 0) & (actual == 1)).sum()

        # Metrics
        accuracy = (tp + tn) / max(1, len(df))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        # False positive/negative rates
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)

        return {
            "total_posts": len(df),
            "actual_misinfo_count": int(actual.sum()),
            "predicted_misinfo_count": int(predicted.sum()),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        }
