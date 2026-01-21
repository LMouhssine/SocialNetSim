"""User churn prediction model."""

from typing import Any

import numpy as np
import pandas as pd

from models import User
from engine.state import SimulationState
from ai.features import FeatureExtractor
from .base_trainer import BaseTrainer


class ChurnPredictor(BaseTrainer):
    """Predicts whether a user will churn (become inactive).

    Uses user traits, activity history, and network features to predict
    if a user will stop being active on the platform.
    """

    def __init__(
        self,
        churn_threshold_steps: int = 10,
        model_type: str = "xgboost",
        **model_params,
    ):
        """Initialize churn predictor.

        Args:
            churn_threshold_steps: Steps of inactivity to be considered churned
            model_type: Type of model to use
            **model_params: Model parameters
        """
        super().__init__(
            model_type=model_type,
            task_type="classification",
            **model_params,
        )
        self.churn_threshold_steps = churn_threshold_steps
        self.feature_extractor = FeatureExtractor()

    def get_feature_columns(self) -> list[str]:
        """Get feature columns for churn prediction."""
        return [
            # Trait features
            "ideology",
            "ideology_abs",
            "confirmation_bias",
            "misinfo_susceptibility",
            "emotional_reactivity",
            "activity_level",
            # Network features
            "follower_count",
            "following_count",
            "follower_ratio",
            "influence_score",
            "credibility_score",
            # Activity features
            "total_posts",
            "total_interactions",
            "num_interests",
            "avg_interest_weight",
            # State features
            "fatigue",
            "steps_since_active",
            "session_interactions",
            "seen_posts_count",
            "historical_engagement_rate",
        ]

    def create_training_data(
        self,
        users: dict[str, User],
        state: SimulationState,
    ) -> pd.DataFrame:
        """Create training dataset from simulation data.

        Args:
            users: Dictionary of users
            state: Simulation state

        Returns:
            DataFrame with features and target
        """
        records = []

        for user in users.values():
            features = self.feature_extractor.extract_user_features(user, state)

            # Churn target
            user_state = state.get_user_state(user.user_id)
            if user_state:
                steps_inactive = state.current_step - user_state.last_active_step
                features["target"] = float(steps_inactive >= self.churn_threshold_steps)
            else:
                features["target"] = 0.0

            features["user_id"] = user.user_id
            records.append(features)

        return pd.DataFrame(records)

    def train_from_simulation(
        self,
        users: dict[str, User],
        state: SimulationState,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Train model from simulation data.

        Args:
            users: Dictionary of users
            state: Simulation state
            test_size: Test set fraction

        Returns:
            Training metrics
        """
        df = self.create_training_data(users, state)

        if len(df) < 20:
            raise ValueError("Not enough users for training")

        return self.train(df, target_column="target", test_size=test_size)

    def predict_churn(
        self,
        user: User,
        state: SimulationState | None = None,
    ) -> tuple[bool, float]:
        """Predict if a user will churn.

        Args:
            user: User to evaluate
            state: Simulation state

        Returns:
            Tuple of (will_churn_prediction, probability)
        """
        features = self.feature_extractor.extract_user_features(user, state)
        df = pd.DataFrame([features])

        prediction = self.predict(df)[0]
        proba = self.predict_proba(df)[0, 1]

        return bool(prediction), float(proba)

    def predict_batch(
        self,
        users: dict[str, User],
        state: SimulationState,
    ) -> pd.DataFrame:
        """Predict churn for multiple users.

        Args:
            users: Dictionary of users
            state: Simulation state

        Returns:
            DataFrame with predictions
        """
        records = []
        for user in users.values():
            features = self.feature_extractor.extract_user_features(user, state)
            features["user_id"] = user.user_id
            records.append(features)

        df = pd.DataFrame(records)

        predictions = self.predict(df)
        probabilities = self.predict_proba(df)[:, 1]

        df["churn_prediction"] = predictions
        df["churn_probability"] = probabilities

        return df[["user_id", "churn_prediction", "churn_probability"]]

    def get_at_risk_users(
        self,
        users: dict[str, User],
        state: SimulationState,
        risk_threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """Get users at high risk of churning.

        Args:
            users: Dictionary of users
            state: Simulation state
            risk_threshold: Probability threshold for high risk

        Returns:
            List of (user_id, churn_probability) tuples
        """
        predictions_df = self.predict_batch(users, state)

        at_risk = predictions_df[
            predictions_df["churn_probability"] >= risk_threshold
        ]

        return list(zip(
            at_risk["user_id"],
            at_risk["churn_probability"],
        ))

    def get_churn_factors(self) -> dict[str, float]:
        """Get features most associated with churn.

        Returns:
            Dictionary of feature importance for top features
        """
        importance = self.get_feature_importance()

        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return dict(sorted_importance[:10])

    def analyze_user(
        self,
        user: User,
        state: SimulationState | None = None,
    ) -> dict[str, Any]:
        """Get detailed analysis of user churn risk.

        Args:
            user: User to analyze
            state: Simulation state

        Returns:
            Analysis dictionary
        """
        features = self.feature_extractor.extract_user_features(user, state)

        will_churn, probability = self.predict_churn(user, state)

        # Get feature importance
        importance = self.get_feature_importance()

        # Calculate risk factors
        risk_factors = {}
        for feature, value in features.items():
            if feature in importance and isinstance(value, (int, float)):
                # Higher importance * value indicates higher risk
                risk_factors[feature] = value * importance.get(feature, 0)

        # Sort by absolute contribution
        sorted_factors = sorted(
            risk_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Determine risk level
        if probability >= 0.8:
            risk_level = "critical"
        elif probability >= 0.6:
            risk_level = "high"
        elif probability >= 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "user_id": user.user_id,
            "churn_prediction": will_churn,
            "churn_probability": probability,
            "risk_level": risk_level,
            "churn_threshold_steps": self.churn_threshold_steps,
            "top_risk_factors": dict(sorted_factors[:5]),
            "user_traits": user.traits.to_dict(),
            "feature_values": {
                k: v for k, v in features.items()
                if k in self.feature_names
            },
        }
