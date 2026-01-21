"""Tests for AI trainers."""

import pytest
import numpy as np
import pandas as pd

from ai.trainers.virality_predictor import ViralityPredictor
from ai.trainers.churn_predictor import ChurnPredictor
from ai.trainers.misinfo_detector import MisinfoDetector
from ai.evaluation import ModelEvaluator


class TestViralityPredictor:
    """Tests for ViralityPredictor class."""

    @pytest.fixture
    def virality_predictor(self):
        """Create virality predictor."""
        return ViralityPredictor()

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "quality_score": np.random.uniform(0, 1, n_samples),
            "sentiment": np.random.uniform(-1, 1, n_samples),
            "controversy_score": np.random.uniform(0, 1, n_samples),
            "emotional_intensity": np.random.uniform(0, 1, n_samples),
            "author_followers": np.random.randint(10, 10000, n_samples),
            "author_influence": np.random.uniform(0, 1, n_samples),
            "early_likes": np.random.randint(0, 100, n_samples),
            "early_shares": np.random.randint(0, 50, n_samples),
        })

        # Target: viral if quality + emotional_intensity > 1.2
        y = ((X["quality_score"] + X["emotional_intensity"]) > 1.2).astype(int)

        return X, y

    def test_train(self, virality_predictor, training_data):
        """Test training the model."""
        X, y = training_data
        virality_predictor.train(X, y)

        assert virality_predictor.model is not None
        assert virality_predictor.is_trained

    def test_predict(self, virality_predictor, training_data):
        """Test making predictions."""
        X, y = training_data
        virality_predictor.train(X, y)

        predictions = virality_predictor.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, virality_predictor, training_data):
        """Test probability predictions."""
        X, y = training_data
        virality_predictor.train(X, y)

        probas = virality_predictor.predict_proba(X)

        assert len(probas) == len(X)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_feature_importance(self, virality_predictor, training_data):
        """Test getting feature importance."""
        X, y = training_data
        virality_predictor.train(X, y)

        importance = virality_predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(v >= 0 for v in importance.values())

    def test_untrained_raises(self, virality_predictor, training_data):
        """Test predicting without training raises error."""
        X, _ = training_data

        with pytest.raises(ValueError):
            virality_predictor.predict(X)


class TestChurnPredictor:
    """Tests for ChurnPredictor class."""

    @pytest.fixture
    def churn_predictor(self):
        """Create churn predictor."""
        return ChurnPredictor()

    @pytest.fixture
    def churn_data(self):
        """Create sample churn data."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "days_since_last_active": np.random.randint(0, 30, n_samples),
            "total_posts": np.random.randint(0, 100, n_samples),
            "total_interactions": np.random.randint(0, 500, n_samples),
            "follower_count": np.random.randint(0, 1000, n_samples),
            "avg_engagement_rate": np.random.uniform(0, 0.5, n_samples),
            "activity_trend": np.random.uniform(-1, 1, n_samples),
        })

        # Churn if inactive and low engagement
        y = ((X["days_since_last_active"] > 14) & (X["avg_engagement_rate"] < 0.1)).astype(int)

        return X, y

    def test_train_and_predict(self, churn_predictor, churn_data):
        """Test training and predicting."""
        X, y = churn_data
        churn_predictor.train(X, y)

        predictions = churn_predictor.predict(X)

        assert len(predictions) == len(X)


class TestMisinfoDetector:
    """Tests for MisinfoDetector class."""

    @pytest.fixture
    def misinfo_detector(self):
        """Create misinfo detector."""
        return MisinfoDetector()

    @pytest.fixture
    def misinfo_data(self):
        """Create sample misinfo data."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "quality_score": np.random.uniform(0, 1, n_samples),
            "source_credibility": np.random.uniform(0, 1, n_samples),
            "controversy_score": np.random.uniform(0, 1, n_samples),
            "emotional_intensity": np.random.uniform(0, 1, n_samples),
            "claim_count": np.random.randint(0, 10, n_samples),
            "share_velocity": np.random.uniform(0, 100, n_samples),
        })

        # Misinfo if low quality, low credibility, high controversy
        y = (
            (X["quality_score"] < 0.3) &
            (X["source_credibility"] < 0.3) &
            (X["controversy_score"] > 0.7)
        ).astype(int)

        return X, y

    def test_train_and_predict(self, misinfo_detector, misinfo_data):
        """Test training and predicting."""
        X, y = misinfo_data
        misinfo_detector.train(X, y)

        predictions = misinfo_detector.predict(X)

        assert len(predictions) == len(X)


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create model evaluator."""
        return ModelEvaluator()

    def test_evaluate_binary(self, evaluator):
        """Test binary classification evaluation."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.7, 0.1])

        metrics = evaluator.evaluate_binary(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics

        # Check metrics are in valid range
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_cross_validate(self, evaluator):
        """Test cross-validation."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        scores = evaluator.cross_validate(model, X, y, cv=3)

        assert "mean_accuracy" in scores
        assert "std_accuracy" in scores
        assert len(scores.get("fold_scores", [])) == 3
