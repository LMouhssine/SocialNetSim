"""Base trainer class for AI models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(
        self,
        model_type: str = "xgboost",
        task_type: str = "classification",
        **model_params,
    ):
        """Initialize trainer.

        Args:
            model_type: Type of model ("xgboost", "random_forest", "logistic")
            task_type: "classification" or "regression"
            **model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.task_type = task_type
        self.model_params = model_params
        self.model = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.training_metrics: dict[str, float] = {}

    @abstractmethod
    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names.

        Returns:
            List of feature names used by this model
        """
        pass

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "xgboost":
            if self.task_type == "classification":
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 6),
                    learning_rate=self.model_params.get("learning_rate", 0.1),
                    random_state=self.model_params.get("random_state", 42),
                    use_label_encoder=False,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    base_score=0.5,
                    verbosity=0,
                )
            else:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 6),
                    learning_rate=self.model_params.get("learning_rate", 0.1),
                    random_state=self.model_params.get("random_state", 42),
                )
        elif self.model_type == "random_forest":
            if self.task_type == "classification":
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 10),
                    random_state=self.model_params.get("random_state", 42),
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 10),
                    random_state=self.model_params.get("random_state", 42),
                )
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=self.model_params.get("max_iter", 1000),
                random_state=self.model_params.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        df: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | None = None,
        target_column: str = "target",
        test_size: float = 0.2,
        validate: bool = True,
    ) -> dict[str, float]:
        """Train the model on provided data.

        Supports both:
        - train(df, target_column=...)
        - train(X, y)
        """
        # Backward compatibility: train(df, "target")
        if isinstance(y, str) and target_column == "target":
            target_column = y
            y = None

        # DataFrame path
        if isinstance(df, pd.DataFrame):
            if y is None:
                # Use predefined feature columns
                self.feature_names = self.get_feature_columns()
                available_features = [f for f in self.feature_names if f in df.columns]
                if not available_features:
                    raise ValueError("No matching feature columns found in DataFrame")
                X = df[available_features].values
                y = df[target_column].values
            else:
                # Explicit features + target
                self.feature_names = list(df.columns)
                X = df.values
                y = np.array(y)
        else:
            # Numpy array path
            if y is None:
                raise ValueError("Target values are required when training from arrays")
            X = np.array(df)
            y = np.array(y)
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # For classification, ensure binary labels are 0 or 1
        if self.task_type == "classification":
            unique_vals = np.unique(y)
            if len(unique_vals) == 1:
                # If only one class, add a few samples of the other class
                y = np.concatenate([y, np.ones_like(y[:2]) * (1 - y[0])])
                X = np.concatenate([X, X[:2]])
            y = y.astype(int)

        # Split data
        if validate and len(df) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Create and train model
        self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Compute metrics
        metrics = self._compute_metrics(X_train, y_train, X_test, y_test)
        self.training_metrics = metrics

        return metrics

    def _compute_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Compute training and validation metrics.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        if self.task_type == "classification":
            # Classification metrics
            metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)

            # Handle binary vs multiclass
            if len(np.unique(y_train)) == 2:
                metrics["train_precision"] = precision_score(y_train, y_train_pred, zero_division=0)
                metrics["test_precision"] = precision_score(y_test, y_test_pred, zero_division=0)
                metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
                metrics["test_recall"] = recall_score(y_test, y_test_pred, zero_division=0)
                metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)
                metrics["test_f1"] = f1_score(y_test, y_test_pred, zero_division=0)

                # AUC if we have predict_proba
                if hasattr(self.model, "predict_proba"):
                    try:
                        y_train_prob = self.model.predict_proba(X_train)[:, 1]
                        y_test_prob = self.model.predict_proba(X_test)[:, 1]
                        metrics["train_auc"] = roc_auc_score(y_train, y_train_prob)
                        metrics["test_auc"] = roc_auc_score(y_test, y_test_prob)
                    except Exception:
                        pass
        else:
            # Regression metrics
            metrics["train_mse"] = mean_squared_error(y_train, y_train_pred)
            metrics["test_mse"] = mean_squared_error(y_test, y_test_pred)
            metrics["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            metrics["test_mae"] = mean_absolute_error(y_test, y_test_pred)
            metrics["train_r2"] = r2_score(y_train, y_train_pred)
            metrics["test_r2"] = r2_score(y_test, y_test_pred)
            metrics["train_rmse"] = np.sqrt(metrics["train_mse"])
            metrics["test_rmse"] = np.sqrt(metrics["test_mse"])

        return metrics

    def predict(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            df: DataFrame with features

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if isinstance(df, pd.DataFrame):
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in df.columns]
                X = df[available_features].values
            else:
                X = df.values
        else:
            X = np.array(df)

        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    def predict_proba(
        self,
        df: pd.DataFrame | np.ndarray,
        positive_class_only: bool = True,
    ) -> np.ndarray:
        """Get probability predictions (classification only).

        Args:
            df: DataFrame with features

        Returns:
            Probability array
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"Model type {self.model_type} doesn't support predict_proba")

        if isinstance(df, pd.DataFrame):
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in df.columns]
                X = df[available_features].values
            else:
                X = df.values
        else:
            X = np.array(df)
        X = np.nan_to_num(X, nan=0.0)
        probs = self.model.predict_proba(X)
        if positive_class_only and probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature name to importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return {}

        return dict(zip(self.feature_names, importances))

    def save(self, path: str | Path) -> None:
        """Save model to file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "model_params": self.model_params,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load model from file.

        Args:
            path: Path to load from
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.model_type = data["model_type"]
        self.task_type = data["task_type"]
        self.model_params = data["model_params"]
        self.feature_names = data["feature_names"]
        self.is_trained = data["is_trained"]
        self.training_metrics = data["training_metrics"]
