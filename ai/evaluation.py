"""Model evaluation and comparison utilities."""

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .trainers.base_trainer import BaseTrainer


class ModelEvaluator:
    """Evaluates and compares ML models."""

    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_history: list[dict[str, Any]] = []

    def evaluate_model(
        self,
        trainer: BaseTrainer,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Evaluate a trained model.

        Args:
            trainer: Trained model trainer
            X: Feature matrix
            y: True labels
            model_name: Name for this evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        if not trainer.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = trainer.model.predict(X)

        results = {
            "model_name": model_name,
            "model_type": trainer.model_type,
            "task_type": trainer.task_type,
            "n_samples": len(y),
        }

        if trainer.task_type == "classification":
            results.update(self._evaluate_classification(
                y, predictions, trainer.model, X
            ))
        else:
            results.update(self._evaluate_regression(y, predictions))

        self.evaluation_history.append(results)
        return results

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate classification model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model: Trained model
            X: Feature matrix

        Returns:
            Classification metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
        }

        # Binary classification metrics
        if len(np.unique(y_true)) == 2:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics["true_positives"] = int(tp)
                metrics["true_negatives"] = int(tn)
                metrics["false_positives"] = int(fp)
                metrics["false_negatives"] = int(fn)
                metrics["specificity"] = tn / max(1, tn + fp)
                metrics["false_positive_rate"] = fp / max(1, fp + tn)

            # AUC if available
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                    metrics["auc"] = roc_auc_score(y_true, y_prob)

                    # ROC curve points
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    metrics["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                    }

                    # PR curve points
                    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
                    metrics["pr_curve"] = {
                        "precision": precision_arr.tolist(),
                        "recall": recall_arr.tolist(),
                    }
                except Exception:
                    pass

        return metrics

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate regression model.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mean_error": float(np.mean(y_pred - y_true)),
            "std_error": float(np.std(y_pred - y_true)),
        }

    def cross_validate(
        self,
        trainer: BaseTrainer,
        df: pd.DataFrame,
        target_column: str = "target",
        n_folds: int = 5,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Perform cross-validation.

        Args:
            trainer: Model trainer (will be retrained)
            df: DataFrame with features and target
            target_column: Name of target column
            n_folds: Number of CV folds
            model_name: Name for this evaluation

        Returns:
            Cross-validation results
        """
        feature_cols = trainer.get_feature_columns()
        available_features = [f for f in feature_cols if f in df.columns]

        X = df[available_features].values
        y = df[target_column].values
        X = np.nan_to_num(X, nan=0.0)

        # Create model
        trainer._create_model()

        # Scoring metric
        if trainer.task_type == "classification":
            scoring = "accuracy"
            if len(np.unique(y)) == 2:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                cv = n_folds
        else:
            scoring = "neg_mean_squared_error"
            cv = n_folds

        # Cross-validation
        scores = cross_val_score(
            trainer.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
        )

        results = {
            "model_name": model_name,
            "model_type": trainer.model_type,
            "n_folds": n_folds,
            "n_samples": len(y),
            "scores": scores.tolist(),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

        if trainer.task_type == "regression":
            # Convert negative MSE to positive
            results["scores"] = [-s for s in results["scores"]]
            results["mean_score"] = -results["mean_score"]
            results["min_score"], results["max_score"] = (
                -results["max_score"],
                -results["min_score"],
            )

        return results

    def compare_models(
        self,
        trainers: dict[str, BaseTrainer],
        df: pd.DataFrame,
        target_column: str = "target",
        test_size: float = 0.2,
    ) -> pd.DataFrame:
        """Compare multiple models on the same data.

        Args:
            trainers: Dictionary mapping name to trainer
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Test set fraction

        Returns:
            DataFrame comparing model performance
        """
        results = []

        for name, trainer in trainers.items():
            # Train model
            metrics = trainer.train(df, target_column, test_size)

            # Add model info
            metrics["model_name"] = name
            metrics["model_type"] = trainer.model_type
            results.append(metrics)

        return pd.DataFrame(results)

    def get_feature_importance_comparison(
        self,
        trainers: dict[str, BaseTrainer],
    ) -> pd.DataFrame:
        """Compare feature importance across models.

        Args:
            trainers: Dictionary mapping name to trained trainer

        Returns:
            DataFrame with feature importance comparison
        """
        all_features = set()
        importance_data = {}

        for name, trainer in trainers.items():
            if not trainer.is_trained:
                continue

            importance = trainer.get_feature_importance()
            importance_data[name] = importance
            all_features.update(importance.keys())

        # Create comparison DataFrame
        records = []
        for feature in sorted(all_features):
            record = {"feature": feature}
            for name, importance in importance_data.items():
                record[name] = importance.get(feature, 0.0)
            records.append(record)

        df = pd.DataFrame(records)

        # Add average importance
        model_cols = [c for c in df.columns if c != "feature"]
        if model_cols:
            df["avg_importance"] = df[model_cols].mean(axis=1)
            df = df.sort_values("avg_importance", ascending=False)

        return df

    def generate_report(
        self,
        trainer: BaseTrainer,
        df: pd.DataFrame,
        target_column: str = "target",
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Generate comprehensive evaluation report.

        Args:
            trainer: Trained model trainer
            df: DataFrame with features and target
            target_column: Name of target column
            model_name: Name for report

        Returns:
            Comprehensive report dictionary
        """
        feature_cols = trainer.get_feature_columns()
        available_features = [f for f in feature_cols if f in df.columns]

        X = df[available_features].values
        y = df[target_column].values
        X = np.nan_to_num(X, nan=0.0)

        # Basic evaluation
        eval_results = self.evaluate_model(trainer, X, y, model_name)

        # Cross-validation
        cv_results = self.cross_validate(trainer, df, target_column, model_name=model_name)

        # Feature importance
        importance = trainer.get_feature_importance()
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        report = {
            "model_name": model_name,
            "model_type": trainer.model_type,
            "task_type": trainer.task_type,
            "training_metrics": trainer.training_metrics,
            "evaluation_metrics": eval_results,
            "cross_validation": cv_results,
            "feature_importance": dict(sorted_importance[:15]),
            "n_features": len(available_features),
            "features_used": available_features,
        }

        return report

    def save_evaluation_history(self, path: str | Path) -> None:
        """Save evaluation history to file.

        Args:
            path: Path to save to
        """
        df = pd.DataFrame(self.evaluation_history)
        df.to_csv(path, index=False)

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history = []
