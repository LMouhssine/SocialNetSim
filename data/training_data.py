"""ML-ready dataset preparation from simulation data.

Provides:
- Temporal train/test splitting
- Feature/label extraction
- Balanced sampling
- Data augmentation strategies
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from enum import Enum
import json
import random

import numpy as np
import pandas as pd

from models import User, Post, Interaction, Cascade
from models.enums import InteractionType
from engine.state import SimulationState


class DatasetType(Enum):
    """Types of ML datasets."""

    VIRALITY_PREDICTION = "virality"
    ENGAGEMENT_PREDICTION = "engagement"
    CHURN_PREDICTION = "churn"
    MISINFO_DETECTION = "misinfo"
    CASCADE_SIZE = "cascade_size"
    USER_BEHAVIOR = "user_behavior"


@dataclass
class DatasetConfig:
    """Configuration for dataset creation.

    Attributes:
        dataset_type: Type of prediction task
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        temporal_split: Use temporal splitting (train on earlier data)
        balance_classes: Balance positive/negative classes
        min_samples_per_class: Minimum samples per class after balancing
        include_features: Features to include (None for all)
        exclude_features: Features to exclude
        label_threshold: Threshold for binary classification
    """

    dataset_type: DatasetType
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    temporal_split: bool = True
    balance_classes: bool = True
    min_samples_per_class: int = 100
    include_features: list[str] | None = None
    exclude_features: list[str] = field(default_factory=list)
    label_threshold: float | None = None


@dataclass
class Dataset:
    """An ML-ready dataset.

    Attributes:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
        label_name: Name of label column
        metadata: Additional information
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert to pandas DataFrames.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        train_df[self.label_name] = self.y_train

        val_df = pd.DataFrame(self.X_val, columns=self.feature_names)
        val_df[self.label_name] = self.y_val

        test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        test_df[self.label_name] = self.y_test

        return train_df, val_df, test_df

    def save(self, path: str | Path) -> None:
        """Save dataset to disk.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.savez_compressed(
            path / "arrays.npz",
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
        )

        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "label_name": self.label_name,
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Load dataset from disk.

        Args:
            path: Directory to load from

        Returns:
            Loaded Dataset
        """
        path = Path(path)

        # Load arrays
        arrays = np.load(path / "arrays.npz")

        # Load metadata
        with open(path / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            X_train=arrays["X_train"],
            y_train=arrays["y_train"],
            X_val=arrays["X_val"],
            y_val=arrays["y_val"],
            X_test=arrays["X_test"],
            y_test=arrays["y_test"],
            feature_names=meta["feature_names"],
            label_name=meta["label_name"],
            metadata=meta.get("metadata", {}),
        )


class TrainingDataPreparer:
    """Prepares ML-ready datasets from simulation data."""

    def __init__(self, seed: int | None = None):
        """Initialize preparer.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

    def prepare_virality_dataset(
        self,
        posts: list[Post],
        users: dict[str, User],
        state: SimulationState,
        config: DatasetConfig,
        feature_extractor: Any = None,
    ) -> Dataset:
        """Prepare dataset for virality prediction.

        Args:
            posts: List of posts
            users: Dictionary of users
            state: Simulation state
            config: Dataset configuration
            feature_extractor: Optional custom feature extractor

        Returns:
            Prepared Dataset
        """
        from ai.features import FeatureExtractor

        extractor = feature_extractor or FeatureExtractor()

        # Extract features and labels
        records = []
        for post in posts:
            author = users.get(post.author_id)
            features = extractor.extract_post_features(post, author, state)

            # Determine label based on config
            if config.label_threshold is not None:
                label = float(post.share_count >= config.label_threshold)
            else:
                # Default: viral if share_count >= 10
                label = float(post.share_count >= 10)

            features["_label"] = label
            features["_step"] = post.created_step
            records.append(features)

        df = pd.DataFrame(records)
        return self._create_dataset(df, config, "_label")

    def prepare_engagement_dataset(
        self,
        interactions: list[tuple[User, Post, bool]],
        config: DatasetConfig,
        feature_extractor: Any = None,
    ) -> Dataset:
        """Prepare dataset for engagement prediction.

        Args:
            interactions: List of (user, post, engaged) tuples
            config: Dataset configuration
            feature_extractor: Optional custom feature extractor

        Returns:
            Prepared Dataset
        """
        from ai.features import FeatureExtractor

        extractor = feature_extractor or FeatureExtractor()

        records = []
        for user, post, engaged in interactions:
            author = None  # Could pass if available
            features = extractor.extract_interaction_features(user, post, author)
            features["_label"] = float(engaged)
            features["_step"] = post.created_step
            records.append(features)

        df = pd.DataFrame(records)
        return self._create_dataset(df, config, "_label")

    def prepare_churn_dataset(
        self,
        users: dict[str, User],
        state: SimulationState,
        config: DatasetConfig,
        churn_threshold_steps: int = 10,
        feature_extractor: Any = None,
    ) -> Dataset:
        """Prepare dataset for churn prediction.

        Args:
            users: Dictionary of users
            state: Simulation state
            config: Dataset configuration
            churn_threshold_steps: Steps of inactivity = churn
            feature_extractor: Optional custom feature extractor

        Returns:
            Prepared Dataset
        """
        from ai.features import FeatureExtractor

        extractor = feature_extractor or FeatureExtractor()

        records = []
        for user in users.values():
            features = extractor.extract_user_features(user, state)

            # Churn label
            user_state = state.get_user_state(user.user_id)
            if user_state:
                steps_inactive = state.current_step - user_state.last_active_step
                label = float(steps_inactive >= churn_threshold_steps)
            else:
                label = 0.0

            features["_label"] = label
            features["_step"] = user.last_active_step
            records.append(features)

        df = pd.DataFrame(records)
        return self._create_dataset(df, config, "_label")

    def prepare_cascade_dataset(
        self,
        cascades: list[Cascade],
        posts: dict[str, Post],
        users: dict[str, User],
        state: SimulationState,
        config: DatasetConfig,
        feature_extractor: Any = None,
    ) -> Dataset:
        """Prepare dataset for cascade size prediction.

        Args:
            cascades: List of cascades
            posts: Dictionary of posts
            users: Dictionary of users
            state: Simulation state
            config: Dataset configuration
            feature_extractor: Optional custom feature extractor

        Returns:
            Prepared Dataset
        """
        from ai.features import FeatureExtractor

        extractor = feature_extractor or FeatureExtractor()

        records = []
        for cascade in cascades:
            post = posts.get(cascade.post_id)
            if post is None:
                continue

            author = users.get(post.author_id)
            features = extractor.extract_post_features(post, author, state)

            # Add cascade-specific features
            features["cascade_start_step"] = cascade.start_step
            features["cascade_initial_reach"] = min(10, cascade.total_reach)

            # Label: final cascade size
            if config.label_threshold is not None:
                label = float(cascade.total_shares >= config.label_threshold)
            else:
                label = float(cascade.total_shares)

            features["_label"] = label
            features["_step"] = cascade.start_step
            records.append(features)

        df = pd.DataFrame(records)
        return self._create_dataset(df, config, "_label")

    def _create_dataset(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        label_col: str,
    ) -> Dataset:
        """Create dataset from DataFrame with splitting and balancing.

        Args:
            df: DataFrame with features and label
            config: Dataset configuration
            label_col: Name of label column

        Returns:
            Prepared Dataset
        """
        # Identify step column if exists
        step_col = "_step" if "_step" in df.columns else None

        # Get feature columns
        feature_cols = [
            c for c in df.columns
            if not c.startswith("_") and c != label_col
        ]

        # Apply include/exclude filters
        if config.include_features:
            feature_cols = [c for c in feature_cols if c in config.include_features]
        feature_cols = [c for c in feature_cols if c not in config.exclude_features]

        # Split data
        if config.temporal_split and step_col:
            df = df.sort_values(step_col)
            n = len(df)
            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
        else:
            # Random split
            indices = list(range(len(df)))
            self.random.shuffle(indices)

            n = len(df)
            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            train_df = df.iloc[indices[:train_end]]
            val_df = df.iloc[indices[train_end:val_end]]
            test_df = df.iloc[indices[val_end:]]

        # Balance classes if needed
        if config.balance_classes:
            train_df = self._balance_classes(train_df, label_col, config.min_samples_per_class)

        # Extract arrays
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[label_col].values.astype(np.float32)

        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        return Dataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            label_name=label_col.strip("_"),
            metadata={
                "config": {
                    "dataset_type": config.dataset_type.value,
                    "train_ratio": config.train_ratio,
                    "val_ratio": config.val_ratio,
                    "temporal_split": config.temporal_split,
                    "balance_classes": config.balance_classes,
                },
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "n_features": len(feature_cols),
            },
        )

    def _balance_classes(
        self,
        df: pd.DataFrame,
        label_col: str,
        min_samples: int,
    ) -> pd.DataFrame:
        """Balance classes through undersampling majority class.

        Args:
            df: DataFrame to balance
            label_col: Label column name
            min_samples: Minimum samples per class

        Returns:
            Balanced DataFrame
        """
        # Group by label
        positive = df[df[label_col] > 0.5]
        negative = df[df[label_col] <= 0.5]

        # Determine target size
        target_size = max(min_samples, min(len(positive), len(negative)))

        # Sample from each class
        if len(positive) > target_size:
            positive = positive.sample(target_size, random_state=self.random.randint(0, 2**31))
        if len(negative) > target_size:
            negative = negative.sample(target_size, random_state=self.random.randint(0, 2**31))

        return pd.concat([positive, negative]).sample(frac=1, random_state=self.random.randint(0, 2**31))

    def create_sliding_window_dataset(
        self,
        state: SimulationState,
        users: dict[str, User],
        window_size: int = 10,
        step_size: int = 5,
        config: DatasetConfig | None = None,
    ) -> Iterator[Dataset]:
        """Create datasets using sliding window over simulation steps.

        Useful for time-series analysis and temporal validation.

        Args:
            state: Simulation state
            users: Dictionary of users
            window_size: Size of each window in steps
            step_size: Steps between window starts
            config: Dataset configuration

        Yields:
            Dataset for each window
        """
        config = config or DatasetConfig(
            dataset_type=DatasetType.VIRALITY_PREDICTION,
            temporal_split=False,
        )

        max_step = state.current_step

        for start_step in range(0, max_step - window_size + 1, step_size):
            end_step = start_step + window_size

            # Filter posts to this window
            window_posts = [
                p for p in state.posts.values()
                if start_step <= p.created_step < end_step
            ]

            if len(window_posts) < config.min_samples_per_class * 2:
                continue

            # Create dataset for this window
            dataset = self.prepare_virality_dataset(
                window_posts,
                users,
                state,
                config,
            )

            dataset.metadata["window"] = {
                "start_step": start_step,
                "end_step": end_step,
                "n_posts": len(window_posts),
            }

            yield dataset


class DataAugmenter:
    """Data augmentation strategies for simulation data."""

    def __init__(self, seed: int | None = None):
        """Initialize augmenter.

        Args:
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)

    def augment_features(
        self,
        X: np.ndarray,
        noise_scale: float = 0.01,
        dropout_rate: float = 0.0,
    ) -> np.ndarray:
        """Augment feature matrix with noise and dropout.

        Args:
            X: Feature matrix
            noise_scale: Scale of Gaussian noise
            dropout_rate: Probability of zeroing features

        Returns:
            Augmented feature matrix
        """
        X_aug = X.copy()

        # Add Gaussian noise
        if noise_scale > 0:
            noise = self.rng.normal(0, noise_scale, X.shape)
            X_aug = X_aug + noise

        # Apply dropout
        if dropout_rate > 0:
            mask = self.rng.random(X.shape) > dropout_rate
            X_aug = X_aug * mask

        return X_aug

    def oversample_minority(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oversample minority class with augmentation.

        Args:
            X: Feature matrix
            y: Labels
            target_ratio: Target ratio of minority to majority

        Returns:
            Tuple of (augmented X, augmented y)
        """
        # Identify minority class
        positive_mask = y > 0.5
        n_positive = positive_mask.sum()
        n_negative = (~positive_mask).sum()

        if n_positive >= n_negative:
            return X, y

        # Calculate samples needed
        target_positive = int(n_negative * target_ratio)
        samples_needed = target_positive - n_positive

        if samples_needed <= 0:
            return X, y

        # Sample and augment minority class
        positive_indices = np.where(positive_mask)[0]
        sample_indices = self.rng.choice(
            positive_indices,
            size=samples_needed,
            replace=True,
        )

        X_new = self.augment_features(X[sample_indices])
        y_new = y[sample_indices]

        return np.vstack([X, X_new]), np.concatenate([y, y_new])
