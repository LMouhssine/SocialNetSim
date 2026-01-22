"""Temporal feature extraction for ML models.

Provides time-based features including:
- Engagement velocity and acceleration
- Peak timing and decay estimation
- User activity trends and periodicity
- Cascade dynamics (R0 estimation)
"""

from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature extraction.

    Attributes:
        velocity_windows: Window sizes for velocity calculation
        acceleration_windows: Window sizes for acceleration
        trend_window: Window for trend detection
        periodicity_max_lag: Maximum lag for periodicity detection
        decay_estimation_window: Window for decay rate estimation
    """

    velocity_windows: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    acceleration_windows: list[int] = field(default_factory=lambda: [5, 10])
    trend_window: int = 20
    periodicity_max_lag: int = 24
    decay_estimation_window: int = 30


@dataclass
class TemporalFeatures:
    """Container for extracted temporal features.

    Attributes:
        velocities: Engagement velocities at different windows
        accelerations: Engagement accelerations at different windows
        peak_step: Step of peak engagement
        decay_rate: Estimated decay rate
        trend_slope: Activity trend slope
        periodicity_score: Strength of periodic behavior
        time_since_peak: Steps since peak
        momentum: Engagement momentum
    """

    velocities: dict[int, float] = field(default_factory=dict)
    accelerations: dict[int, float] = field(default_factory=dict)
    peak_step: int = 0
    decay_rate: float = 0.0
    trend_slope: float = 0.0
    periodicity_score: float = 0.0
    time_since_peak: int = 0
    momentum: float = 0.0


class TemporalFeatureExtractor:
    """Extracts temporal features from engagement data."""

    def __init__(self, config: TemporalFeatureConfig | None = None):
        """Initialize extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config or TemporalFeatureConfig()

    def extract_post_features(
        self,
        engagement_by_step: dict[int, int],
        current_step: int,
        created_step: int,
    ) -> TemporalFeatures:
        """Extract temporal features for a post.

        Args:
            engagement_by_step: Mapping of step to engagement count
            current_step: Current simulation step
            created_step: Step when post was created

        Returns:
            Extracted temporal features
        """
        features = TemporalFeatures()

        if not engagement_by_step:
            return features

        # Convert to array for easier processing
        steps = sorted(engagement_by_step.keys())
        values = np.array([engagement_by_step[s] for s in steps])

        # Calculate velocities
        features.velocities = self._calculate_velocities(
            engagement_by_step, current_step
        )

        # Calculate accelerations
        features.accelerations = self._calculate_accelerations(
            engagement_by_step, current_step
        )

        # Find peak
        if len(values) > 0:
            peak_idx = np.argmax(values)
            features.peak_step = steps[peak_idx]
            features.time_since_peak = current_step - features.peak_step

        # Estimate decay rate
        features.decay_rate = self._estimate_decay_rate(
            engagement_by_step, current_step
        )

        # Calculate trend
        features.trend_slope = self._calculate_trend(values)

        # Calculate momentum
        features.momentum = self._calculate_momentum(
            engagement_by_step, current_step
        )

        return features

    def extract_user_features(
        self,
        activity_by_step: dict[int, int],
        current_step: int,
    ) -> TemporalFeatures:
        """Extract temporal features for a user's activity.

        Args:
            activity_by_step: Mapping of step to activity count
            current_step: Current simulation step

        Returns:
            Extracted temporal features
        """
        features = TemporalFeatures()

        if not activity_by_step:
            return features

        # Convert to array
        steps = sorted(activity_by_step.keys())
        values = np.array([activity_by_step[s] for s in steps])

        # Velocities and accelerations
        features.velocities = self._calculate_velocities(
            activity_by_step, current_step
        )
        features.accelerations = self._calculate_accelerations(
            activity_by_step, current_step
        )

        # Trend
        features.trend_slope = self._calculate_trend(values)

        # Periodicity
        features.periodicity_score = self._calculate_periodicity(values)

        return features

    def extract_cascade_features(
        self,
        shares_by_step: dict[int, int],
        current_step: int,
        start_step: int,
    ) -> dict[str, float]:
        """Extract cascade-specific temporal features.

        Args:
            shares_by_step: Mapping of step to share count
            current_step: Current simulation step
            start_step: Step when cascade started

        Returns:
            Dictionary of cascade temporal features
        """
        features = {}

        if not shares_by_step:
            return {
                "cascade_age": current_step - start_step,
                "cascade_velocity": 0.0,
                "cascade_acceleration": 0.0,
                "estimated_R0": 0.0,
                "peak_velocity": 0.0,
                "time_to_peak": 0,
                "decay_rate": 0.0,
                "burst_count": 0,
            }

        # Basic temporal features
        age = current_step - start_step
        features["cascade_age"] = age

        # Velocity (recent)
        recent_window = min(5, age)
        recent_shares = sum(
            shares_by_step.get(s, 0)
            for s in range(current_step - recent_window, current_step + 1)
        )
        features["cascade_velocity"] = recent_shares / max(recent_window, 1)

        # Acceleration
        if age > 5:
            prev_window_shares = sum(
                shares_by_step.get(s, 0)
                for s in range(current_step - 10, current_step - 5)
            )
            prev_velocity = prev_window_shares / 5
            features["cascade_acceleration"] = (
                features["cascade_velocity"] - prev_velocity
            )
        else:
            features["cascade_acceleration"] = 0.0

        # Estimate R0 (basic reproduction number)
        features["estimated_R0"] = self._estimate_R0(shares_by_step, start_step)

        # Peak velocity
        step_values = list(shares_by_step.values())
        if step_values:
            features["peak_velocity"] = max(step_values)
            peak_step = max(shares_by_step.keys(), key=lambda s: shares_by_step[s])
            features["time_to_peak"] = peak_step - start_step
        else:
            features["peak_velocity"] = 0.0
            features["time_to_peak"] = 0

        # Decay rate
        features["decay_rate"] = self._estimate_decay_rate(
            shares_by_step, current_step
        )

        # Burst detection
        features["burst_count"] = self._count_bursts(shares_by_step)

        return features

    def _calculate_velocities(
        self,
        values_by_step: dict[int, int],
        current_step: int,
    ) -> dict[int, float]:
        """Calculate velocities for different window sizes.

        Args:
            values_by_step: Values indexed by step
            current_step: Current step

        Returns:
            Dictionary mapping window size to velocity
        """
        velocities = {}

        for window in self.config.velocity_windows:
            total = sum(
                values_by_step.get(s, 0)
                for s in range(current_step - window, current_step + 1)
            )
            velocities[window] = total / window

        return velocities

    def _calculate_accelerations(
        self,
        values_by_step: dict[int, int],
        current_step: int,
    ) -> dict[int, float]:
        """Calculate accelerations for different window sizes.

        Args:
            values_by_step: Values indexed by step
            current_step: Current step

        Returns:
            Dictionary mapping window size to acceleration
        """
        accelerations = {}

        for window in self.config.acceleration_windows:
            # Current velocity
            current_total = sum(
                values_by_step.get(s, 0)
                for s in range(current_step - window, current_step + 1)
            )
            current_vel = current_total / window

            # Previous velocity
            prev_total = sum(
                values_by_step.get(s, 0)
                for s in range(current_step - 2 * window, current_step - window)
            )
            prev_vel = prev_total / window

            accelerations[window] = current_vel - prev_vel

        return accelerations

    def _estimate_decay_rate(
        self,
        values_by_step: dict[int, int],
        current_step: int,
    ) -> float:
        """Estimate exponential decay rate from engagement data.

        Uses linear regression on log-transformed data.

        Args:
            values_by_step: Values indexed by step
            current_step: Current step

        Returns:
            Estimated decay rate (negative = decay)
        """
        if len(values_by_step) < 3:
            return 0.0

        # Find peak
        peak_step = max(values_by_step.keys(), key=lambda s: values_by_step[s])

        # Get post-peak data
        post_peak = {s: v for s, v in values_by_step.items() if s > peak_step and v > 0}

        if len(post_peak) < 2:
            return 0.0

        # Log-linear regression
        steps = np.array(list(post_peak.keys())) - peak_step
        log_values = np.log(np.array(list(post_peak.values())) + 1)

        if len(steps) < 2:
            return 0.0

        # Simple linear regression
        n = len(steps)
        sum_x = np.sum(steps)
        sum_y = np.sum(log_values)
        sum_xy = np.sum(steps * log_values)
        sum_x2 = np.sum(steps ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    def _calculate_trend(self, values: NDArray[np.float64]) -> float:
        """Calculate trend slope from value array.

        Args:
            values: Array of values

        Returns:
            Trend slope
        """
        if len(values) < 2:
            return 0.0

        # Use last N values for trend
        window = min(len(values), self.config.trend_window)
        recent = values[-window:]

        # Simple linear regression
        x = np.arange(window)
        n = window
        sum_x = np.sum(x)
        sum_y = np.sum(recent)
        sum_xy = np.sum(x * recent)
        sum_x2 = np.sum(x ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    def _calculate_periodicity(self, values: NDArray[np.float64]) -> float:
        """Calculate periodicity score using autocorrelation.

        Args:
            values: Array of values

        Returns:
            Periodicity score (0-1)
        """
        if len(values) < self.config.periodicity_max_lag * 2:
            return 0.0

        # Normalize
        values = values - np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return 0.0
        values = values / std

        # Calculate autocorrelation at various lags
        autocorrs = []
        for lag in range(1, self.config.periodicity_max_lag + 1):
            if lag >= len(values):
                break
            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(abs(corr))

        if not autocorrs:
            return 0.0

        # Return max autocorrelation as periodicity score
        return max(autocorrs)

    def _calculate_momentum(
        self,
        values_by_step: dict[int, int],
        current_step: int,
    ) -> float:
        """Calculate engagement momentum.

        Momentum = velocity * acceleration sign

        Args:
            values_by_step: Values indexed by step
            current_step: Current step

        Returns:
            Momentum value
        """
        velocities = self._calculate_velocities(values_by_step, current_step)
        accelerations = self._calculate_accelerations(values_by_step, current_step)

        if 5 not in velocities or 5 not in accelerations:
            return 0.0

        velocity = velocities[5]
        accel_sign = 1 if accelerations[5] > 0 else -1

        return velocity * accel_sign

    def _estimate_R0(
        self,
        shares_by_step: dict[int, int],
        start_step: int,
    ) -> float:
        """Estimate basic reproduction number R0.

        R0 = average secondary shares per share in early phase.

        Args:
            shares_by_step: Shares indexed by step
            start_step: Cascade start step

        Returns:
            Estimated R0
        """
        if not shares_by_step:
            return 0.0

        # Look at early cascade growth
        total_shares = sum(shares_by_step.values())
        if total_shares <= 1:
            return 0.0

        # Simple estimation: total shares / initial shares
        # More sophisticated would use generation tracking
        early_steps = 5
        early_shares = sum(
            shares_by_step.get(s, 0)
            for s in range(start_step, start_step + early_steps)
        )

        if early_shares == 0:
            return 0.0

        late_shares = total_shares - early_shares
        return late_shares / early_shares

    def _count_bursts(
        self,
        values_by_step: dict[int, int],
        threshold_multiplier: float = 2.0,
    ) -> int:
        """Count burst events in time series.

        A burst is when value exceeds threshold_multiplier * mean.

        Args:
            values_by_step: Values indexed by step
            threshold_multiplier: Multiplier for burst threshold

        Returns:
            Number of bursts
        """
        if not values_by_step:
            return 0

        values = np.array(list(values_by_step.values()))
        mean_val = np.mean(values)

        if mean_val < 1e-10:
            return 0

        threshold = mean_val * threshold_multiplier
        bursts = np.sum(values > threshold)

        return int(bursts)


class TemporalWindowAggregator:
    """Aggregates temporal features across multiple windows."""

    def __init__(
        self,
        windows: list[int] | None = None,
    ):
        """Initialize aggregator.

        Args:
            windows: Window sizes to aggregate over
        """
        self.windows = windows or [1, 5, 10, 20, 50]

    def aggregate_engagement(
        self,
        engagement_history: list[dict[str, Any]],
        current_step: int,
    ) -> dict[str, float]:
        """Aggregate engagement features across windows.

        Args:
            engagement_history: List of engagement events with 'step' and 'type'
            current_step: Current simulation step

        Returns:
            Dictionary of aggregated features
        """
        features = {}

        # Count by type
        type_counts = defaultdict(lambda: defaultdict(int))
        for event in engagement_history:
            step = event.get("step", 0)
            event_type = event.get("type", "view")
            for window in self.windows:
                if step >= current_step - window:
                    type_counts[window][event_type] += 1

        # Generate features for each window
        for window in self.windows:
            prefix = f"w{window}_"
            counts = type_counts[window]

            features[f"{prefix}views"] = counts.get("view", 0)
            features[f"{prefix}likes"] = counts.get("like", 0)
            features[f"{prefix}shares"] = counts.get("share", 0)
            features[f"{prefix}comments"] = counts.get("comment", 0)
            features[f"{prefix}total"] = sum(counts.values())

            # Rates
            features[f"{prefix}engagement_rate"] = (
                (counts.get("like", 0) + counts.get("share", 0)) /
                max(counts.get("view", 0), 1)
            )

        return features

    def aggregate_activity(
        self,
        activity_steps: list[int],
        current_step: int,
    ) -> dict[str, float]:
        """Aggregate user activity features across windows.

        Args:
            activity_steps: List of steps when user was active
            current_step: Current simulation step

        Returns:
            Dictionary of aggregated features
        """
        features = {}

        for window in self.windows:
            prefix = f"w{window}_"
            cutoff = current_step - window

            active_count = sum(1 for s in activity_steps if s >= cutoff)
            features[f"{prefix}active_count"] = active_count
            features[f"{prefix}activity_rate"] = active_count / window

            # Recency
            recent_activity = [s for s in activity_steps if s >= cutoff]
            if recent_activity:
                features[f"{prefix}last_active_ago"] = current_step - max(recent_activity)
            else:
                features[f"{prefix}last_active_ago"] = window

        return features


def extract_all_temporal_features(
    post_engagements: dict[str, dict[int, int]],
    user_activities: dict[str, dict[int, int]],
    cascade_shares: dict[str, dict[int, int]],
    cascade_starts: dict[str, int],
    current_step: int,
) -> dict[str, dict[str, float]]:
    """Extract all temporal features for posts, users, and cascades.

    Args:
        post_engagements: Post ID to step->engagement count mapping
        user_activities: User ID to step->activity count mapping
        cascade_shares: Cascade ID to step->share count mapping
        cascade_starts: Cascade ID to start step mapping
        current_step: Current simulation step

    Returns:
        Dictionary with 'posts', 'users', 'cascades' feature dictionaries
    """
    extractor = TemporalFeatureExtractor()

    results = {
        "posts": {},
        "users": {},
        "cascades": {},
    }

    # Extract post features
    for post_id, engagements in post_engagements.items():
        if engagements:
            created_step = min(engagements.keys())
            features = extractor.extract_post_features(
                engagements, current_step, created_step
            )
            results["posts"][post_id] = {
                "velocities": features.velocities,
                "accelerations": features.accelerations,
                "peak_step": features.peak_step,
                "decay_rate": features.decay_rate,
                "trend_slope": features.trend_slope,
                "momentum": features.momentum,
                "time_since_peak": features.time_since_peak,
            }

    # Extract user features
    for user_id, activities in user_activities.items():
        features = extractor.extract_user_features(activities, current_step)
        results["users"][user_id] = {
            "velocities": features.velocities,
            "accelerations": features.accelerations,
            "trend_slope": features.trend_slope,
            "periodicity_score": features.periodicity_score,
        }

    # Extract cascade features
    for cascade_id, shares in cascade_shares.items():
        start_step = cascade_starts.get(cascade_id, 0)
        results["cascades"][cascade_id] = extractor.extract_cascade_features(
            shares, current_step, start_step
        )

    return results
