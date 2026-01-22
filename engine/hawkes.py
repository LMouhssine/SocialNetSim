"""Hawkes process implementation for viral cascade dynamics.

Implements self-exciting point processes where:
- Each event increases the probability of future events
- Intensity: λ(t) = μ + Σ α * exp(-β * (t - t_i))

Used for:
- Realistic cascade timing
- Multi-dimensional event modeling (like, share, comment)
- Cascade prediction
"""

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
from numpy.random import Generator


@dataclass
class HawkesEvent:
    """An event in the Hawkes process.

    Attributes:
        time: Event timestamp
        event_type: Type of event (for multi-dimensional)
        user_id: User who triggered the event
        metadata: Additional event data
    """

    time: float
    event_type: str = "default"
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HawkesConfig:
    """Configuration for Hawkes process.

    Attributes:
        baseline: Baseline intensity μ
        branching_ratio: Expected offspring per event (α/β)
        decay: Exponential decay rate β
        max_intensity: Cap on intensity to prevent explosion
    """

    baseline: float = 0.01
    branching_ratio: float = 0.8
    decay: float = 0.1
    max_intensity: float = 10.0

    @property
    def alpha(self) -> float:
        """Get excitation parameter α = branching_ratio * decay."""
        return self.branching_ratio * self.decay


class HawkesProcess:
    """Self-exciting point process for viral dynamics.

    Intensity function: λ(t) = μ + Σ α * exp(-β * (t - t_i))

    Where:
    - μ: Baseline intensity
    - α: Excitation parameter (jump size)
    - β: Decay rate
    - t_i: Times of past events
    """

    def __init__(
        self,
        config: HawkesConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize Hawkes process.

        Args:
            config: Process configuration
            seed: Random seed
        """
        self.config = config or HawkesConfig()
        self.rng = np.random.default_rng(seed)

        # Event history
        self.events: list[HawkesEvent] = []
        self.event_times: list[float] = []

        # For efficient intensity calculation
        self._intensity_cache: dict[float, float] = {}

    def reset(self) -> None:
        """Reset process state."""
        self.events = []
        self.event_times = []
        self._intensity_cache = {}

    def add_event(self, time: float, event_type: str = "default", **kwargs) -> HawkesEvent:
        """Add an event to the process.

        Args:
            time: Event timestamp
            event_type: Type of event
            **kwargs: Additional event metadata

        Returns:
            Created HawkesEvent
        """
        event = HawkesEvent(time=time, event_type=event_type, **kwargs)
        self.events.append(event)
        self.event_times.append(time)
        self._intensity_cache = {}  # Invalidate cache
        return event

    def intensity(self, t: float) -> float:
        """Calculate intensity at time t.

        λ(t) = μ + Σ α * exp(-β * (t - t_i)) for all t_i < t

        Args:
            t: Time to calculate intensity

        Returns:
            Intensity value
        """
        # Baseline
        intensity = self.config.baseline

        # Add contributions from past events
        alpha = self.config.alpha
        beta = self.config.decay

        for event_time in self.event_times:
            if event_time < t:
                # Excitation contribution
                intensity += alpha * math.exp(-beta * (t - event_time))

        # Cap intensity to prevent explosion
        return min(intensity, self.config.max_intensity)

    def intensity_integral(self, t_start: float, t_end: float) -> float:
        """Calculate integral of intensity from t_start to t_end.

        ∫[t_start, t_end] λ(t) dt

        Used for thinning algorithm.

        Args:
            t_start: Start time
            t_end: End time

        Returns:
            Integral value
        """
        # Baseline contribution
        integral = self.config.baseline * (t_end - t_start)

        # Excitation contributions
        alpha = self.config.alpha
        beta = self.config.decay

        for event_time in self.event_times:
            if event_time < t_end:
                # Integral of α * exp(-β * (t - t_i)) from max(t_start, t_i) to t_end
                t_lower = max(t_start, event_time)
                if t_lower < t_end:
                    # ∫ α * exp(-β * (t - t_i)) dt = -α/β * exp(-β * (t - t_i))
                    contrib = (alpha / beta) * (
                        math.exp(-beta * (t_lower - event_time)) -
                        math.exp(-beta * (t_end - event_time))
                    )
                    integral += contrib

        return integral

    def sample_next_event(
        self,
        t_start: float,
        t_max: float | None = None,
    ) -> float | None:
        """Sample next event time using Ogata's thinning algorithm.

        Args:
            t_start: Start time for sampling
            t_max: Maximum time (returns None if no event before t_max)

        Returns:
            Next event time or None
        """
        t = t_start

        # Upper bound on intensity
        intensity_bound = self.config.max_intensity

        while True:
            # Sample from homogeneous Poisson with rate intensity_bound
            dt = self.rng.exponential(1.0 / intensity_bound)
            t = t + dt

            # Check if past maximum time
            if t_max is not None and t > t_max:
                return None

            # Accept/reject based on actual intensity
            actual_intensity = self.intensity(t)
            acceptance_prob = actual_intensity / intensity_bound

            if self.rng.random() < acceptance_prob:
                return t

    def simulate(
        self,
        t_max: float,
        t_start: float = 0.0,
    ) -> list[float]:
        """Simulate the process up to time t_max.

        Args:
            t_max: Maximum simulation time
            t_start: Starting time

        Returns:
            List of event times
        """
        simulated_times = []
        t = t_start

        while True:
            t_next = self.sample_next_event(t, t_max)
            if t_next is None:
                break
            simulated_times.append(t_next)
            self.add_event(t_next)
            t = t_next

        return simulated_times

    def expected_events(self, t_start: float, t_end: float) -> float:
        """Calculate expected number of events in time interval.

        Args:
            t_start: Start time
            t_end: End time

        Returns:
            Expected event count
        """
        return self.intensity_integral(t_start, t_end)

    def branching_factor(self) -> float:
        """Get branching factor (expected offspring per event).

        Returns:
            Branching factor
        """
        return self.config.branching_ratio


@dataclass
class MultiDimHawkesConfig:
    """Configuration for multi-dimensional Hawkes process.

    Attributes:
        dimensions: List of event type names
        baselines: Baseline intensity for each dimension
        excitation_matrix: α[i,j] = excitation of dim j by event in dim i
        decay_matrix: β[i,j] = decay rate for excitation of dim j by dim i
    """

    dimensions: list[str] = field(default_factory=lambda: ["like", "share", "comment"])
    baselines: np.ndarray | None = None
    excitation_matrix: np.ndarray | None = None
    decay_matrix: np.ndarray | None = None

    def __post_init__(self):
        """Initialize default matrices if not provided."""
        d = len(self.dimensions)

        if self.baselines is None:
            # Default baselines: shares rare, likes common
            self.baselines = np.array([0.05, 0.01, 0.02])[:d]

        if self.excitation_matrix is None:
            # Default: each event type excites itself and others
            self.excitation_matrix = np.array([
                [0.03, 0.01, 0.01],  # Likes trigger more likes, some shares/comments
                [0.02, 0.05, 0.02],  # Shares trigger cascade (more shares)
                [0.02, 0.01, 0.04],  # Comments trigger more comments
            ])[:d, :d]

        if self.decay_matrix is None:
            # Default decay rates
            self.decay_matrix = np.ones((d, d)) * 0.1


class MultiDimensionalHawkes:
    """Multi-dimensional Hawkes process for different event types.

    Models interactions between event types:
    - Likes may trigger shares
    - Shares may trigger more shares (cascade)
    - Comments may trigger more comments
    """

    def __init__(
        self,
        config: MultiDimHawkesConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize multi-dimensional Hawkes process.

        Args:
            config: Process configuration
            seed: Random seed
        """
        self.config = config or MultiDimHawkesConfig()
        self.rng = np.random.default_rng(seed)

        self.d = len(self.config.dimensions)
        self.dim_to_idx = {dim: i for i, dim in enumerate(self.config.dimensions)}

        # Events by dimension
        self.events: dict[str, list[HawkesEvent]] = {
            dim: [] for dim in self.config.dimensions
        }
        self.event_times: dict[str, list[float]] = {
            dim: [] for dim in self.config.dimensions
        }

    def reset(self) -> None:
        """Reset process state."""
        self.events = {dim: [] for dim in self.config.dimensions}
        self.event_times = {dim: [] for dim in self.config.dimensions}

    def add_event(
        self,
        time: float,
        event_type: str,
        **kwargs,
    ) -> HawkesEvent:
        """Add an event to the process.

        Args:
            time: Event timestamp
            event_type: Type of event (must be in dimensions)
            **kwargs: Additional event metadata

        Returns:
            Created HawkesEvent
        """
        if event_type not in self.dim_to_idx:
            raise ValueError(f"Unknown event type: {event_type}")

        event = HawkesEvent(time=time, event_type=event_type, **kwargs)
        self.events[event_type].append(event)
        self.event_times[event_type].append(time)
        return event

    def intensity(self, t: float, dimension: str) -> float:
        """Calculate intensity for a specific dimension at time t.

        λ_j(t) = μ_j + Σ_i Σ_{t_k^i < t} α_{ij} * exp(-β_{ij} * (t - t_k^i))

        Args:
            t: Time to calculate intensity
            dimension: Which dimension to calculate

        Returns:
            Intensity value
        """
        j = self.dim_to_idx[dimension]

        # Baseline
        intensity = self.config.baselines[j]

        # Add contributions from all dimensions
        for i, dim_i in enumerate(self.config.dimensions):
            alpha = self.config.excitation_matrix[i, j]
            beta = self.config.decay_matrix[i, j]

            for event_time in self.event_times[dim_i]:
                if event_time < t:
                    intensity += alpha * math.exp(-beta * (t - event_time))

        return intensity

    def total_intensity(self, t: float) -> float:
        """Calculate total intensity across all dimensions.

        Args:
            t: Time to calculate intensity

        Returns:
            Total intensity
        """
        return sum(
            self.intensity(t, dim)
            for dim in self.config.dimensions
        )

    def sample_next_event(
        self,
        t_start: float,
        t_max: float | None = None,
    ) -> tuple[float, str] | None:
        """Sample next event time and type using thinning.

        Args:
            t_start: Start time for sampling
            t_max: Maximum time

        Returns:
            Tuple of (time, dimension) or None
        """
        t = t_start

        # Upper bound on total intensity
        max_total = sum(self.config.baselines) + np.sum(self.config.excitation_matrix) * 10
        intensity_bound = max(1.0, max_total)

        while True:
            # Sample from homogeneous Poisson
            dt = self.rng.exponential(1.0 / intensity_bound)
            t = t + dt

            if t_max is not None and t > t_max:
                return None

            # Calculate actual intensities
            intensities = [
                self.intensity(t, dim)
                for dim in self.config.dimensions
            ]
            total = sum(intensities)

            # Accept/reject
            acceptance_prob = total / intensity_bound
            if self.rng.random() < acceptance_prob:
                # Select dimension proportional to intensity
                probs = np.array(intensities) / total
                dim_idx = self.rng.choice(self.d, p=probs)
                return t, self.config.dimensions[dim_idx]

    def simulate(
        self,
        t_max: float,
        t_start: float = 0.0,
    ) -> list[tuple[float, str]]:
        """Simulate the process up to time t_max.

        Args:
            t_max: Maximum simulation time
            t_start: Starting time

        Returns:
            List of (time, dimension) tuples
        """
        simulated = []
        t = t_start

        while True:
            result = self.sample_next_event(t, t_max)
            if result is None:
                break
            time, dim = result
            simulated.append((time, dim))
            self.add_event(time, dim)
            t = time

        return simulated

    def get_statistics(self) -> dict[str, Any]:
        """Get process statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "dimensions": self.config.dimensions,
            "event_counts": {
                dim: len(self.events[dim])
                for dim in self.config.dimensions
            },
            "baselines": self.config.baselines.tolist(),
        }


class CascadeHawkesProcess:
    """Hawkes process specialized for cascade modeling.

    Tracks a single cascade with intensity based on:
    - Initial virality potential
    - Recent share activity
    - Content and author factors
    """

    def __init__(
        self,
        baseline: float = 0.01,
        branching_ratio: float = 0.8,
        decay: float = 0.1,
        virality_boost: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize cascade Hawkes process.

        Args:
            baseline: Base intensity
            branching_ratio: Expected offspring per share
            decay: Decay rate
            virality_boost: Multiplier for viral content
            seed: Random seed
        """
        self.config = HawkesConfig(
            baseline=baseline,
            branching_ratio=branching_ratio,
            decay=decay,
        )
        self.virality_boost = virality_boost
        self.rng = np.random.default_rng(seed)

        self.hawkes = HawkesProcess(self.config, seed)

    def initialize_cascade(
        self,
        start_time: float,
        virality_score: float,
        author_influence: float,
    ) -> None:
        """Initialize cascade with initial event.

        Args:
            start_time: Cascade start time
            virality_score: Content virality score
            author_influence: Author's influence score
        """
        self.hawkes.reset()

        # Adjust baseline based on virality and author
        self.hawkes.config.baseline = (
            self.config.baseline *
            (1 + virality_score * self.virality_boost) *
            (1 + author_influence * 0.5)
        )

        # Add initial event
        self.hawkes.add_event(start_time, event_type="initial")

    def record_share(self, time: float, user_id: str) -> None:
        """Record a share event.

        Args:
            time: Share timestamp
            user_id: Sharing user ID
        """
        self.hawkes.add_event(time, event_type="share", user_id=user_id)

    def get_share_probability(self, t: float, base_prob: float) -> float:
        """Get share probability at time t, modulated by Hawkes intensity.

        Args:
            t: Current time
            base_prob: Base share probability

        Returns:
            Modulated share probability
        """
        intensity = self.hawkes.intensity(t)

        # Intensity modulates the base probability
        # Higher intensity = higher share probability
        modulated = base_prob * (1 + intensity / self.config.baseline)

        return min(0.9, modulated)

    def sample_next_share_time(
        self,
        t_start: float,
        t_max: float | None = None,
    ) -> float | None:
        """Sample next share time.

        Args:
            t_start: Start time
            t_max: Maximum time

        Returns:
            Next share time or None
        """
        return self.hawkes.sample_next_event(t_start, t_max)

    def get_expected_shares(self, t_start: float, t_end: float) -> float:
        """Get expected number of shares in time interval.

        Args:
            t_start: Start time
            t_end: End time

        Returns:
            Expected share count
        """
        return self.hawkes.expected_events(t_start, t_end)

    def get_current_intensity(self, t: float) -> float:
        """Get current cascade intensity.

        Args:
            t: Current time

        Returns:
            Intensity value
        """
        return self.hawkes.intensity(t)
