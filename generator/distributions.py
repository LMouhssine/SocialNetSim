"""Statistical distribution utilities for data generation."""

from typing import Any

import numpy as np
from numpy.random import Generator

from config.schemas import DistributionConfig


def sample_normal(
    rng: Generator,
    mean: float = 0.0,
    std: float = 1.0,
    min_val: float | None = None,
    max_val: float | None = None,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from a normal distribution with optional clipping.

    Args:
        rng: NumPy random generator
        mean: Mean of the distribution
        std: Standard deviation
        min_val: Minimum value (clip)
        max_val: Maximum value (clip)
        size: Output shape

    Returns:
        Sampled value(s)
    """
    samples = rng.normal(mean, std, size=size)

    if min_val is not None or max_val is not None:
        samples = np.clip(samples, min_val, max_val)

    return samples


def sample_beta(
    rng: Generator,
    alpha: float = 2.0,
    beta: float = 5.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from a beta distribution scaled to [min_val, max_val].

    Args:
        rng: NumPy random generator
        alpha: Alpha parameter (shape)
        beta: Beta parameter (shape)
        min_val: Minimum value
        max_val: Maximum value
        size: Output shape

    Returns:
        Sampled value(s) in [min_val, max_val]
    """
    samples = rng.beta(alpha, beta, size=size)

    # Scale to desired range
    samples = min_val + samples * (max_val - min_val)

    return samples


def sample_uniform(
    rng: Generator,
    min_val: float = 0.0,
    max_val: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from a uniform distribution.

    Args:
        rng: NumPy random generator
        min_val: Minimum value
        max_val: Maximum value
        size: Output shape

    Returns:
        Sampled value(s)
    """
    return rng.uniform(min_val, max_val, size=size)


def sample_power_law(
    rng: Generator,
    alpha: float = 2.0,
    min_val: float = 1.0,
    max_val: float | None = None,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from a power law distribution.

    Uses inverse transform sampling: X = x_min * (1 - U)^(-1/(alpha-1))

    Args:
        rng: NumPy random generator
        alpha: Exponent (must be > 1)
        min_val: Minimum value (x_min)
        max_val: Maximum value (optional cap)
        size: Output shape

    Returns:
        Sampled value(s)
    """
    if alpha <= 1:
        raise ValueError("Alpha must be greater than 1 for power law")

    u = rng.uniform(0, 1, size=size)
    samples = min_val * (1 - u) ** (-1 / (alpha - 1))

    if max_val is not None:
        samples = np.minimum(samples, max_val)

    return samples


def sample_exponential(
    rng: Generator,
    scale: float = 1.0,
    min_val: float = 0.0,
    max_val: float | None = None,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from an exponential distribution.

    Args:
        rng: NumPy random generator
        scale: Scale parameter (1/lambda)
        min_val: Minimum value (shift)
        max_val: Maximum value (cap)
        size: Output shape

    Returns:
        Sampled value(s)
    """
    samples = rng.exponential(scale, size=size) + min_val

    if max_val is not None:
        samples = np.minimum(samples, max_val)

    return samples


def sample_from_config(
    rng: Generator,
    config: DistributionConfig,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample from a distribution based on configuration.

    Args:
        rng: NumPy random generator
        config: Distribution configuration
        size: Output shape

    Returns:
        Sampled value(s)
    """
    dist_type = config.type.lower()

    if dist_type == "normal":
        return sample_normal(
            rng,
            mean=config.mean,
            std=config.std,
            min_val=config.min_val,
            max_val=config.max_val,
            size=size,
        )
    elif dist_type == "beta":
        return sample_beta(
            rng,
            alpha=config.alpha,
            beta=config.beta,
            min_val=config.min_val,
            max_val=config.max_val,
            size=size,
        )
    elif dist_type == "uniform":
        return sample_uniform(
            rng,
            min_val=config.min_val,
            max_val=config.max_val,
            size=size,
        )
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def sample_categorical(
    rng: Generator,
    probabilities: list[float] | np.ndarray,
    size: int | None = None,
) -> int | np.ndarray:
    """Sample from a categorical distribution.

    Args:
        rng: NumPy random generator
        probabilities: Probability for each category (will be normalized)
        size: Number of samples

    Returns:
        Category index/indices
    """
    probs = np.array(probabilities)
    probs = probs / probs.sum()  # Normalize

    return rng.choice(len(probs), size=size, p=probs)


def sample_weighted_choice(
    rng: Generator,
    items: list[Any],
    weights: list[float] | np.ndarray,
    size: int | None = None,
    replace: bool = True,
) -> Any | list[Any]:
    """Sample items with weights.

    Args:
        rng: NumPy random generator
        items: Items to choose from
        weights: Weight for each item
        size: Number of items to select
        replace: Whether to sample with replacement

    Returns:
        Selected item(s)
    """
    weights = np.array(weights)
    probs = weights / weights.sum()

    indices = rng.choice(len(items), size=size, p=probs, replace=replace)

    if size is None:
        return items[indices]
    return [items[i] for i in indices]


def create_bimodal_samples(
    rng: Generator,
    n_samples: int,
    mode1_mean: float = -0.5,
    mode2_mean: float = 0.5,
    std: float = 0.2,
    mode1_weight: float = 0.5,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """Create samples from a bimodal distribution.

    Useful for creating polarized ideology distributions.

    Args:
        rng: NumPy random generator
        n_samples: Number of samples
        mode1_mean: Mean of first mode
        mode2_mean: Mean of second mode
        std: Standard deviation for both modes
        mode1_weight: Weight for first mode (0-1)
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Array of samples
    """
    # Determine which mode each sample comes from
    mode_choice = rng.random(n_samples) < mode1_weight

    samples = np.empty(n_samples)
    n_mode1 = mode_choice.sum()
    n_mode2 = n_samples - n_mode1

    samples[mode_choice] = rng.normal(mode1_mean, std, n_mode1)
    samples[~mode_choice] = rng.normal(mode2_mean, std, n_mode2)

    return np.clip(samples, min_val, max_val)
