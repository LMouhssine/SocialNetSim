"""Tests for distribution utilities."""

import numpy as np
import pytest

from generator.distributions import (
    sample_normal,
    sample_beta,
    sample_uniform,
    sample_power_law,
    sample_from_config,
    create_bimodal_samples,
)
from config.schemas import DistributionConfig


class TestSampleNormal:
    """Tests for sample_normal function."""

    def test_returns_value_in_range(self, rng):
        """Test normal samples are clipped to range."""
        samples = sample_normal(rng, mean=0.5, std=0.1, min_val=0.0, max_val=1.0, size=1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_mean_is_approximately_correct(self, rng):
        """Test samples have approximately correct mean."""
        samples = sample_normal(rng, mean=0.5, std=0.1, size=10000)
        assert abs(np.mean(samples) - 0.5) < 0.05

    def test_reproducibility_with_seed(self):
        """Test same seed produces same samples."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples1 = sample_normal(rng1, mean=0.0, std=1.0, size=100)
        samples2 = sample_normal(rng2, mean=0.0, std=1.0, size=100)

        np.testing.assert_array_equal(samples1, samples2)


class TestSampleBeta:
    """Tests for sample_beta function."""

    def test_returns_value_in_range(self, rng):
        """Test beta samples are in specified range."""
        samples = sample_beta(rng, alpha=2.0, beta=5.0, min_val=0.0, max_val=1.0, size=1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_scaled_range(self, rng):
        """Test beta samples scale to custom range."""
        samples = sample_beta(rng, alpha=2.0, beta=2.0, min_val=-1.0, max_val=1.0, size=1000)
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)


class TestSampleUniform:
    """Tests for sample_uniform function."""

    def test_returns_value_in_range(self, rng):
        """Test uniform samples are in range."""
        samples = sample_uniform(rng, min_val=5.0, max_val=10.0, size=1000)
        assert np.all(samples >= 5.0)
        assert np.all(samples <= 10.0)


class TestSamplePowerLaw:
    """Tests for sample_power_law function."""

    def test_returns_values_above_min(self, rng):
        """Test power law samples are above minimum."""
        samples = sample_power_law(rng, alpha=2.0, min_val=1.0, size=1000)
        assert np.all(samples >= 1.0)

    def test_respects_max_value(self, rng):
        """Test power law samples are capped at max."""
        samples = sample_power_law(rng, alpha=2.0, min_val=1.0, max_val=5.0, size=1000)
        assert np.all(samples <= 5.0)

    def test_invalid_alpha_raises(self, rng):
        """Test alpha <= 1 raises error."""
        with pytest.raises(ValueError):
            sample_power_law(rng, alpha=1.0, min_val=1.0)


class TestSampleFromConfig:
    """Tests for sample_from_config function."""

    def test_normal_config(self, rng):
        """Test sampling from normal config."""
        config = DistributionConfig(type="normal", mean=0.5, std=0.1)
        samples = sample_from_config(rng, config, size=100)
        assert len(samples) == 100

    def test_beta_config(self, rng):
        """Test sampling from beta config."""
        config = DistributionConfig(type="beta", alpha=2.0, beta=5.0)
        samples = sample_from_config(rng, config, size=100)
        assert len(samples) == 100
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_uniform_config(self, rng):
        """Test sampling from uniform config."""
        config = DistributionConfig(type="uniform", min_val=0.0, max_val=1.0)
        samples = sample_from_config(rng, config, size=100)
        assert len(samples) == 100

    def test_unknown_type_raises(self, rng):
        """Test unknown distribution type raises error."""
        config = DistributionConfig(type="unknown")
        with pytest.raises(ValueError):
            sample_from_config(rng, config)


class TestCreateBimodalSamples:
    """Tests for create_bimodal_samples function."""

    def test_correct_size(self, rng):
        """Test correct number of samples."""
        samples = create_bimodal_samples(rng, n_samples=500)
        assert len(samples) == 500

    def test_values_in_range(self, rng):
        """Test samples are in specified range."""
        samples = create_bimodal_samples(rng, n_samples=1000, min_val=-1.0, max_val=1.0)
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)
