"""Experiment runner for what-if scenarios."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import copy
import json

from loguru import logger
import pandas as pd

from config.schemas import SimulationConfig
from generator import World
from engine import Simulation
from engine.metrics import StepMetrics


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Attributes:
        name: Experiment name
        description: What this experiment tests
        base_config: Base simulation configuration
        variations: Dict mapping variation name to config modifications
        num_runs: Number of runs per variation (for statistical significance)
        metrics_to_track: Specific metrics to compare
    """

    name: str
    description: str = ""
    base_config: SimulationConfig = field(default_factory=SimulationConfig)
    variations: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_runs: int = 1
    metrics_to_track: list[str] = field(default_factory=lambda: [
        "total_posts",
        "total_interactions",
        "total_shares",
        "engagement_rate",
        "misinfo_share_rate",
        "active_cascades",
    ])


@dataclass
class VariationResult:
    """Results from a single variation run."""

    variation_name: str
    run_index: int
    final_metrics: dict[str, Any]
    step_metrics: pd.DataFrame
    summary_metrics: dict[str, Any]


@dataclass
class ExperimentResult:
    """Complete results from an experiment."""

    experiment_name: str
    description: str
    variation_results: dict[str, list[VariationResult]]
    comparison_summary: pd.DataFrame | None = None

    def get_variation_means(self, metric: str) -> dict[str, float]:
        """Get mean of a metric across runs for each variation."""
        means = {}
        for var_name, results in self.variation_results.items():
            values = [r.summary_metrics.get(metric, 0) for r in results]
            means[var_name] = sum(values) / len(values) if values else 0
        return means

    def get_variation_stds(self, metric: str) -> dict[str, float]:
        """Get std of a metric across runs for each variation."""
        import numpy as np
        stds = {}
        for var_name, results in self.variation_results.items():
            values = [r.summary_metrics.get(metric, 0) for r in results]
            stds[var_name] = float(np.std(values)) if values else 0
        return stds


class Experiment:
    """Runs experiments with multiple variations."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: ExperimentResult | None = None
        self.progress_callback: Callable[[str, int, int], None] | None = None

    def run(self, share_world: bool = True) -> ExperimentResult:
        """Run the experiment.

        Args:
            share_world: If True, all variations use the same initial world

        Returns:
            ExperimentResult with all variation results
        """
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Variations: {list(self.config.variations.keys())}")
        logger.info(f"Runs per variation: {self.config.num_runs}")

        # Build shared world if requested
        shared_world = None
        if share_world:
            logger.info("Building shared world...")
            shared_world = World(self.config.base_config)
            shared_world.build()

        variation_results: dict[str, list[VariationResult]] = {}

        total_runs = len(self.config.variations) * self.config.num_runs
        current_run = 0

        for var_name, var_mods in self.config.variations.items():
            logger.info(f"Running variation: {var_name}")
            variation_results[var_name] = []

            for run_idx in range(self.config.num_runs):
                current_run += 1

                if self.progress_callback:
                    self.progress_callback(var_name, run_idx, total_runs)

                # Create modified config
                var_config = self._apply_modifications(
                    self.config.base_config,
                    var_mods,
                )

                # Modify seed for different runs
                if var_config.seed is not None:
                    var_config.seed = var_config.seed + run_idx * 1000

                # Run simulation
                result = self._run_variation(
                    var_name,
                    run_idx,
                    var_config,
                    shared_world,
                )
                variation_results[var_name].append(result)

                logger.info(
                    f"  Run {run_idx + 1}/{self.config.num_runs} complete - "
                    f"engagement: {result.summary_metrics.get('total_interactions', 0)}"
                )

        # Create comparison summary
        comparison_df = self._create_comparison_summary(variation_results)

        self.results = ExperimentResult(
            experiment_name=self.config.name,
            description=self.config.description,
            variation_results=variation_results,
            comparison_summary=comparison_df,
        )

        logger.info(f"Experiment complete: {self.config.name}")
        return self.results

    def _apply_modifications(
        self,
        base_config: SimulationConfig,
        modifications: dict[str, Any],
    ) -> SimulationConfig:
        """Apply modifications to base config.

        Args:
            base_config: Base configuration
            modifications: Dict of modifications (can be nested)

        Returns:
            Modified configuration
        """
        config_dict = base_config.model_dump()
        self._deep_update(config_dict, modifications)
        return SimulationConfig(**config_dict)

    def _deep_update(self, base: dict, updates: dict) -> None:
        """Deep update a dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _run_variation(
        self,
        var_name: str,
        run_idx: int,
        config: SimulationConfig,
        shared_world: World | None,
    ) -> VariationResult:
        """Run a single variation.

        Args:
            var_name: Variation name
            run_idx: Run index
            config: Variation configuration
            shared_world: Shared world (or None to build new)

        Returns:
            VariationResult
        """
        # Create simulation
        sim = Simulation(config)

        # Use shared world or build new
        if shared_world:
            # Create a fresh copy of the world state
            world_copy = self._copy_world(shared_world, config)
            sim.initialize(world_copy)
        else:
            sim.initialize()

        # Run simulation
        results = sim.run(show_progress=False)

        # Get metrics DataFrame
        metrics_df = sim.get_metrics_dataframe()

        return VariationResult(
            variation_name=var_name,
            run_index=run_idx,
            final_metrics=sim.state.get_summary_statistics(),
            step_metrics=metrics_df,
            summary_metrics=results.get("metrics_summary", {}),
        )

    def _copy_world(self, world: World, config: SimulationConfig) -> World:
        """Create a copy of a world with new config.

        Args:
            world: World to copy
            config: New configuration

        Returns:
            Copied world
        """
        new_world = World(config)
        new_world.topics = copy.deepcopy(world.topics)
        new_world.users = copy.deepcopy(world.users)
        new_world.graph = world.graph.copy()

        # Reinitialize generators with copied data
        new_world.topic_generator = world.topic_generator
        new_world.user_generator = world.user_generator
        new_world.network_generator = world.network_generator
        new_world.content_generator = world.content_generator
        new_world._built = True

        return new_world

    def _create_comparison_summary(
        self,
        variation_results: dict[str, list[VariationResult]],
    ) -> pd.DataFrame:
        """Create summary comparison DataFrame.

        Args:
            variation_results: Results by variation

        Returns:
            Comparison DataFrame
        """
        import numpy as np

        records = []

        for var_name, results in variation_results.items():
            record = {"variation": var_name}

            for metric in self.config.metrics_to_track:
                values = [r.summary_metrics.get(metric, 0) for r in results]
                if values:
                    record[f"{metric}_mean"] = np.mean(values)
                    record[f"{metric}_std"] = np.std(values)
                else:
                    record[f"{metric}_mean"] = 0
                    record[f"{metric}_std"] = 0

            records.append(record)

        return pd.DataFrame(records)

    def save_results(self, path: str | Path) -> None:
        """Save experiment results.

        Args:
            path: Directory to save to
        """
        if not self.results:
            raise ValueError("No results to save - run experiment first")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save summary
        if self.results.comparison_summary is not None:
            self.results.comparison_summary.to_csv(path / "comparison_summary.csv", index=False)

        # Save detailed results per variation
        for var_name, results in self.results.variation_results.items():
            var_path = path / var_name
            var_path.mkdir(exist_ok=True)

            for i, result in enumerate(results):
                result.step_metrics.to_parquet(var_path / f"run_{i}_metrics.parquet")

                with open(var_path / f"run_{i}_summary.json", "w") as f:
                    json.dump({
                        "final_metrics": result.final_metrics,
                        "summary_metrics": result.summary_metrics,
                    }, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")

    def set_progress_callback(
        self,
        callback: Callable[[str, int, int], None],
    ) -> None:
        """Set progress callback.

        Args:
            callback: Function(variation_name, run_index, total_runs)
        """
        self.progress_callback = callback
