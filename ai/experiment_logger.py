"""Experiment tracking and logging for ML experiments.

Provides:
- Configuration logging
- Metric tracking over time
- Artifact storage
- Experiment comparison
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime
import json
import hashlib
import shutil


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking.

    Attributes:
        base_dir: Base directory for experiment storage
        experiment_name: Name of the experiment
        log_metrics_to_file: Whether to log metrics to file
        log_artifacts: Whether to store artifacts
        auto_save_interval: Auto-save metrics every N updates
    """

    base_dir: str = "experiments"
    experiment_name: str = "default"
    log_metrics_to_file: bool = True
    log_artifacts: bool = True
    auto_save_interval: int = 100


@dataclass
class MetricRecord:
    """A single metric record.

    Attributes:
        name: Metric name
        value: Metric value
        step: Training/evaluation step
        timestamp: When metric was recorded
        tags: Additional tags
    """

    name: str
    value: float
    step: int
    timestamp: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """A single experiment run.

    Attributes:
        run_id: Unique run identifier
        experiment_name: Parent experiment name
        config: Configuration used for this run
        metrics: Recorded metrics
        artifacts: Stored artifact paths
        start_time: When run started
        end_time: When run ended
        status: Run status (running, completed, failed)
        notes: User notes
    """

    run_id: str
    experiment_name: str
    config: dict[str, Any]
    metrics: list[MetricRecord] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    notes: str = ""


class ExperimentLogger:
    """Tracks ML experiments with configs, metrics, and artifacts."""

    def __init__(self, config: ExperimentConfig | None = None):
        """Initialize experiment logger.

        Args:
            config: Logger configuration
        """
        self.config = config or ExperimentConfig()

        self.base_dir = Path(self.config.base_dir)
        self.experiment_dir = self.base_dir / self.config.experiment_name

        self.current_run: ExperimentRun | None = None
        self._metric_buffer: list[MetricRecord] = []
        self._update_count = 0

    def start_run(
        self,
        run_config: dict[str, Any],
        run_name: str | None = None,
        notes: str = "",
    ) -> str:
        """Start a new experiment run.

        Args:
            run_config: Configuration for this run
            run_name: Optional run name (auto-generated if None)
            notes: User notes

        Returns:
            Run ID
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(run_config, sort_keys=True).encode()
        ).hexdigest()[:8]

        run_id = run_name or f"run_{timestamp}_{config_hash}"

        # Create run
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.config.experiment_name,
            config=run_config,
            start_time=datetime.now().isoformat(),
            notes=notes,
        )

        # Create directories
        run_dir = self.experiment_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        # Save config
        self._save_config(run_config, run_dir)

        return run_id

    def _save_config(self, config: dict[str, Any], run_dir: Path) -> None:
        """Save configuration to file.

        Args:
            config: Configuration dictionary
            run_dir: Run directory
        """
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
            tags: Optional tags
        """
        if self.current_run is None:
            raise RuntimeError("No active run - call start_run first")

        record = MetricRecord(
            name=name,
            value=value,
            step=step or self._update_count,
            timestamp=datetime.now().isoformat(),
            tags=tags or {},
        )

        self.current_run.metrics.append(record)
        self._metric_buffer.append(record)
        self._update_count += 1

        # Auto-save if needed
        if len(self._metric_buffer) >= self.config.auto_save_interval:
            self._flush_metrics()

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
            tags: Optional tags for all metrics
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, tags)

    def log_artifact(
        self,
        name: str,
        artifact_path: str | Path,
        copy: bool = True,
    ) -> str:
        """Log an artifact file.

        Args:
            name: Artifact name
            artifact_path: Path to artifact file
            copy: Whether to copy the file (vs just reference)

        Returns:
            Path to stored artifact
        """
        if self.current_run is None:
            raise RuntimeError("No active run - call start_run first")

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        run_dir = self.experiment_dir / self.current_run.run_id
        dest_dir = run_dir / "artifacts"

        if copy:
            dest_path = dest_dir / artifact_path.name
            shutil.copy2(artifact_path, dest_path)
            stored_path = str(dest_path)
        else:
            stored_path = str(artifact_path.absolute())

        self.current_run.artifacts[name] = stored_path
        return stored_path

    def log_artifact_data(
        self,
        name: str,
        data: Any,
        format: str = "json",
    ) -> str:
        """Log data directly as an artifact.

        Args:
            name: Artifact name
            data: Data to store
            format: Storage format ('json' or 'pickle')

        Returns:
            Path to stored artifact
        """
        if self.current_run is None:
            raise RuntimeError("No active run - call start_run first")

        run_dir = self.experiment_dir / self.current_run.run_id
        dest_dir = run_dir / "artifacts"

        if format == "json":
            file_path = dest_dir / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "pickle":
            import pickle
            file_path = dest_dir / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.current_run.artifacts[name] = str(file_path)
        return str(file_path)

    def end_run(self, status: str = "completed") -> None:
        """End the current run.

        Args:
            status: Final run status
        """
        if self.current_run is None:
            return

        self.current_run.end_time = datetime.now().isoformat()
        self.current_run.status = status

        # Flush remaining metrics
        self._flush_metrics()

        # Save run metadata
        run_dir = self.experiment_dir / self.current_run.run_id
        metadata_path = run_dir / "run_metadata.json"

        metadata = {
            "run_id": self.current_run.run_id,
            "experiment_name": self.current_run.experiment_name,
            "start_time": self.current_run.start_time,
            "end_time": self.current_run.end_time,
            "status": self.current_run.status,
            "notes": self.current_run.notes,
            "artifacts": self.current_run.artifacts,
            "metric_count": len(self.current_run.metrics),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.current_run = None
        self._update_count = 0

    def _flush_metrics(self) -> None:
        """Flush buffered metrics to file."""
        if not self._metric_buffer or self.current_run is None:
            return

        if not self.config.log_metrics_to_file:
            self._metric_buffer.clear()
            return

        run_dir = self.experiment_dir / self.current_run.run_id
        metrics_path = run_dir / "metrics.jsonl"

        with open(metrics_path, "a") as f:
            for record in self._metric_buffer:
                line = json.dumps({
                    "name": record.name,
                    "value": record.value,
                    "step": record.step,
                    "timestamp": record.timestamp,
                    "tags": record.tags,
                })
                f.write(line + "\n")

        self._metric_buffer.clear()

    def get_metrics(
        self,
        name: str | None = None,
        step_range: tuple[int, int] | None = None,
    ) -> list[MetricRecord]:
        """Get metrics from current run.

        Args:
            name: Filter by metric name
            step_range: Filter by step range (start, end)

        Returns:
            List of matching metric records
        """
        if self.current_run is None:
            return []

        metrics = self.current_run.metrics

        if name:
            metrics = [m for m in metrics if m.name == name]

        if step_range:
            start, end = step_range
            metrics = [m for m in metrics if start <= m.step <= end]

        return metrics

    def get_metric_summary(self, name: str) -> dict[str, float]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with min, max, mean, last values
        """
        metrics = self.get_metrics(name)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1],
            "count": len(values),
        }


class ExperimentComparator:
    """Compare metrics across experiment runs."""

    def __init__(self, base_dir: str = "experiments"):
        """Initialize comparator.

        Args:
            base_dir: Base experiments directory
        """
        self.base_dir = Path(base_dir)

    def list_experiments(self) -> list[str]:
        """List all experiments.

        Returns:
            List of experiment names
        """
        if not self.base_dir.exists():
            return []

        return [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_runs(self, experiment_name: str) -> list[dict[str, Any]]:
        """List all runs for an experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            List of run metadata dictionaries
        """
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            return []

        runs = []
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    runs.append(json.load(f))

        return sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)

    def load_run_metrics(
        self,
        experiment_name: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Load all metrics for a run.

        Args:
            experiment_name: Experiment name
            run_id: Run identifier

        Returns:
            List of metric records
        """
        metrics_path = self.base_dir / experiment_name / run_id / "metrics.jsonl"
        if not metrics_path.exists():
            return []

        metrics = []
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        return metrics

    def load_run_config(
        self,
        experiment_name: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Load configuration for a run.

        Args:
            experiment_name: Experiment name
            run_id: Run identifier

        Returns:
            Configuration dictionary
        """
        config_path = self.base_dir / experiment_name / run_id / "config.json"
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            return json.load(f)

    def compare_runs(
        self,
        experiment_name: str,
        run_ids: list[str],
        metric_names: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compare specific metrics across runs.

        Args:
            experiment_name: Experiment name
            run_ids: List of run IDs to compare
            metric_names: List of metric names to compare

        Returns:
            Nested dict: run_id -> metric_name -> summary stats
        """
        results = {}

        for run_id in run_ids:
            metrics = self.load_run_metrics(experiment_name, run_id)
            results[run_id] = {}

            for metric_name in metric_names:
                matching = [m for m in metrics if m["name"] == metric_name]

                if matching:
                    values = [m["value"] for m in matching]
                    results[run_id][metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "last": values[-1],
                    }
                else:
                    results[run_id][metric_name] = {}

        return results

    def find_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        mode: str = "max",
    ) -> tuple[str | None, float | None]:
        """Find the best run for a given metric.

        Args:
            experiment_name: Experiment name
            metric_name: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Tuple of (run_id, best_value)
        """
        runs = self.list_runs(experiment_name)
        if not runs:
            return None, None

        best_run = None
        best_value = None

        for run in runs:
            run_id = run["run_id"]
            metrics = self.load_run_metrics(experiment_name, run_id)

            matching = [m for m in metrics if m["name"] == metric_name]
            if not matching:
                continue

            # Get final value
            final_value = matching[-1]["value"]

            if best_value is None:
                best_value = final_value
                best_run = run_id
            elif mode == "max" and final_value > best_value:
                best_value = final_value
                best_run = run_id
            elif mode == "min" and final_value < best_value:
                best_value = final_value
                best_run = run_id

        return best_run, best_value


class MetricTracker:
    """Simple metric tracker for training loops."""

    def __init__(self):
        """Initialize tracker."""
        self.metrics: dict[str, list[float]] = {}
        self.steps: dict[str, list[int]] = {}
        self._current_step = 0

    def update(self, name: str, value: float, step: int | None = None) -> None:
        """Update a metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.metrics:
            self.metrics[name] = []
            self.steps[name] = []

        self.metrics[name].append(value)
        self.steps[name].append(step or self._current_step)
        self._current_step += 1

    def update_many(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Update multiple metrics.

        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.update(name, value, step)

    def get(self, name: str) -> list[float]:
        """Get all values for a metric.

        Args:
            name: Metric name

        Returns:
            List of values
        """
        return self.metrics.get(name, [])

    def get_last(self, name: str) -> float | None:
        """Get last value for a metric.

        Args:
            name: Metric name

        Returns:
            Last value or None
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_best(self, name: str, mode: str = "max") -> tuple[float | None, int | None]:
        """Get best value and step for a metric.

        Args:
            name: Metric name
            mode: 'max' or 'min'

        Returns:
            Tuple of (best_value, best_step)
        """
        values = self.metrics.get(name, [])
        steps = self.steps.get(name, [])

        if not values:
            return None, None

        if mode == "max":
            idx = max(range(len(values)), key=lambda i: values[i])
        else:
            idx = min(range(len(values)), key=lambda i: values[i])

        return values[idx], steps[idx]

    def get_summary(self, name: str) -> dict[str, float]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Summary statistics
        """
        values = self.metrics.get(name, [])

        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1],
            "count": len(values),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.steps.clear()
        self._current_step = 0
