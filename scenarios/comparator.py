"""Experiment comparison utilities."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .experiment import ExperimentResult


class ExperimentComparator:
    """Compares results across experiments and variations."""

    def __init__(self):
        """Initialize comparator."""
        pass

    def compare_variations(
        self,
        result: ExperimentResult,
        baseline_variation: str | None = None,
    ) -> pd.DataFrame:
        """Compare variations within an experiment.

        Args:
            result: Experiment result
            baseline_variation: Variation to use as baseline (first if None)

        Returns:
            Comparison DataFrame with relative changes
        """
        if not result.variation_results:
            return pd.DataFrame()

        # Determine baseline
        if baseline_variation is None:
            baseline_variation = list(result.variation_results.keys())[0]

        if baseline_variation not in result.variation_results:
            raise ValueError(f"Baseline variation not found: {baseline_variation}")

        baseline_results = result.variation_results[baseline_variation]

        # Get all metrics from first result
        sample_metrics = baseline_results[0].summary_metrics
        metric_names = [k for k in sample_metrics.keys() if isinstance(sample_metrics[k], (int, float))]

        records = []

        for var_name, var_results in result.variation_results.items():
            record = {"variation": var_name, "is_baseline": var_name == baseline_variation}

            for metric in metric_names:
                # Get values for this variation
                var_values = [r.summary_metrics.get(metric, 0) for r in var_results]
                base_values = [r.summary_metrics.get(metric, 0) for r in baseline_results]

                var_mean = np.mean(var_values) if var_values else 0
                base_mean = np.mean(base_values) if base_values else 0

                record[f"{metric}_value"] = var_mean
                record[f"{metric}_vs_baseline"] = (
                    (var_mean - base_mean) / base_mean * 100
                    if base_mean != 0 else 0
                )

                # Statistical significance (if multiple runs)
                if len(var_values) > 1 and len(base_values) > 1:
                    try:
                        _, p_value = stats.ttest_ind(var_values, base_values)
                        record[f"{metric}_p_value"] = p_value
                        record[f"{metric}_significant"] = p_value < 0.05
                    except Exception:
                        record[f"{metric}_p_value"] = None
                        record[f"{metric}_significant"] = None

            records.append(record)

        return pd.DataFrame(records)

    def compare_experiments(
        self,
        experiments: dict[str, ExperimentResult],
        metric: str,
    ) -> pd.DataFrame:
        """Compare a metric across multiple experiments.

        Args:
            experiments: Dict mapping experiment name to result
            metric: Metric to compare

        Returns:
            Comparison DataFrame
        """
        records = []

        for exp_name, result in experiments.items():
            for var_name, var_results in result.variation_results.items():
                values = [r.summary_metrics.get(metric, 0) for r in var_results]

                records.append({
                    "experiment": exp_name,
                    "variation": var_name,
                    "mean": np.mean(values) if values else 0,
                    "std": np.std(values) if values else 0,
                    "min": np.min(values) if values else 0,
                    "max": np.max(values) if values else 0,
                    "n_runs": len(values),
                })

        return pd.DataFrame(records)

    def get_time_series_comparison(
        self,
        result: ExperimentResult,
        metric: str,
    ) -> pd.DataFrame:
        """Get time series of a metric for all variations.

        Args:
            result: Experiment result
            metric: Metric to extract

        Returns:
            DataFrame with step as index and variations as columns
        """
        data = {}

        for var_name, var_results in result.variation_results.items():
            # Average across runs if multiple
            all_series = []
            for var_result in var_results:
                if metric in var_result.step_metrics.columns:
                    all_series.append(var_result.step_metrics[metric].values)

            if all_series:
                # Align series lengths
                min_len = min(len(s) for s in all_series)
                aligned = [s[:min_len] for s in all_series]
                data[var_name] = np.mean(aligned, axis=0)

        if not data:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(data)
        df.index.name = "step"
        return df

    def calculate_effect_sizes(
        self,
        result: ExperimentResult,
        baseline_variation: str | None = None,
    ) -> pd.DataFrame:
        """Calculate Cohen's d effect sizes vs baseline.

        Args:
            result: Experiment result
            baseline_variation: Baseline variation name

        Returns:
            DataFrame with effect sizes
        """
        if not result.variation_results:
            return pd.DataFrame()

        if baseline_variation is None:
            baseline_variation = list(result.variation_results.keys())[0]

        baseline_results = result.variation_results[baseline_variation]
        sample_metrics = baseline_results[0].summary_metrics
        metric_names = [k for k in sample_metrics.keys() if isinstance(sample_metrics[k], (int, float))]

        records = []

        for var_name, var_results in result.variation_results.items():
            if var_name == baseline_variation:
                continue

            record = {"variation": var_name}

            for metric in metric_names:
                var_values = np.array([r.summary_metrics.get(metric, 0) for r in var_results])
                base_values = np.array([r.summary_metrics.get(metric, 0) for r in baseline_results])

                # Cohen's d
                pooled_std = np.sqrt(
                    (np.var(var_values) + np.var(base_values)) / 2
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(var_values) - np.mean(base_values)) / pooled_std
                else:
                    cohens_d = 0

                record[f"{metric}_cohens_d"] = cohens_d

                # Effect size interpretation
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    effect = "negligible"
                elif abs_d < 0.5:
                    effect = "small"
                elif abs_d < 0.8:
                    effect = "medium"
                else:
                    effect = "large"
                record[f"{metric}_effect"] = effect

            records.append(record)

        return pd.DataFrame(records)

    def generate_summary_report(
        self,
        result: ExperimentResult,
        baseline_variation: str | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive comparison report.

        Args:
            result: Experiment result
            baseline_variation: Baseline variation

        Returns:
            Report dictionary
        """
        comparison = self.compare_variations(result, baseline_variation)
        effect_sizes = self.calculate_effect_sizes(result, baseline_variation)

        # Determine winners for key metrics
        key_metrics = ["total_interactions", "engagement_rate", "misinfo_share_rate"]
        winners = {}

        for metric in key_metrics:
            col = f"{metric}_value"
            if col in comparison.columns:
                if metric == "misinfo_share_rate":
                    # Lower is better
                    best_idx = comparison[col].idxmin()
                else:
                    # Higher is better
                    best_idx = comparison[col].idxmax()
                winners[metric] = comparison.loc[best_idx, "variation"]

        report = {
            "experiment_name": result.experiment_name,
            "description": result.description,
            "num_variations": len(result.variation_results),
            "baseline_variation": baseline_variation or list(result.variation_results.keys())[0],
            "winners_by_metric": winners,
            "comparison_table": comparison.to_dict(orient="records"),
            "effect_sizes": effect_sizes.to_dict(orient="records") if not effect_sizes.empty else [],
        }

        # Add key findings
        findings = []
        if not comparison.empty:
            for _, row in comparison.iterrows():
                if row["is_baseline"]:
                    continue

                var = row["variation"]

                # Check engagement change
                eng_change = row.get("total_interactions_vs_baseline", 0)
                if abs(eng_change) > 10:
                    direction = "increased" if eng_change > 0 else "decreased"
                    findings.append(
                        f"{var}: Engagement {direction} by {abs(eng_change):.1f}%"
                    )

                # Check misinfo change
                misinfo_change = row.get("misinfo_share_rate_vs_baseline", 0)
                if abs(misinfo_change) > 10:
                    direction = "increased" if misinfo_change > 0 else "decreased"
                    findings.append(
                        f"{var}: Misinformation sharing {direction} by {abs(misinfo_change):.1f}%"
                    )

        report["key_findings"] = findings

        return report
