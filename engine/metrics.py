"""Real-time metrics collection and computation."""

from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from models import User, Post, Interaction, Cascade
from models.enums import InteractionType
from .state import SimulationState


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""

    step: int
    total_posts: int = 0
    new_posts: int = 0
    total_interactions: int = 0
    new_interactions: int = 0
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    active_users: int = 0
    new_cascades: int = 0
    active_cascades: int = 0
    total_cascade_reach: int = 0
    misinfo_posts: int = 0
    misinfo_shares: int = 0
    moderation_actions: int = 0
    active_events: int = 0
    additional_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "total_posts": self.total_posts,
            "new_posts": self.new_posts,
            "total_interactions": self.total_interactions,
            "new_interactions": self.new_interactions,
            "views": self.views,
            "likes": self.likes,
            "shares": self.shares,
            "comments": self.comments,
            "active_users": self.active_users,
            "new_cascades": self.new_cascades,
            "active_cascades": self.active_cascades,
            "total_cascade_reach": self.total_cascade_reach,
            "misinfo_posts": self.misinfo_posts,
            "misinfo_shares": self.misinfo_shares,
            "moderation_actions": self.moderation_actions,
            "active_events": self.active_events,
            "additional_metrics": self.additional_metrics,
        }


class MetricsCollector:
    """Collects and computes simulation metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.step_metrics: list[StepMetrics] = []
        self.previous_totals: dict[str, int] = defaultdict(int)

    def collect_step_metrics(
        self,
        state: SimulationState,
        new_interactions: list[Interaction],
        new_posts: list[Post],
        new_cascades: list[Cascade],
        moderation_actions: int = 0,
    ) -> StepMetrics:
        """Collect metrics for current step.

        Args:
            state: Current simulation state
            new_interactions: Interactions created this step
            new_posts: Posts created this step
            new_cascades: Cascades created this step
            moderation_actions: Number of moderation actions this step

        Returns:
            StepMetrics for current step
        """
        metrics = StepMetrics(step=state.current_step)

        # Post metrics
        metrics.total_posts = len(state.posts)
        metrics.new_posts = len(new_posts)
        metrics.misinfo_posts = sum(
            1 for p in new_posts if p.content.is_misinformation
        )

        # Interaction metrics
        metrics.total_interactions = len(state.interactions)
        metrics.new_interactions = len(new_interactions)

        # Count by type
        for interaction in new_interactions:
            if interaction.interaction_type == InteractionType.VIEW:
                metrics.views += 1
            elif interaction.interaction_type == InteractionType.LIKE:
                metrics.likes += 1
            elif interaction.interaction_type == InteractionType.SHARE:
                metrics.shares += 1

                # Track misinfo shares
                post = state.get_post(interaction.post_id)
                if post and post.content.is_misinformation:
                    metrics.misinfo_shares += 1

            elif interaction.interaction_type == InteractionType.COMMENT:
                metrics.comments += 1

        # User metrics
        metrics.active_users = len(state.get_active_users(lookback_steps=1))

        # Cascade metrics
        metrics.new_cascades = len(new_cascades)
        metrics.active_cascades = len(state.get_active_cascades())
        metrics.total_cascade_reach = sum(
            c.total_reach for c in state.cascades.values()
        )

        # Moderation metrics
        metrics.moderation_actions = moderation_actions

        # Event metrics
        metrics.active_events = len(state.active_events)

        # Store and return
        self.step_metrics.append(metrics)
        state.record_step_metrics(metrics.to_dict())

        return metrics

    def get_summary_metrics(self) -> dict[str, Any]:
        """Get summary metrics across all steps.

        Returns:
            Dictionary of summary metrics
        """
        if not self.step_metrics:
            return {}

        total_posts = sum(m.new_posts for m in self.step_metrics)
        total_interactions = sum(m.new_interactions for m in self.step_metrics)
        total_views = sum(m.views for m in self.step_metrics)
        total_likes = sum(m.likes for m in self.step_metrics)
        total_shares = sum(m.shares for m in self.step_metrics)
        total_comments = sum(m.comments for m in self.step_metrics)
        total_misinfo_posts = sum(m.misinfo_posts for m in self.step_metrics)
        total_misinfo_shares = sum(m.misinfo_shares for m in self.step_metrics)

        # Averages
        n_steps = len(self.step_metrics)
        avg_posts_per_step = total_posts / n_steps
        avg_interactions_per_step = total_interactions / n_steps
        avg_active_users = np.mean([m.active_users for m in self.step_metrics])

        # Peak values
        peak_interactions = max(m.new_interactions for m in self.step_metrics)
        peak_active_users = max(m.active_users for m in self.step_metrics)
        peak_cascades = max(m.active_cascades for m in self.step_metrics)

        # Engagement rates
        engagement_rate = (total_likes + total_shares + total_comments) / max(1, total_views)
        share_rate = total_shares / max(1, total_views)
        misinfo_share_rate = total_misinfo_shares / max(1, total_shares) if total_shares > 0 else 0

        return {
            "total_steps": n_steps,
            "total_posts": total_posts,
            "total_interactions": total_interactions,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_shares": total_shares,
            "total_comments": total_comments,
            "total_misinfo_posts": total_misinfo_posts,
            "total_misinfo_shares": total_misinfo_shares,
            "avg_posts_per_step": avg_posts_per_step,
            "avg_interactions_per_step": avg_interactions_per_step,
            "avg_active_users": avg_active_users,
            "peak_interactions": peak_interactions,
            "peak_active_users": peak_active_users,
            "peak_cascades": peak_cascades,
            "engagement_rate": engagement_rate,
            "share_rate": share_rate,
            "misinfo_share_rate": misinfo_share_rate,
        }

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as a pandas DataFrame.

        Returns:
            DataFrame with step metrics
        """
        return pd.DataFrame([m.to_dict() for m in self.step_metrics])

    def get_engagement_over_time(self) -> dict[str, list[int]]:
        """Get engagement metrics over time.

        Returns:
            Dictionary with lists of values over time
        """
        return {
            "steps": [m.step for m in self.step_metrics],
            "views": [m.views for m in self.step_metrics],
            "likes": [m.likes for m in self.step_metrics],
            "shares": [m.shares for m in self.step_metrics],
            "comments": [m.comments for m in self.step_metrics],
        }

    def get_cascade_metrics_over_time(self) -> dict[str, list]:
        """Get cascade metrics over time.

        Returns:
            Dictionary with cascade metrics
        """
        return {
            "steps": [m.step for m in self.step_metrics],
            "new_cascades": [m.new_cascades for m in self.step_metrics],
            "active_cascades": [m.active_cascades for m in self.step_metrics],
            "total_reach": [m.total_cascade_reach for m in self.step_metrics],
        }

    def get_misinfo_metrics_over_time(self) -> dict[str, list]:
        """Get misinformation metrics over time.

        Returns:
            Dictionary with misinfo metrics
        """
        return {
            "steps": [m.step for m in self.step_metrics],
            "misinfo_posts": [m.misinfo_posts for m in self.step_metrics],
            "misinfo_shares": [m.misinfo_shares for m in self.step_metrics],
            "moderation_actions": [m.moderation_actions for m in self.step_metrics],
        }

    def compute_user_metrics(
        self,
        state: SimulationState,
        users: dict[str, User],
    ) -> dict[str, Any]:
        """Compute user-level metrics.

        Args:
            state: Simulation state
            users: Dictionary of users

        Returns:
            Dictionary of user metrics
        """
        # Gather user statistics
        activity_levels = []
        influence_scores = []
        credibility_scores = []
        interaction_counts = []
        post_counts = []

        for user in users.values():
            activity_levels.append(user.traits.activity_level)
            influence_scores.append(user.influence_score)
            credibility_scores.append(user.credibility_score)
            interaction_counts.append(user.total_interactions)
            post_counts.append(user.total_posts)

        return {
            "total_users": len(users),
            "avg_activity_level": float(np.mean(activity_levels)),
            "avg_influence_score": float(np.mean(influence_scores)),
            "avg_credibility_score": float(np.mean(credibility_scores)),
            "avg_interactions_per_user": float(np.mean(interaction_counts)),
            "avg_posts_per_user": float(np.mean(post_counts)),
            "max_influence_score": float(np.max(influence_scores)),
            "max_interactions": int(np.max(interaction_counts)),
            "max_posts": int(np.max(post_counts)),
        }

    def compute_network_polarization(
        self,
        state: SimulationState,
        users: dict[str, User],
    ) -> dict[str, float]:
        """Compute network polarization metrics.

        Args:
            state: Simulation state
            users: Dictionary of users

        Returns:
            Dictionary of polarization metrics
        """
        ideologies = [u.traits.ideology for u in users.values()]

        if not ideologies:
            return {"polarization_index": 0.0}

        # Variance as simple polarization measure
        ideology_variance = float(np.var(ideologies))

        # Bimodality coefficient (Sarle's)
        n = len(ideologies)
        skewness = float(np.abs(np.mean([(x - np.mean(ideologies))**3 for x in ideologies]) /
                         (np.std(ideologies)**3 + 1e-10)))
        kurtosis = float(np.mean([(x - np.mean(ideologies))**4 for x in ideologies]) /
                        (np.std(ideologies)**4 + 1e-10) - 3)

        # Simple bimodality coefficient
        bimodality = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)) + 1e-10)

        return {
            "ideology_variance": ideology_variance,
            "ideology_std": float(np.std(ideologies)),
            "ideology_mean": float(np.mean(ideologies)),
            "ideology_skewness": skewness,
            "bimodality_coefficient": float(bimodality),
        }

    def compute_content_metrics(
        self,
        state: SimulationState,
    ) -> dict[str, Any]:
        """Compute content-level metrics."""
        posts = list(state.posts.values())

        if not posts:
            return {}

        quality_scores = [p.content.quality_score for p in posts]
        controversy_scores = [p.content.controversy_score for p in posts]
        engagement_counts = [p.total_engagement for p in posts]
        virality_scores = [p.virality_score for p in posts]

        misinfo_count = sum(1 for p in posts if p.content.is_misinformation)
        misinfo_rate = misinfo_count / len(posts)

        return {
            "total_posts": len(posts),
            "avg_quality": float(np.mean(quality_scores)),
            "avg_controversy": float(np.mean(controversy_scores)),
            "avg_engagement": float(np.mean(engagement_counts)),
            "avg_virality": float(np.mean(virality_scores)),
            "max_engagement": int(np.max(engagement_counts)),
            "misinformation_rate": misinfo_rate,
            "misinfo_count": misinfo_count,
        }


class PolarizationMetrics:
    """Enhanced polarization metrics calculator.

    Computes:
    - Bimodality coefficient (Sarle's)
    - Echo chamber index
    - Disagreement exposure (filter bubble detection)
    - Opinion variance and cluster tracking
    """

    def __init__(self):
        """Initialize polarization metrics calculator."""
        self.history: list[dict[str, float]] = []

    def compute_bimodality_coefficient(
        self,
        values: list[float],
    ) -> float:
        """Compute Sarle's bimodality coefficient.

        BC > 0.555 suggests bimodal distribution.

        Args:
            values: List of values to analyze

        Returns:
            Bimodality coefficient
        """
        if len(values) < 4:
            return 0.0

        n = len(values)
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std < 1e-10:
            return 0.0

        # Third moment (skewness)
        m3 = np.mean((arr - mean) ** 3)
        g1 = m3 / (std ** 3)

        # Fourth moment (kurtosis)
        m4 = np.mean((arr - mean) ** 4)
        g2 = m4 / (std ** 4) - 3

        # Sarle's bimodality coefficient
        numerator = g1 ** 2 + 1
        denominator = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))

        if abs(denominator) < 1e-10:
            return 0.0

        return float(numerator / denominator)

    def compute_echo_chamber_index(
        self,
        users: dict[str, User],
        opinions: dict[str, float],
        threshold: float = 0.2,
    ) -> float:
        """Compute echo chamber index.

        Ratio of within-cluster edges to between-cluster edges.
        Higher values indicate stronger echo chambers.

        Args:
            users: Dictionary of users
            opinions: User opinions (user_id -> opinion)
            threshold: Opinion difference threshold for "same cluster"

        Returns:
            Echo chamber index
        """
        within_cluster = 0
        between_cluster = 0

        for user_id, user in users.items():
            if user_id not in opinions:
                continue

            user_opinion = opinions[user_id]

            for friend_id in user.following:
                if friend_id not in opinions:
                    continue

                friend_opinion = opinions[friend_id]
                diff = abs(user_opinion - friend_opinion)

                if diff < threshold:
                    within_cluster += 1
                else:
                    between_cluster += 1

        if between_cluster == 0:
            return float("inf") if within_cluster > 0 else 0.0

        return within_cluster / between_cluster

    def compute_disagreement_exposure(
        self,
        users: dict[str, User],
        opinions: dict[str, float],
    ) -> dict[str, float]:
        """Compute disagreement exposure metrics.

        Lower values indicate filter bubbles.

        Args:
            users: Dictionary of users
            opinions: User opinions

        Returns:
            Dictionary with mean, std, and median disagreement exposure
        """
        exposures = []

        for user_id, user in users.items():
            if user_id not in opinions:
                continue

            user_opinion = opinions[user_id]
            disagreements = []

            for friend_id in user.following:
                if friend_id in opinions:
                    disagreements.append(abs(user_opinion - opinions[friend_id]))

            if disagreements:
                exposures.append(np.mean(disagreements))

        if not exposures:
            return {"mean": 0.0, "std": 0.0, "median": 0.0}

        return {
            "mean": float(np.mean(exposures)),
            "std": float(np.std(exposures)),
            "median": float(np.median(exposures)),
        }

    def compute_opinion_clusters(
        self,
        opinions: dict[str, float],
        n_bins: int = 5,
    ) -> dict[str, Any]:
        """Analyze opinion distribution by clusters.

        Args:
            opinions: User opinions
            n_bins: Number of bins for histogram

        Returns:
            Cluster analysis results
        """
        if not opinions:
            return {}

        values = list(opinions.values())

        # Histogram
        hist, bin_edges = np.histogram(values, bins=n_bins, range=(-1, 1))

        # Find dominant clusters
        dominant_idx = np.argmax(hist)

        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "dominant_cluster": int(dominant_idx),
            "dominant_cluster_size": int(hist[dominant_idx]),
            "n_clusters_above_10pct": int(np.sum(hist > len(values) * 0.1)),
        }

    def compute_all_metrics(
        self,
        users: dict[str, User],
        opinions: dict[str, float] | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Compute all polarization metrics.

        Args:
            users: Dictionary of users
            opinions: Optional explicit opinions (defaults to ideology)
            step: Current step for history tracking

        Returns:
            Dictionary of all metrics
        """
        if opinions is None:
            opinions = {u.user_id: u.traits.ideology for u in users.values()}

        if not opinions:
            return {}

        opinion_values = list(opinions.values())

        metrics = {
            "n_users": len(opinions),
            "mean_opinion": float(np.mean(opinion_values)),
            "std_opinion": float(np.std(opinion_values)),
            "variance": float(np.var(opinion_values)),
            "bimodality_coefficient": self.compute_bimodality_coefficient(opinion_values),
            "echo_chamber_index": self.compute_echo_chamber_index(users, opinions),
            "disagreement_exposure": self.compute_disagreement_exposure(users, opinions),
            "clusters": self.compute_opinion_clusters(opinions),
            "min_opinion": float(np.min(opinion_values)),
            "max_opinion": float(np.max(opinion_values)),
            "range": float(np.max(opinion_values) - np.min(opinion_values)),
        }

        # Extreme opinion counts
        metrics["extreme_left_pct"] = float(np.mean([v < -0.7 for v in opinion_values]))
        metrics["extreme_right_pct"] = float(np.mean([v > 0.7 for v in opinion_values]))
        metrics["moderate_pct"] = float(np.mean([abs(v) < 0.3 for v in opinion_values]))

        if step is not None:
            metrics["step"] = step
            self.history.append(metrics)

        return metrics

    def get_trend(
        self,
        metric_name: str,
        window: int = 10,
    ) -> dict[str, float] | None:
        """Get trend for a metric over time.

        Args:
            metric_name: Name of metric to analyze
            window: Number of recent observations

        Returns:
            Trend statistics or None
        """
        if len(self.history) < 2:
            return None

        recent = self.history[-window:]
        values = [m.get(metric_name) for m in recent if metric_name in m]

        if len(values) < 2:
            return None

        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "start_value": float(values[0]),
            "end_value": float(values[-1]),
            "change": float(values[-1] - values[0]),
            "change_pct": float((values[-1] - values[0]) / (abs(values[0]) + 1e-10)),
        }

    def compute_content_metrics(
        self,
        state: SimulationState,
    ) -> dict[str, Any]:
        """Compute content-level metrics.

        Args:
            state: Simulation state

        Returns:
            Dictionary of content metrics
        """
        posts = list(state.posts.values())

        if not posts:
            return {}

        quality_scores = [p.content.quality_score for p in posts]
        controversy_scores = [p.content.controversy_score for p in posts]
        engagement_counts = [p.total_engagement for p in posts]
        virality_scores = [p.virality_score for p in posts]

        misinfo_count = sum(1 for p in posts if p.content.is_misinformation)
        misinfo_rate = misinfo_count / len(posts)

        return {
            "total_posts": len(posts),
            "avg_quality": float(np.mean(quality_scores)),
            "avg_controversy": float(np.mean(controversy_scores)),
            "avg_engagement": float(np.mean(engagement_counts)),
            "avg_virality": float(np.mean(virality_scores)),
            "max_engagement": int(np.max(engagement_counts)),
            "misinformation_rate": misinfo_rate,
            "misinfo_count": misinfo_count,
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.step_metrics = []
        self.previous_totals = defaultdict(int)
