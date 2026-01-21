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
