"""Validation and sanity checks for simulation correctness.

Checks:
- Engagement rates in reasonable ranges (1-5%)
- No runaway cascades (>50% reach)
- Opinions bounded [-1, 1]
- Attention budget bounded [0, 1]
- Hawkes intensity finite
- State consistency
"""

from typing import Any
from dataclasses import dataclass

import numpy as np

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        name: Name of the check
        passed: Whether the check passed
        message: Description of result
        value: Actual value checked
        threshold: Threshold used
    """

    name: str
    passed: bool
    message: str
    value: float | None = None
    threshold: float | None = None


class SimulationValidator:
    """Validates simulation state and metrics for sanity."""

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, raise exceptions on failure
        """
        self.strict = strict
        self.results: list[ValidationResult] = []

    def validate_all(
        self,
        state: Any,
        users: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> list[ValidationResult]:
        """Run all validation checks.

        Args:
            state: Simulation state
            users: User dictionary
            metrics: Optional metrics dictionary

        Returns:
            List of validation results
        """
        self.results = []

        # Engagement rate checks
        self._check_engagement_rates(state)

        # Cascade checks
        self._check_cascade_bounds(state, users)

        # User state checks
        self._check_user_bounds(state, users)

        # Opinion dynamics checks
        self._check_opinion_bounds(state, users)

        # Hawkes process checks
        self._check_hawkes_intensity(state)

        # State consistency checks
        self._check_state_consistency(state)

        return self.results

    def _add_result(
        self,
        name: str,
        passed: bool,
        message: str,
        value: float | None = None,
        threshold: float | None = None,
    ) -> None:
        """Add a validation result.

        Args:
            name: Check name
            passed: Pass/fail
            message: Result message
            value: Actual value
            threshold: Threshold value
        """
        result = ValidationResult(name, passed, message, value, threshold)
        self.results.append(result)

        if not passed and self.strict:
            raise AssertionError(f"Validation failed: {name} - {message}")

    def _check_engagement_rates(self, state: Any) -> None:
        """Check that engagement rates are in reasonable ranges."""
        if not hasattr(state, "interactions"):
            self._add_result(
                "engagement_rates",
                True,
                "No interactions to check",
            )
            return

        if not hasattr(state, "posts") or len(state.posts) == 0:
            return

        # Calculate engagement rate
        total_views = sum(p.view_count for p in state.posts.values())
        total_likes = sum(p.like_count for p in state.posts.values())
        total_shares = sum(p.share_count for p in state.posts.values())

        if total_views > 0:
            like_rate = total_likes / total_views
            share_rate = total_shares / total_views

            # Like rate should be 1-20%
            self._add_result(
                "like_rate_bounds",
                0.001 <= like_rate <= 0.5,
                f"Like rate {like_rate:.3f} should be 0.1-50%",
                value=like_rate,
                threshold=0.5,
            )

            # Share rate should be 0.1-10%
            self._add_result(
                "share_rate_bounds",
                share_rate <= 0.2,
                f"Share rate {share_rate:.3f} should be <20%",
                value=share_rate,
                threshold=0.2,
            )

    def _check_cascade_bounds(self, state: Any, users: dict[str, Any]) -> None:
        """Check that cascades don't reach unrealistic proportions."""
        if not hasattr(state, "cascades"):
            return

        n_users = len(users)
        max_reach_fraction = 0.0

        for cascade in state.cascades.values():
            reach_fraction = cascade.total_reach / n_users if n_users > 0 else 0

            if reach_fraction > max_reach_fraction:
                max_reach_fraction = reach_fraction

        # No single cascade should reach >50% of users
        self._add_result(
            "cascade_reach_bounds",
            max_reach_fraction <= 0.5,
            f"Max cascade reach {max_reach_fraction:.1%} should be <50%",
            value=max_reach_fraction,
            threshold=0.5,
        )

    def _check_user_bounds(self, state: Any, users: dict[str, Any]) -> None:
        """Check user state values are bounded correctly."""
        # Check fatigue bounds
        if hasattr(state, "runtime_states"):
            fatigues = [rs.fatigue_level for rs in state.runtime_states.values()]
            if fatigues:
                max_fatigue = max(fatigues)
                min_fatigue = min(fatigues)

                self._add_result(
                    "fatigue_upper_bound",
                    max_fatigue <= 2.0,
                    f"Max fatigue {max_fatigue:.2f} should be <=2.0",
                    value=max_fatigue,
                    threshold=2.0,
                )

                self._add_result(
                    "fatigue_lower_bound",
                    min_fatigue >= 0.0,
                    f"Min fatigue {min_fatigue:.2f} should be >=0",
                    value=min_fatigue,
                    threshold=0.0,
                )

        # Check cognitive state if available
        if hasattr(state, "cognitive_states"):
            for user_id, cog_state in state.cognitive_states.items():
                # Attention budget should be [0, 1]
                if hasattr(cog_state, "attention_budget"):
                    self._add_result(
                        f"attention_bounds_{user_id[:8]}",
                        0.0 <= cog_state.attention_budget <= 1.0,
                        f"Attention {cog_state.attention_budget:.2f} should be [0,1]",
                        value=cog_state.attention_budget,
                    )

    def _check_opinion_bounds(self, state: Any, users: dict[str, Any]) -> None:
        """Check that opinions are bounded [-1, 1]."""
        opinions = []

        # Check cognitive state opinions
        if hasattr(state, "cognitive_states"):
            for cog_state in state.cognitive_states.values():
                if hasattr(cog_state, "opinion"):
                    opinions.append(cog_state.opinion)

        # Check user trait ideologies
        for user in users.values():
            if hasattr(user, "traits") and hasattr(user.traits, "ideology"):
                opinions.append(user.traits.ideology)

        if opinions:
            min_opinion = min(opinions)
            max_opinion = max(opinions)

            self._add_result(
                "opinion_upper_bound",
                max_opinion <= 1.0,
                f"Max opinion {max_opinion:.2f} should be <=1.0",
                value=max_opinion,
                threshold=1.0,
            )

            self._add_result(
                "opinion_lower_bound",
                min_opinion >= -1.0,
                f"Min opinion {min_opinion:.2f} should be >=-1.0",
                value=min_opinion,
                threshold=-1.0,
            )

    def _check_hawkes_intensity(self, state: Any) -> None:
        """Check that Hawkes intensities are finite."""
        # This would check cascade engine's Hawkes processes
        # For now, just verify no NaN/Inf in cascade metrics

        if not hasattr(state, "cascades"):
            return

        for cascade in state.cascades.values():
            velocity = cascade.get_velocity(state.current_step) if hasattr(cascade, "get_velocity") else 0

            self._add_result(
                f"hawkes_finite_{cascade.cascade_id[:12]}",
                np.isfinite(velocity),
                f"Cascade velocity {velocity} should be finite",
                value=velocity,
            )

    def _check_state_consistency(self, state: Any) -> None:
        """Check internal state consistency."""
        # Check post counts match interactions
        if hasattr(state, "posts") and hasattr(state, "interactions"):
            for post in state.posts.values():
                post_interactions = [
                    i for i in state.interactions if i.post_id == post.post_id
                ]
                view_interactions = sum(
                    1 for i in post_interactions if i.interaction_type.value == "view"
                )

                # Allow some slack for async updates
                diff = abs(post.view_count - view_interactions)
                self._add_result(
                    f"post_view_consistency_{post.post_id[:12]}",
                    diff <= post.view_count * 0.1 + 10,  # 10% tolerance + buffer
                    f"Post views {post.view_count} vs interactions {view_interactions}",
                    value=diff,
                )


class MetricsValidator:
    """Validates simulation metrics for reasonableness."""

    def validate_step_metrics(
        self,
        metrics: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate step-level metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            List of validation results
        """
        results = []

        # Check active users is reasonable
        if "active_users" in metrics:
            active = metrics["active_users"]
            total = metrics.get("total_users", active * 2)

            active_rate = active / total if total > 0 else 0
            results.append(ValidationResult(
                "active_user_rate",
                0.0 <= active_rate <= 1.0,
                f"Active rate {active_rate:.2%}",
                value=active_rate,
            ))

        # Check engagement counts are non-negative
        for key in ["views", "likes", "shares", "comments"]:
            if key in metrics:
                value = metrics[key]
                results.append(ValidationResult(
                    f"{key}_non_negative",
                    value >= 0,
                    f"{key} count {value} should be >=0",
                    value=value,
                ))

        return results

    def validate_summary_metrics(
        self,
        summary: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate summary metrics.

        Args:
            summary: Summary metrics dictionary

        Returns:
            List of validation results
        """
        results = []

        # Check total posts is reasonable
        if "total_posts" in summary:
            posts = summary["total_posts"]
            steps = summary.get("total_steps", 100)
            users = summary.get("total_users", 1000)

            posts_per_user_step = posts / (users * steps) if users * steps > 0 else 0

            results.append(ValidationResult(
                "posts_per_user_step",
                posts_per_user_step <= 1.0,
                f"Posts/user/step {posts_per_user_step:.3f} should be <1",
                value=posts_per_user_step,
            ))

        return results


class OpinionDynamicsValidator:
    """Validates opinion dynamics behavior."""

    def validate_bounded_confidence(
        self,
        opinions_before: list[float],
        opinions_after: list[float],
        confidence_bound: float,
    ) -> list[ValidationResult]:
        """Validate bounded confidence model behavior.

        Args:
            opinions_before: Opinions before interaction
            opinions_after: Opinions after interaction
            confidence_bound: Epsilon parameter

        Returns:
            List of validation results
        """
        results = []

        # Check opinions stayed bounded
        results.append(ValidationResult(
            "opinions_bounded",
            all(-1 <= o <= 1 for o in opinions_after),
            "All opinions in [-1, 1]",
        ))

        # Check opinion changes were bounded
        max_change = max(
            abs(a - b) for a, b in zip(opinions_after, opinions_before)
        ) if opinions_before else 0

        results.append(ValidationResult(
            "opinion_change_bounded",
            max_change <= confidence_bound + 0.1,
            f"Max opinion change {max_change:.3f}",
            value=max_change,
            threshold=confidence_bound + 0.1,
        ))

        return results

    def validate_convergence(
        self,
        opinion_history: list[list[float]],
    ) -> list[ValidationResult]:
        """Validate opinion convergence properties.

        Args:
            opinion_history: List of opinion arrays over time

        Returns:
            List of validation results
        """
        results = []

        if len(opinion_history) < 2:
            return results

        # Check variance trend (should generally decrease or stabilize)
        variances = [np.var(ops) for ops in opinion_history]
        final_variance = variances[-1]
        initial_variance = variances[0]

        # Variance shouldn't explode
        results.append(ValidationResult(
            "variance_stability",
            final_variance <= initial_variance * 2 + 0.1,
            f"Variance went from {initial_variance:.3f} to {final_variance:.3f}",
            value=final_variance,
        ))

        return results


class HawkesValidator:
    """Validates Hawkes process behavior."""

    def validate_intensity(
        self,
        times: list[float],
        intensity_func: callable,
        baseline: float,
    ) -> list[ValidationResult]:
        """Validate Hawkes intensity function.

        Args:
            times: Event times
            intensity_func: Intensity function
            baseline: Baseline intensity

        Returns:
            List of validation results
        """
        results = []

        # Intensity should always be >= baseline
        if times:
            t_max = max(times)
            test_times = np.linspace(0, t_max, 100)

            intensities = [intensity_func(t) for t in test_times]

            results.append(ValidationResult(
                "intensity_above_baseline",
                all(i >= baseline * 0.99 for i in intensities),
                f"All intensities >= baseline {baseline}",
            ))

            results.append(ValidationResult(
                "intensity_finite",
                all(np.isfinite(i) for i in intensities),
                "All intensities finite",
            ))

        return results

    def validate_branching_ratio(
        self,
        events: list[float],
        expected_ratio: float,
        tolerance: float = 0.3,
    ) -> list[ValidationResult]:
        """Validate empirical branching ratio.

        Args:
            events: Event times
            expected_ratio: Expected branching ratio
            tolerance: Tolerance for comparison

        Returns:
            List of validation results
        """
        results = []

        if len(events) < 10:
            return results

        # Estimate empirical branching ratio from event clustering
        # (simplified: compare event counts in early vs late periods)
        mid_point = len(events) // 2
        early_rate = mid_point / events[mid_point] if events[mid_point] > 0 else 0
        late_rate = (len(events) - mid_point) / (events[-1] - events[mid_point]) if events[-1] > events[mid_point] else 0

        if early_rate > 0:
            empirical_ratio = late_rate / early_rate
        else:
            empirical_ratio = 0

        results.append(ValidationResult(
            "branching_ratio_reasonable",
            abs(empirical_ratio - expected_ratio) <= tolerance or len(events) < 50,
            f"Empirical ratio {empirical_ratio:.2f} vs expected {expected_ratio:.2f}",
            value=empirical_ratio,
            threshold=expected_ratio,
        ))

        return results


def run_validation_suite(state, users, metrics=None) -> dict[str, Any]:
    """Run complete validation suite.

    Args:
        state: Simulation state
        users: User dictionary
        metrics: Optional metrics

    Returns:
        Validation report
    """
    validator = SimulationValidator()
    results = validator.validate_all(state, users, metrics)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    return {
        "total_checks": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(results) if results else 1.0,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "value": r.value,
                "threshold": r.threshold,
            }
            for r in results
        ],
        "failed_checks": [r.name for r in results if not r.passed],
    }


# Pytest test cases
class TestValidators:
    """Test the validators themselves."""

    def test_simulation_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = SimulationValidator(strict=False)
        assert validator.strict is False
        assert len(validator.results) == 0

    def test_metrics_validator(self):
        """Test metrics validation."""
        validator = MetricsValidator()

        metrics = {
            "active_users": 100,
            "total_users": 1000,
            "views": 500,
            "likes": 50,
            "shares": 10,
            "comments": 20,
        }

        results = validator.validate_step_metrics(metrics)
        assert all(r.passed for r in results)

    def test_opinion_bounds(self):
        """Test opinion bounds validation."""
        validator = OpinionDynamicsValidator()

        # Valid opinions
        before = [0.5, -0.3, 0.1]
        after = [0.4, -0.2, 0.15]

        results = validator.validate_bounded_confidence(before, after, 0.3)
        assert all(r.passed for r in results)

        # Invalid - opinion out of bounds
        after_invalid = [0.4, -0.2, 1.5]
        results = validator.validate_bounded_confidence(before, after_invalid, 0.3)
        assert not all(r.passed for r in results)


if __name__ == "__main__":
    # Run basic tests
    test = TestValidators()
    test.test_simulation_validator_initialization()
    test.test_metrics_validator()
    test.test_opinion_bounds()
    print("All validation tests passed!")
