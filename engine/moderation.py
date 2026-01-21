"""Content moderation system."""

from typing import Any
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from config.schemas import ModerationConfig
from models import Post, User
from models.enums import PostState
from .state import SimulationState


@dataclass
class ModerationDecision:
    """Result of a moderation check."""

    post_id: str
    is_flagged: bool
    confidence: float
    action: str  # "none", "suppress", "remove"
    is_false_positive: bool = False


class ModerationEngine:
    """Simulates content moderation system.

    Features:
    - Misinformation detection with configurable accuracy
    - False positive simulation
    - Content suppression and removal
    - User credibility impact
    """

    def __init__(
        self,
        config: ModerationConfig,
        seed: int | None = None,
    ):
        """Initialize moderation engine.

        Args:
            config: Moderation configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Track moderation history
        self.decisions: list[ModerationDecision] = []
        self.total_flagged = 0
        self.total_suppressed = 0
        self.total_removed = 0
        self.false_positives = 0
        self.false_negatives = 0

    def moderate_post(
        self,
        post: Post,
        state: SimulationState,
    ) -> ModerationDecision:
        """Perform moderation check on a post.

        Args:
            post: Post to moderate
            state: Simulation state

        Returns:
            ModerationDecision
        """
        if not self.config.enabled:
            return ModerationDecision(
                post_id=post.post_id,
                is_flagged=False,
                confidence=0.0,
                action="none",
            )

        is_misinfo = post.content.is_misinformation

        # Simulate detection with configured accuracy
        detected, confidence = self._detect_misinformation(post)

        # Determine if this is a false positive/negative
        is_false_positive = detected and not is_misinfo
        is_false_negative = not detected and is_misinfo

        # Determine action based on confidence
        action = self._determine_action(detected, confidence)

        # Apply action
        if action == "suppress":
            post.suppress()
            self.total_suppressed += 1
        elif action == "remove":
            post.remove()
            self.total_removed += 1

        # Update statistics
        if detected:
            self.total_flagged += 1
        if is_false_positive:
            self.false_positives += 1
        if is_false_negative:
            self.false_negatives += 1

        # Store moderation score on post
        post.moderation_score = confidence

        decision = ModerationDecision(
            post_id=post.post_id,
            is_flagged=detected,
            confidence=confidence,
            action=action,
            is_false_positive=is_false_positive,
        )
        self.decisions.append(decision)

        return decision

    def _detect_misinformation(self, post: Post) -> tuple[bool, float]:
        """Simulate misinformation detection.

        Args:
            post: Post to check

        Returns:
            Tuple of (is_detected, confidence)
        """
        is_misinfo = post.content.is_misinformation

        # Generate base confidence score
        if is_misinfo:
            # True positive scenario
            # Confidence depends on detection accuracy and content signals
            base_confidence = self.config.detection_accuracy

            # Content signals that make detection easier
            signal_boost = 0.0
            if post.content.quality_score < 0.3:
                signal_boost += 0.1
            if post.content.controversy_score > 0.7:
                signal_boost += 0.05
            if post.content.emotional_intensity > 0.7:
                signal_boost += 0.05

            confidence = min(0.99, base_confidence + signal_boost + self.rng.normal(0, 0.1))
            confidence = max(0.0, confidence)

            # Detection based on accuracy
            detected = self.rng.random() < self.config.detection_accuracy

        else:
            # Potential false positive scenario
            # Some legitimate content may be flagged

            # Base false positive rate
            fp_rate = self.config.false_positive_rate

            # Content signals that increase false positive risk
            if post.content.controversy_score > 0.6:
                fp_rate *= 1.5
            if post.content.quality_score < 0.4:
                fp_rate *= 1.3

            fp_rate = min(0.3, fp_rate)

            detected = self.rng.random() < fp_rate

            if detected:
                # Lower confidence for false positives
                confidence = float(self.rng.uniform(0.3, 0.7))
            else:
                confidence = float(self.rng.uniform(0.0, 0.3))

        return detected, float(confidence)

    def _determine_action(
        self,
        detected: bool,
        confidence: float,
    ) -> str:
        """Determine moderation action based on detection result.

        Args:
            detected: Whether misinformation was detected
            confidence: Confidence score

        Returns:
            Action string: "none", "suppress", or "remove"
        """
        if not detected:
            return "none"

        if confidence >= self.config.removal_threshold:
            return "remove"
        elif confidence >= self.config.suppression_factor:
            return "suppress"
        else:
            return "none"

    def apply_suppression_factor(
        self,
        post: Post,
        base_visibility: float,
    ) -> float:
        """Apply suppression factor to post visibility.

        Args:
            post: Post to check
            base_visibility: Base visibility score

        Returns:
            Modified visibility
        """
        if post.state == PostState.SUPPRESSED:
            return base_visibility * (1 - self.config.suppression_factor)
        elif post.state == PostState.REMOVED:
            return 0.0
        return base_visibility

    def update_user_credibility(
        self,
        user: User,
        decision: ModerationDecision,
    ) -> None:
        """Update user credibility based on moderation decision.

        Args:
            user: Post author
            decision: Moderation decision
        """
        if decision.action == "remove":
            # Significant credibility hit for removed content
            user.update_credibility(-0.15)
        elif decision.action == "suppress":
            # Smaller credibility hit for suppressed content
            user.update_credibility(-0.05)

    def get_statistics(self) -> dict[str, Any]:
        """Get moderation statistics.

        Returns:
            Dictionary of statistics
        """
        total_decisions = len(self.decisions)

        if total_decisions == 0:
            return {
                "enabled": self.config.enabled,
                "total_decisions": 0,
            }

        return {
            "enabled": self.config.enabled,
            "total_decisions": total_decisions,
            "total_flagged": self.total_flagged,
            "total_suppressed": self.total_suppressed,
            "total_removed": self.total_removed,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "flag_rate": self.total_flagged / total_decisions,
            "suppression_rate": self.total_suppressed / total_decisions,
            "removal_rate": self.total_removed / total_decisions,
            "false_positive_rate": self.false_positives / max(1, self.total_flagged),
        }

    def get_recent_decisions(
        self,
        n: int = 10,
    ) -> list[ModerationDecision]:
        """Get recent moderation decisions.

        Args:
            n: Number of decisions to return

        Returns:
            List of recent decisions
        """
        return self.decisions[-n:]

    def reset_statistics(self) -> None:
        """Reset moderation statistics."""
        self.decisions = []
        self.total_flagged = 0
        self.total_suppressed = 0
        self.total_removed = 0
        self.false_positives = 0
        self.false_negatives = 0
