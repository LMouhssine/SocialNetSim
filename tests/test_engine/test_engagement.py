"""Tests for engagement model."""

import pytest
import numpy as np

from engine.engagement import EngagementModel
from models.user import User, UserTraits
from models.post import Post, PostContent
from models.enums import InteractionType


class TestEngagementModel:
    """Tests for EngagementModel class."""

    @pytest.fixture
    def engagement_model(self, small_world):
        """Create engagement model for testing."""
        return EngagementModel(small_world.config.engagement, seed=small_world.seed)

    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            user_id="test_user",
            interests={"politics": 0.8, "tech": 0.5},
            traits=UserTraits(
                ideology=0.3,
                confirmation_bias=0.5,
                misinfo_susceptibility=0.3,
                emotional_reactivity=0.5,
                activity_level=0.6,
                openness=0.6,
                conscientiousness=0.4,
            ),
        )

    @pytest.fixture
    def sample_post(self):
        """Create sample post for testing."""
        return Post(
            post_id="test_post",
            author_id="author_1",
            content=PostContent(
                topics={"politics": 0.9},
                sentiment=0.5,
                quality_score=0.7,
                controversy_score=0.3,
                emotional_intensity=0.5,
                is_misinformation=False,
            ),
            created_step=0,
        )

    def test_engagement_returns_probability(self, engagement_model, sample_user, sample_post):
        """Test engagement model returns valid probability."""
        prob = engagement_model.calculate_engagement_probability(
            user=sample_user,
            post=sample_post,
            current_step=1,
        )

        assert 0.0 <= prob <= 1.0

    def test_interest_match_increases_engagement(self, engagement_model, sample_user, sample_post):
        """Test that matching interests increase engagement."""
        # High interest match
        prob_high = engagement_model.calculate_engagement_probability(
            user=sample_user,
            post=sample_post,  # politics topic matches user interest
            current_step=1,
        )

        # Create post with non-matching topic
        non_matching_post = Post(
            post_id="test_post_2",
            author_id="author_1",
            content=PostContent(
                topics={"sports": 0.9},  # User has no sports interest
                sentiment=0.5,
                quality_score=0.7,
                controversy_score=0.3,
                emotional_intensity=0.5,
                is_misinformation=False,
            ),
            created_step=0,
        )

        prob_low = engagement_model.calculate_engagement_probability(
            user=sample_user,
            post=non_matching_post,
            current_step=1,
        )

        assert prob_high > prob_low

    def test_post_age_affects_engagement(self, engagement_model, sample_user, sample_post):
        """Test older posts have lower engagement."""
        prob_fresh = engagement_model.calculate_engagement_probability(
            user=sample_user,
            post=sample_post,
            current_step=1,
        )

        prob_old = engagement_model.calculate_engagement_probability(
            user=sample_user,
            post=sample_post,
            current_step=100,
        )

        assert prob_fresh >= prob_old

    def test_decide_interaction_type(self, engagement_model, sample_user, sample_post):
        """Test interaction type decision."""
        interaction = engagement_model.decide_interaction_type(
            user=sample_user,
            post=sample_post,
        )

        assert interaction in [
            InteractionType.VIEW,
            InteractionType.LIKE,
            InteractionType.COMMENT,
            InteractionType.SHARE,
            None,
        ]
