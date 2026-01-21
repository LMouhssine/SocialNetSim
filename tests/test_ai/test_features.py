"""Tests for feature engineering."""

import pytest
import numpy as np
import pandas as pd

from ai.features import FeatureExtractor
from models.post import Post, PostContent
from models.user import User, UserTraits
from models.interaction import Interaction
from models.enums import InteractionType


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor for testing."""
        return FeatureExtractor()

    @pytest.fixture
    def sample_post(self):
        """Create sample post."""
        return Post(
            post_id="test_post",
            author_id="author_1",
            content=PostContent(
                topics={"politics": 0.8, "news": 0.5},
                sentiment=0.6,
                quality_score=0.7,
                controversy_score=0.4,
                emotional_intensity=0.65,
                is_misinformation=False,
            ),
            created_step=5,
            like_count=50,
            share_count=10,
            comment_count=25,
        )

    @pytest.fixture
    def sample_user(self):
        """Create sample user."""
        return User(
            user_id="user_1",
            interests={"politics": 0.9, "tech": 0.3},
            traits=UserTraits(
                ideology=0.2,
                confirmation_bias=0.4,
                misinfo_susceptibility=0.2,
                emotional_reactivity=0.5,
                activity_level=0.7,
            ),
            followers=["f1", "f2", "f3"],
            following=["u1", "u2"],
            influence_score=0.6,
            credibility_score=0.8,
        )

    def test_extract_post_features(self, feature_extractor, sample_post):
        """Test extracting features from a post."""
        features = feature_extractor.extract_post_features(sample_post)

        assert isinstance(features, dict)
        assert "quality_score" in features
        assert "sentiment" in features
        assert "controversy_score" in features
        assert "emotional_intensity" in features
        assert "like_count" in features
        assert "share_count" in features
        assert features["quality_score"] == 0.7
        assert features["like_count"] == 50

    def test_extract_user_features(self, feature_extractor, sample_user):
        """Test extracting features from a user."""
        features = feature_extractor.extract_user_features(sample_user)

        assert isinstance(features, dict)
        assert "follower_count" in features
        assert "following_count" in features
        assert "influence_score" in features
        assert "activity_level" in features
        assert features["follower_count"] == 3
        assert features["influence_score"] == 0.6

    def test_extract_early_signals(self, feature_extractor, sample_post):
        """Test extracting early engagement signals."""
        interactions = [
            Interaction(
                user_id="u1",
                target_id=sample_post.post_id,
                interaction_type=InteractionType.LIKE,
                step=6,
            ),
            Interaction(
                user_id="u2",
                target_id=sample_post.post_id,
                interaction_type=InteractionType.SHARE,
                step=6,
            ),
            Interaction(
                user_id="u3",
                target_id=sample_post.post_id,
                interaction_type=InteractionType.LIKE,
                step=7,
            ),
        ]

        features = feature_extractor.extract_early_signals(
            post=sample_post,
            interactions=interactions,
            window_steps=5,
        )

        assert isinstance(features, dict)
        assert "early_likes" in features
        assert "early_shares" in features
        assert features["early_likes"] >= 0
        assert features["early_shares"] >= 0

    def test_build_feature_matrix(self, feature_extractor):
        """Test building feature matrix from multiple posts."""
        posts = [
            Post(
                post_id=f"post_{i}",
                author_id="author",
                content=PostContent(
                    topics={"topic": 0.5},
                    sentiment=0.5,
                    quality_score=0.3 + i * 0.1,
                    controversy_score=0.2,
                    emotional_intensity=0.4,
                    is_misinformation=False,
                ),
                created_step=i,
            )
            for i in range(5)
        ]

        df = feature_extractor.build_feature_matrix(posts)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "quality_score" in df.columns

    def test_feature_names(self, feature_extractor):
        """Test getting feature names."""
        names = feature_extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
