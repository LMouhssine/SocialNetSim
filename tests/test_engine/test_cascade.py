"""Tests for cascade engine."""

import pytest

from engine.cascade import CascadeEngine
from models.post import Post, PostContent
from models.interaction import Cascade


class TestCascadeEngine:
    """Tests for CascadeEngine class."""

    @pytest.fixture
    def cascade_engine(self, small_world):
        """Create cascade engine for testing."""
        return CascadeEngine(small_world.config, small_world)

    @pytest.fixture
    def viral_post(self):
        """Create a post with high virality potential."""
        return Post(
            post_id="viral_post",
            author_id="influencer",
            content=PostContent(
                topics={"trending": 1.0},
                sentiment=0.8,
                quality_score=0.9,
                controversy_score=0.7,
                emotional_intensity=0.9,
                is_misinformation=False,
            ),
            created_step=0,
            like_count=100,
            share_count=50,
        )

    def test_create_cascade(self, cascade_engine, viral_post):
        """Test cascade creation."""
        cascade = cascade_engine.create_cascade(viral_post)

        assert cascade is not None
        assert cascade.root_post_id == viral_post.post_id
        assert cascade.original_author_id == viral_post.author_id

    def test_cascade_depth_tracking(self, cascade_engine, viral_post):
        """Test cascade tracks depth correctly."""
        cascade = cascade_engine.create_cascade(viral_post)

        assert cascade.max_depth == 0  # Initially just the root
        assert cascade.total_reach >= 1

    def test_cascade_metrics(self, cascade_engine, viral_post):
        """Test cascade computes metrics."""
        cascade = cascade_engine.create_cascade(viral_post)
        metrics = cascade.get_metrics()

        assert "total_reach" in metrics
        assert "max_depth" in metrics
        assert "virality_score" in metrics

    def test_calculate_virality_potential(self, cascade_engine, viral_post):
        """Test virality potential calculation."""
        potential = cascade_engine.calculate_virality_potential(viral_post)

        # Should be a value between 0 and 1
        assert 0.0 <= potential <= 1.0

    def test_high_quality_has_higher_virality(self, cascade_engine):
        """Test high quality posts have higher virality potential."""
        high_quality = Post(
            post_id="hq",
            author_id="author",
            content=PostContent(
                topics={"topic": 0.5},
                sentiment=0.7,
                quality_score=0.95,
                controversy_score=0.5,
                emotional_intensity=0.8,
                is_misinformation=False,
            ),
            created_step=0,
        )

        low_quality = Post(
            post_id="lq",
            author_id="author",
            content=PostContent(
                topics={"topic": 0.5},
                sentiment=0.7,
                quality_score=0.1,
                controversy_score=0.5,
                emotional_intensity=0.2,
                is_misinformation=False,
            ),
            created_step=0,
        )

        hq_potential = cascade_engine.calculate_virality_potential(high_quality)
        lq_potential = cascade_engine.calculate_virality_potential(low_quality)

        assert hq_potential > lq_potential
