"""Tests for feed ranking algorithms."""

import pytest
from unittest.mock import MagicMock

from engine.feed import FeedRanker
from models.post import Post, PostContent
from config.schemas import SimulationConfig


class TestFeedRanker:
    """Tests for FeedRanker class."""

    @pytest.fixture
    def feed_ranker(self, small_config):
        """Create feed ranker for testing."""
        return FeedRanker(small_config)

    @pytest.fixture
    def sample_posts(self):
        """Create sample posts for ranking."""
        posts = []
        for i in range(5):
            posts.append(Post(
                post_id=f"post_{i}",
                author_id=f"author_{i}",
                content=PostContent(
                    topics={"topic_1": 0.5 + i * 0.1},
                    sentiment=0.5,
                    quality_score=0.3 + i * 0.1,
                    controversy_score=0.2,
                    emotional_intensity=0.4,
                    is_misinformation=False,
                ),
                created_step=i,
                like_count=i * 10,
                share_count=i * 2,
            ))
        return posts

    def test_chronological_ranking(self, feed_ranker, sample_posts):
        """Test chronological feed ranking."""
        feed_ranker.algorithm = "chronological"

        ranked = feed_ranker.rank_posts(
            posts=sample_posts,
            user=None,
            current_step=10,
        )

        # Should be sorted by created_step descending (newest first)
        for i in range(len(ranked) - 1):
            assert ranked[i].created_step >= ranked[i + 1].created_step

    def test_engagement_ranking(self, feed_ranker, sample_posts):
        """Test engagement-based feed ranking."""
        feed_ranker.algorithm = "engagement"

        ranked = feed_ranker.rank_posts(
            posts=sample_posts,
            user=None,
            current_step=10,
        )

        # Posts should be returned (order depends on engagement scores)
        assert len(ranked) == len(sample_posts)

    def test_empty_posts_list(self, feed_ranker):
        """Test ranking with empty posts list."""
        ranked = feed_ranker.rank_posts(
            posts=[],
            user=None,
            current_step=0,
        )

        assert ranked == []

    def test_limit_parameter(self, feed_ranker, sample_posts):
        """Test limiting number of posts returned."""
        ranked = feed_ranker.rank_posts(
            posts=sample_posts,
            user=None,
            current_step=10,
            limit=3,
        )

        assert len(ranked) == 3

    def test_seen_posts_excluded(self, feed_ranker, sample_posts):
        """Test seen posts are excluded from feed."""
        seen_post_ids = {"post_0", "post_1"}

        ranked = feed_ranker.rank_posts(
            posts=sample_posts,
            user=None,
            current_step=10,
            seen_post_ids=seen_post_ids,
        )

        for post in ranked:
            assert post.post_id not in seen_post_ids
