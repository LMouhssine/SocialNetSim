"""Simulation state management."""

from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np

from models import User, Post, Interaction, Cascade, Event
from models.enums import UserState


@dataclass
class UserRuntimeState:
    """Runtime state for a user during simulation.

    Attributes:
        user_id: User identifier
        seen_posts: Set of post IDs user has seen
        fatigue: Current fatigue level (0-1)
        last_active_step: Last step user was active
        session_interactions: Interactions in current session
        daily_post_count: Posts created today/recent window
        engagement_history: Recent engagement timestamps for velocity calculation
        topic_exposure_counts: Count of exposures to each topic
        author_interaction_counts: Count of interactions with each author
    """

    user_id: str
    seen_posts: set[str] = field(default_factory=set)
    fatigue: float = 0.0
    last_active_step: int = 0
    session_interactions: int = 0
    daily_post_count: int = 0
    engagement_history: deque = field(default_factory=lambda: deque(maxlen=100))
    topic_exposure_counts: dict[str, int] = field(default_factory=dict)
    author_interaction_counts: dict[str, int] = field(default_factory=dict)

    def mark_seen(self, post_id: str) -> None:
        """Mark a post as seen."""
        self.seen_posts.add(post_id)

    def has_seen(self, post_id: str) -> bool:
        """Check if user has seen a post."""
        return post_id in self.seen_posts

    def add_fatigue(self, amount: float) -> None:
        """Add fatigue."""
        self.fatigue = min(1.0, self.fatigue + amount)

    def recover_fatigue(self, recovery_rate: float) -> None:
        """Recover from fatigue."""
        self.fatigue = max(0.0, self.fatigue - recovery_rate)

    def reset_session(self) -> None:
        """Reset session state."""
        self.session_interactions = 0

    def record_topic_exposure(self, topic_id: str) -> None:
        """Record exposure to a topic."""
        self.topic_exposure_counts[topic_id] = self.topic_exposure_counts.get(topic_id, 0) + 1

    def record_author_interaction(self, author_id: str) -> None:
        """Record interaction with an author."""
        self.author_interaction_counts[author_id] = self.author_interaction_counts.get(author_id, 0) + 1

    def record_engagement(self, step: int, interaction_type: str) -> None:
        """Record an engagement event with timestamp."""
        self.engagement_history.append((step, interaction_type))

    def get_engagement_velocity(self, current_step: int, window: int = 10) -> float:
        """Calculate recent engagement velocity.

        Args:
            current_step: Current simulation step
            window: Number of steps to look back

        Returns:
            Engagements per step in recent window
        """
        min_step = current_step - window
        recent = sum(1 for step, _ in self.engagement_history if step >= min_step)
        return recent / max(1, window)

    def get_topic_novelty(self, topic_id: str) -> float:
        """Calculate novelty score for a topic (lower exposure = higher novelty).

        Args:
            topic_id: Topic to check

        Returns:
            Novelty score (0-1)
        """
        exposure = self.topic_exposure_counts.get(topic_id, 0)
        return 1.0 / (1.0 + np.log1p(exposure))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "seen_posts": list(self.seen_posts),
            "fatigue": self.fatigue,
            "last_active_step": self.last_active_step,
            "session_interactions": self.session_interactions,
            "daily_post_count": self.daily_post_count,
            "engagement_history": list(self.engagement_history),
            "topic_exposure_counts": self.topic_exposure_counts,
            "author_interaction_counts": self.author_interaction_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserRuntimeState":
        """Create from dictionary."""
        state = cls(user_id=data["user_id"])
        state.seen_posts = set(data.get("seen_posts", []))
        state.fatigue = data.get("fatigue", 0.0)
        state.last_active_step = data.get("last_active_step", 0)
        state.session_interactions = data.get("session_interactions", 0)
        state.daily_post_count = data.get("daily_post_count", 0)
        state.engagement_history = deque(data.get("engagement_history", []), maxlen=100)
        state.topic_exposure_counts = data.get("topic_exposure_counts", {})
        state.author_interaction_counts = data.get("author_interaction_counts", {})
        return state


class SimulationState:
    """Manages the complete state of a simulation.

    This includes all posts, interactions, cascades, events, and
    user runtime states.
    """

    def __init__(self, users: dict[str, User]):
        """Initialize simulation state.

        Args:
            users: Dictionary of users in the simulation
        """
        self.users = users
        self.current_step: int = 0

        # Core state
        self.posts: dict[str, Post] = {}
        self.interactions: list[Interaction] = []
        self.cascades: dict[str, Cascade] = {}
        self.events: dict[str, Event] = {}
        self.active_events: list[Event] = []

        # User runtime states
        self.user_states: dict[str, UserRuntimeState] = {
            user_id: UserRuntimeState(user_id=user_id)
            for user_id in users
        }

        # Indexes for efficient lookups
        self._posts_by_author: dict[str, list[str]] = defaultdict(list)
        self._posts_by_step: dict[int, list[str]] = defaultdict(list)
        self._interactions_by_post: dict[str, list[Interaction]] = defaultdict(list)
        self._interactions_by_user: dict[str, list[Interaction]] = defaultdict(list)
        self._interactions_by_step: dict[int, list[Interaction]] = defaultdict(list)

        # Cognitive states for advanced user modeling (Phase 2)
        self.cognitive_states: dict[str, Any] = {}

        # Metrics history
        self.step_metrics: list[dict[str, Any]] = []

    def advance_step(self) -> None:
        """Advance to next simulation step."""
        self.current_step += 1

        # Recover user fatigue
        for user_state in self.user_states.values():
            user_state.recover_fatigue(0.1)  # Configurable recovery rate
            user_state.reset_session()

        # Update active events
        self._update_active_events()

    def _update_active_events(self) -> None:
        """Update list of active events."""
        self.active_events = [
            event for event in self.events.values()
            if event.is_occurring(self.current_step)
        ]

    def add_post(self, post: Post) -> None:
        """Add a post to the simulation.

        Args:
            post: Post to add
        """
        self.posts[post.post_id] = post
        self._posts_by_author[post.author_id].append(post.post_id)
        self._posts_by_step[post.created_step].append(post.post_id)

        # Update author state
        if post.author_id in self.user_states:
            self.user_states[post.author_id].daily_post_count += 1

    def add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction to the simulation.

        Args:
            interaction: Interaction to add
        """
        self.interactions.append(interaction)
        self._interactions_by_post[interaction.post_id].append(interaction)
        self._interactions_by_user[interaction.user_id].append(interaction)
        self._interactions_by_step[interaction.step].append(interaction)

        # Update user state
        if interaction.user_id in self.user_states:
            user_state = self.user_states[interaction.user_id]
            user_state.mark_seen(interaction.post_id)
            user_state.session_interactions += 1
            user_state.last_active_step = interaction.step
            user_state.add_fatigue(0.02)  # Small fatigue per interaction
            user_state.record_engagement(interaction.step, str(interaction.interaction_type))

            # Track topic exposure and author interactions
            post = self.posts.get(interaction.post_id)
            if post:
                for topic in post.content.topics:
                    user_state.record_topic_exposure(topic)
                user_state.record_author_interaction(post.author_id)

    def add_cascade(self, cascade: Cascade) -> None:
        """Add a cascade to the simulation.

        Args:
            cascade: Cascade to add
        """
        self.cascades[cascade.cascade_id] = cascade

    def add_event(self, event: Event) -> None:
        """Add an event to the simulation.

        Args:
            event: Event to add
        """
        self.events[event.event_id] = event
        if event.is_occurring(self.current_step):
            self.active_events.append(event)

    def get_post(self, post_id: str) -> Post | None:
        """Get a post by ID."""
        return self.posts.get(post_id)

    def get_cascade(self, cascade_id: str) -> Cascade | None:
        """Get a cascade by ID."""
        return self.cascades.get(cascade_id)

    def get_user_state(self, user_id: str) -> UserRuntimeState | None:
        """Get user runtime state."""
        return self.user_states.get(user_id)

    def get_posts_by_author(self, author_id: str) -> list[Post]:
        """Get all posts by an author."""
        post_ids = self._posts_by_author.get(author_id, [])
        return [self.posts[pid] for pid in post_ids if pid in self.posts]

    def get_posts_by_step(self, step: int) -> list[Post]:
        """Get all posts created at a step."""
        post_ids = self._posts_by_step.get(step, [])
        return [self.posts[pid] for pid in post_ids if pid in self.posts]

    def get_recent_posts(self, n_steps: int = 10) -> list[Post]:
        """Get posts from recent steps."""
        posts = []
        for step in range(max(0, self.current_step - n_steps), self.current_step + 1):
            posts.extend(self.get_posts_by_step(step))
        return posts

    def get_active_posts(self) -> list[Post]:
        """Get all active (visible) posts."""
        return [p for p in self.posts.values() if p.is_visible()]

    def get_interactions_for_post(self, post_id: str) -> list[Interaction]:
        """Get all interactions for a post."""
        return self._interactions_by_post.get(post_id, [])

    def get_user_interactions(self, user_id: str) -> list[Interaction]:
        """Get all interactions by a user."""
        return self._interactions_by_user.get(user_id, [])

    def get_interactions_by_step(self, step: int) -> list[Interaction]:
        """Get all interactions at a specific step."""
        return self._interactions_by_step.get(step, [])

    def get_recent_interactions(self, n_steps: int = 10) -> list[Interaction]:
        """Get interactions from recent steps.

        Args:
            n_steps: Number of steps to look back

        Returns:
            List of recent interactions
        """
        interactions = []
        for step in range(max(0, self.current_step - n_steps), self.current_step + 1):
            interactions.extend(self.get_interactions_by_step(step))
        return interactions

    def get_active_users(self, lookback_steps: int = 5) -> list[str]:
        """Get IDs of recently active users."""
        threshold = self.current_step - lookback_steps
        return [
            user_id for user_id, state in self.user_states.items()
            if state.last_active_step >= threshold
        ]

    def get_active_cascades(self) -> list[Cascade]:
        """Get all active cascades."""
        return [c for c in self.cascades.values() if c.is_active]

    def get_combined_event_effect(self):
        """Get combined effect of all active events."""
        from models.event import EventEffect

        if not self.active_events:
            return EventEffect()

        effects = [event.effect for event in self.active_events]
        return EventEffect.combine(effects)

    def record_step_metrics(self, metrics: dict[str, Any]) -> None:
        """Record metrics for current step.

        Args:
            metrics: Dictionary of metric values
        """
        metrics["step"] = self.current_step
        self.step_metrics.append(metrics)

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics of simulation state."""
        return {
            "current_step": self.current_step,
            "total_posts": len(self.posts),
            "total_interactions": len(self.interactions),
            "total_cascades": len(self.cascades),
            "active_cascades": len(self.get_active_cascades()),
            "total_events": len(self.events),
            "active_events": len(self.active_events),
            "active_users_last_5_steps": len(self.get_active_users(5)),
        }

    def save(self, path: str | Path) -> None:
        """Save state to file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            "current_step": self.current_step,
            "posts": {pid: p.to_dict() for pid, p in self.posts.items()},
            "interactions": [i.to_dict() for i in self.interactions],
            "cascades": {cid: c.to_dict() for cid, c in self.cascades.items()},
            "events": {eid: e.to_dict() for eid, e in self.events.items()},
            "user_states": {uid: s.to_dict() for uid, s in self.user_states.items()},
            "step_metrics": self.step_metrics,
        }

        with open(path, "w") as f:
            json.dump(state_data, f)

    @classmethod
    def load_from_dict(cls, state_data: dict[str, Any], users: dict[str, User]) -> "SimulationState":
        """Load state from dictionary.

        Args:
            state_data: Dictionary containing state data
            users: Dictionary of users

        Returns:
            Loaded SimulationState
        """
        state = cls(users)
        state.current_step = state_data["current_step"]

        # Load posts
        for post_data in state_data.get("posts", {}).values():
            post = Post.from_dict(post_data)
            state.add_post(post)

        # Load interactions
        for interaction_data in state_data.get("interactions", []):
            interaction = Interaction.from_dict(interaction_data)
            state.interactions.append(interaction)
            state._interactions_by_post[interaction.post_id].append(interaction)
            state._interactions_by_user[interaction.user_id].append(interaction)
            state._interactions_by_step[interaction.step].append(interaction)

        # Load cascades
        for cascade_data in state_data.get("cascades", {}).values():
            cascade = Cascade.from_dict(cascade_data)
            state.cascades[cascade.cascade_id] = cascade

        # Load events
        for event_data in state_data.get("events", {}).values():
            event = Event.from_dict(event_data)
            state.events[event.event_id] = event

        # Load user states
        for user_state_data in state_data.get("user_states", {}).values():
            user_state = UserRuntimeState.from_dict(user_state_data)
            state.user_states[user_state.user_id] = user_state

        # Load metrics
        state.step_metrics = state_data.get("step_metrics", [])

        # Update active events
        state._update_active_events()

        return state

    @classmethod
    def load(cls, path: str | Path, users: dict[str, User]) -> "SimulationState":
        """Load state from file.

        Args:
            path: Path to load from
            users: Dictionary of users

        Returns:
            Loaded SimulationState
        """
        with open(path) as f:
            state_data = json.load(f)

        return cls.load_from_dict(state_data, users)
