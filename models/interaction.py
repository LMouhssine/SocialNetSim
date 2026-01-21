"""Interaction and cascade models."""

from dataclasses import dataclass, field
from typing import Any

from .enums import InteractionType


@dataclass
class Interaction:
    """Represents a user interaction with content.

    Attributes:
        interaction_id: Unique identifier
        user_id: User who made the interaction
        post_id: Post that was interacted with
        interaction_type: Type of interaction (view, like, share, comment)
        step: Simulation step when interaction occurred
        cascade_id: Associated cascade ID (if part of viral spread)
        source_user_id: User who exposed this user to the content (for cascade tracking)
        metadata: Additional data (e.g., comment text simulation)
    """

    interaction_id: str
    user_id: str
    post_id: str
    interaction_type: InteractionType
    step: int = 0
    cascade_id: str | None = None
    source_user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_engagement(self) -> bool:
        """Check if interaction is engagement (not just view)."""
        return self.interaction_type != InteractionType.VIEW

    @property
    def is_amplification(self) -> bool:
        """Check if interaction amplifies content (share)."""
        return self.interaction_type == InteractionType.SHARE

    def to_dict(self) -> dict[str, Any]:
        """Convert interaction to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "post_id": self.post_id,
            "interaction_type": str(self.interaction_type),
            "step": self.step,
            "cascade_id": self.cascade_id,
            "source_user_id": self.source_user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Interaction":
        """Create interaction from dictionary."""
        return cls(
            interaction_id=data["interaction_id"],
            user_id=data["user_id"],
            post_id=data["post_id"],
            interaction_type=InteractionType(data["interaction_type"]),
            step=data.get("step", 0),
            cascade_id=data.get("cascade_id"),
            source_user_id=data.get("source_user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CascadeNode:
    """Node in a viral cascade tree.

    Attributes:
        user_id: User who shared
        step: Step when they shared
        parent_user_id: User who exposed them to content
        depth: Depth in cascade tree (0 = original author)
        children: Child nodes (users who shared from this user)
    """

    user_id: str
    step: int
    parent_user_id: str | None = None
    depth: int = 0
    children: list["CascadeNode"] = field(default_factory=list)

    def add_child(self, child: "CascadeNode") -> None:
        """Add a child node."""
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def get_subtree_size(self) -> int:
        """Get total size of subtree rooted at this node."""
        return 1 + sum(child.get_subtree_size() for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "user_id": self.user_id,
            "step": self.step,
            "parent_user_id": self.parent_user_id,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CascadeNode":
        """Create node from dictionary."""
        node = cls(
            user_id=data["user_id"],
            step=data["step"],
            parent_user_id=data.get("parent_user_id"),
            depth=data.get("depth", 0),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node


@dataclass
class Cascade:
    """Represents a viral cascade (spread of content through network).

    Attributes:
        cascade_id: Unique identifier
        post_id: Original post that started the cascade
        root: Root node of the cascade tree
        start_step: Step when cascade started
        total_shares: Total number of shares
        total_reach: Total unique users reached
        max_depth: Maximum depth reached
        is_active: Whether cascade is still spreading
        peak_velocity: Maximum shares per step
        reached_users: Set of all users who received the content
        metadata: Additional cascade data
    """

    cascade_id: str
    post_id: str
    root: CascadeNode | None = None
    start_step: int = 0
    total_shares: int = 0
    total_reach: int = 0
    max_depth: int = 0
    is_active: bool = True
    peak_velocity: float = 0.0
    reached_users: set[str] = field(default_factory=set)
    shares_by_step: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def initialize(self, author_id: str, step: int) -> None:
        """Initialize cascade with original author."""
        self.root = CascadeNode(user_id=author_id, step=step, depth=0)
        self.start_step = step
        self.reached_users.add(author_id)
        self.total_reach = 1

    def record_share(self, user_id: str, source_user_id: str, step: int) -> CascadeNode | None:
        """Record a share in the cascade.

        Returns the new node if share was recorded, None otherwise.
        """
        if user_id in self.reached_users:
            return None  # Already in cascade

        # Find parent node
        parent = self._find_node(self.root, source_user_id)
        if parent is None:
            return None

        # Create new node
        new_node = CascadeNode(
            user_id=user_id,
            step=step,
            parent_user_id=source_user_id,
        )
        parent.add_child(new_node)

        # Update cascade stats
        self.total_shares += 1
        self.reached_users.add(user_id)
        self.total_reach = len(self.reached_users)
        self.max_depth = max(self.max_depth, new_node.depth)

        # Track shares by step
        self.shares_by_step[step] = self.shares_by_step.get(step, 0) + 1

        return new_node

    def record_reach(self, user_id: str) -> bool:
        """Record that a user was reached (saw the content).

        Returns True if user was newly reached.
        """
        if user_id in self.reached_users:
            return False
        self.reached_users.add(user_id)
        self.total_reach = len(self.reached_users)
        return True

    def _find_node(self, node: CascadeNode | None, user_id: str) -> CascadeNode | None:
        """Find a node by user ID."""
        if node is None:
            return None
        if node.user_id == user_id:
            return node
        for child in node.children:
            found = self._find_node(child, user_id)
            if found:
                return found
        return None

    def get_velocity(self, current_step: int, window: int = 5) -> float:
        """Calculate recent share velocity."""
        recent_shares = sum(
            self.shares_by_step.get(s, 0)
            for s in range(max(self.start_step, current_step - window), current_step + 1)
        )
        return recent_shares / min(window, current_step - self.start_step + 1)

    def update_peak_velocity(self, current_step: int) -> None:
        """Update peak velocity if current velocity is higher."""
        velocity = self.get_velocity(current_step)
        self.peak_velocity = max(self.peak_velocity, velocity)

    def get_depth_distribution(self) -> dict[int, int]:
        """Get distribution of nodes by depth."""
        distribution: dict[int, int] = {}
        self._count_depths(self.root, distribution)
        return distribution

    def _count_depths(self, node: CascadeNode | None, dist: dict[int, int]) -> None:
        """Recursively count nodes at each depth."""
        if node is None:
            return
        dist[node.depth] = dist.get(node.depth, 0) + 1
        for child in node.children:
            self._count_depths(child, dist)

    def get_branching_factor(self) -> float:
        """Calculate average branching factor."""
        if self.root is None:
            return 0.0
        non_leaf_count, total_children = self._count_branching(self.root)
        if non_leaf_count == 0:
            return 0.0
        return total_children / non_leaf_count

    def _count_branching(self, node: CascadeNode) -> tuple[int, int]:
        """Count non-leaf nodes and their children."""
        if node.is_leaf:
            return 0, 0
        non_leaf = 1
        children = len(node.children)
        for child in node.children:
            nl, c = self._count_branching(child)
            non_leaf += nl
            children += c
        return non_leaf, children

    def deactivate(self) -> None:
        """Mark cascade as no longer active."""
        self.is_active = False

    def to_dict(self) -> dict[str, Any]:
        """Convert cascade to dictionary."""
        return {
            "cascade_id": self.cascade_id,
            "post_id": self.post_id,
            "root": self.root.to_dict() if self.root else None,
            "start_step": self.start_step,
            "total_shares": self.total_shares,
            "total_reach": self.total_reach,
            "max_depth": self.max_depth,
            "is_active": self.is_active,
            "peak_velocity": self.peak_velocity,
            "reached_users": list(self.reached_users),
            "shares_by_step": self.shares_by_step,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cascade":
        """Create cascade from dictionary."""
        cascade = cls(
            cascade_id=data["cascade_id"],
            post_id=data["post_id"],
            start_step=data.get("start_step", 0),
            total_shares=data.get("total_shares", 0),
            total_reach=data.get("total_reach", 0),
            max_depth=data.get("max_depth", 0),
            is_active=data.get("is_active", True),
            peak_velocity=data.get("peak_velocity", 0.0),
            reached_users=set(data.get("reached_users", [])),
            shares_by_step=data.get("shares_by_step", {}),
            metadata=data.get("metadata", {}),
        )
        if data.get("root"):
            cascade.root = CascadeNode.from_dict(data["root"])
        return cascade
