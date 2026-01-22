"""State management with checkpointing and coordination.

Provides clean separation between:
- Simulation state (runtime variables)
- Logged data (append-only history)
- Training data (ML-ready datasets)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import json
import hashlib
import pickle
import gzip
from datetime import datetime

import numpy as np

from engine.state import SimulationState
from models import User


@dataclass
class ReproducibilityInfo:
    """Information needed to reproduce a simulation run.

    Attributes:
        seed: Random seed used
        config_hash: Hash of configuration
        numpy_state: NumPy random state at checkpoint
        step: Simulation step at checkpoint
        timestamp: When checkpoint was created
        version: Schema version for compatibility
    """

    seed: int
    config_hash: str
    numpy_state: dict[str, Any] | None = None
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "config_hash": self.config_hash,
            "numpy_state": self.numpy_state,
            "step": self.step,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReproducibilityInfo":
        """Create from dictionary."""
        return cls(
            seed=data["seed"],
            config_hash=data["config_hash"],
            numpy_state=data.get("numpy_state"),
            step=data.get("step", 0),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", "1.0.0"),
        )


class StateSerializer(Protocol):
    """Protocol for state serialization."""

    def serialize(self, state: SimulationState) -> bytes: ...
    def deserialize(self, data: bytes, users: dict[str, User]) -> SimulationState: ...


class JSONStateSerializer:
    """JSON-based state serializer."""

    def serialize(self, state: SimulationState) -> bytes:
        """Serialize state to JSON bytes."""
        state_data = {
            "current_step": state.current_step,
            "posts": {pid: p.to_dict() for pid, p in state.posts.items()},
            "interactions": [i.to_dict() for i in state.interactions],
            "cascades": {cid: c.to_dict() for cid, c in state.cascades.items()},
            "events": {eid: e.to_dict() for eid, e in state.events.items()},
            "user_states": {uid: s.to_dict() for uid, s in state.user_states.items()},
            "step_metrics": state.step_metrics,
        }
        return json.dumps(state_data).encode("utf-8")

    def deserialize(self, data: bytes, users: dict[str, User]) -> SimulationState:
        """Deserialize state from JSON bytes."""
        state_data = json.loads(data.decode("utf-8"))
        return SimulationState.load_from_dict(state_data, users)


class PickleStateSerializer:
    """Pickle-based state serializer for faster performance."""

    def serialize(self, state: SimulationState) -> bytes:
        """Serialize state to pickle bytes with gzip compression."""
        state_data = {
            "current_step": state.current_step,
            "posts": state.posts,
            "interactions": state.interactions,
            "cascades": state.cascades,
            "events": state.events,
            "user_states": state.user_states,
            "step_metrics": state.step_metrics,
            "_posts_by_author": dict(state._posts_by_author),
            "_posts_by_step": dict(state._posts_by_step),
            "_interactions_by_post": dict(state._interactions_by_post),
            "_interactions_by_user": dict(state._interactions_by_user),
        }
        return gzip.compress(pickle.dumps(state_data))

    def deserialize(self, data: bytes, users: dict[str, User]) -> SimulationState:
        """Deserialize state from pickle bytes."""
        from collections import defaultdict

        state_data = pickle.loads(gzip.decompress(data))
        state = SimulationState(users)
        state.current_step = state_data["current_step"]
        state.posts = state_data["posts"]
        state.interactions = state_data["interactions"]
        state.cascades = state_data["cascades"]
        state.events = state_data["events"]
        state.user_states = state_data["user_states"]
        state.step_metrics = state_data["step_metrics"]
        state._posts_by_author = defaultdict(list, state_data["_posts_by_author"])
        state._posts_by_step = defaultdict(list, state_data["_posts_by_step"])
        state._interactions_by_post = defaultdict(list, state_data["_interactions_by_post"])
        state._interactions_by_user = defaultdict(list, state_data["_interactions_by_user"])
        state._update_active_events()
        return state


@dataclass
class Checkpoint:
    """A simulation checkpoint.

    Attributes:
        step: Step at which checkpoint was created
        state_path: Path to serialized state
        repro_info: Reproducibility information
        metadata: Additional metadata
    """

    step: int
    state_path: Path
    repro_info: ReproducibilityInfo
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "state_path": str(self.state_path),
            "repro_info": self.repro_info.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            step=data["step"],
            state_path=Path(data["state_path"]),
            repro_info=ReproducibilityInfo.from_dict(data["repro_info"]),
            metadata=data.get("metadata", {}),
        )


class StateManager:
    """Coordinates simulation state with checkpointing support.

    Provides:
    - Periodic checkpointing
    - State save/restore
    - Reproducibility tracking
    - Efficient serialization
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_interval: int = 10,
        use_compression: bool = True,
        max_checkpoints: int = 10,
    ):
        """Initialize state manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_interval: Steps between automatic checkpoints
            use_compression: Whether to use pickle with compression
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        # Choose serializer
        self.serializer: StateSerializer = (
            PickleStateSerializer() if use_compression else JSONStateSerializer()
        )

        # Checkpoint history
        self.checkpoints: list[Checkpoint] = []
        self._load_checkpoint_index()

        # Track current state info
        self.current_repro_info: ReproducibilityInfo | None = None

    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self.checkpoints = [Checkpoint.from_dict(c) for c in data.get("checkpoints", [])]

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        data = {"checkpoints": [c.to_dict() for c in self.checkpoints]}
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def initialize(
        self,
        seed: int,
        config: Any,
        rng: np.random.Generator | None = None,
    ) -> ReproducibilityInfo:
        """Initialize reproducibility tracking.

        Args:
            seed: Random seed
            config: Configuration object
            rng: NumPy random generator

        Returns:
            ReproducibilityInfo for tracking
        """
        config_str = json.dumps(config.model_dump(), sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        numpy_state = None
        if rng is not None:
            bit_gen_state = rng.bit_generator.state
            numpy_state = {
                "bit_generator": bit_gen_state["bit_generator"],
                "state": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in bit_gen_state["state"].items()
                },
            }

        self.current_repro_info = ReproducibilityInfo(
            seed=seed,
            config_hash=config_hash,
            numpy_state=numpy_state,
        )
        return self.current_repro_info

    def should_checkpoint(self, step: int) -> bool:
        """Check if a checkpoint should be created at this step.

        Args:
            step: Current simulation step

        Returns:
            True if checkpoint should be created
        """
        return step > 0 and step % self.checkpoint_interval == 0

    def create_checkpoint(
        self,
        state: SimulationState,
        rng: np.random.Generator | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint of current state.

        Args:
            state: Simulation state to checkpoint
            rng: NumPy random generator
            metadata: Additional metadata to store

        Returns:
            Created Checkpoint
        """
        if self.current_repro_info is None:
            raise RuntimeError("StateManager not initialized - call initialize() first")

        # Update reproducibility info
        repro_info = ReproducibilityInfo(
            seed=self.current_repro_info.seed,
            config_hash=self.current_repro_info.config_hash,
            step=state.current_step,
        )

        if rng is not None:
            bit_gen_state = rng.bit_generator.state
            repro_info.numpy_state = {
                "bit_generator": bit_gen_state["bit_generator"],
                "state": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in bit_gen_state["state"].items()
                },
            }

        # Serialize state
        ext = ".pkl.gz" if isinstance(self.serializer, PickleStateSerializer) else ".json"
        state_filename = f"state_step_{state.current_step:06d}{ext}"
        state_path = self.checkpoint_dir / state_filename

        state_bytes = self.serializer.serialize(state)
        with open(state_path, "wb") as f:
            f.write(state_bytes)

        # Create checkpoint
        checkpoint = Checkpoint(
            step=state.current_step,
            state_path=state_path,
            repro_info=repro_info,
            metadata=metadata or {},
        )

        self.checkpoints.append(checkpoint)

        # Prune old checkpoints if needed
        self._prune_checkpoints()

        # Save index
        self._save_checkpoint_index()

        return checkpoint

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if oldest.state_path.exists():
                oldest.state_path.unlink()

    def restore_checkpoint(
        self,
        checkpoint: Checkpoint | int | None,
        users: dict[str, User],
    ) -> tuple[SimulationState, ReproducibilityInfo]:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Checkpoint object, step number, or None for latest
            users: Dictionary of users

        Returns:
            Tuple of (restored state, reproducibility info)
        """
        if checkpoint is None:
            if not self.checkpoints:
                raise ValueError("No checkpoints available")
            checkpoint = self.checkpoints[-1]
        elif isinstance(checkpoint, int):
            matching = [c for c in self.checkpoints if c.step == checkpoint]
            if not matching:
                raise ValueError(f"No checkpoint found for step {checkpoint}")
            checkpoint = matching[-1]

        # Load state
        with open(checkpoint.state_path, "rb") as f:
            state_bytes = f.read()

        state = self.serializer.deserialize(state_bytes, users)

        return state, checkpoint.repro_info

    def get_checkpoints(self) -> list[Checkpoint]:
        """Get list of available checkpoints.

        Returns:
            List of checkpoints ordered by step
        """
        return sorted(self.checkpoints, key=lambda c: c.step)

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint.

        Returns:
            Latest checkpoint or None if no checkpoints
        """
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.step)

    def clear_checkpoints(self) -> None:
        """Remove all checkpoints."""
        for checkpoint in self.checkpoints:
            if checkpoint.state_path.exists():
                checkpoint.state_path.unlink()
        self.checkpoints = []
        self._save_checkpoint_index()

    def verify_reproducibility(
        self,
        state1: SimulationState,
        state2: SimulationState,
    ) -> dict[str, bool]:
        """Verify two states are identical for reproducibility testing.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Dictionary of verification results
        """
        results = {
            "step_match": state1.current_step == state2.current_step,
            "posts_match": len(state1.posts) == len(state2.posts),
            "interactions_match": len(state1.interactions) == len(state2.interactions),
            "cascades_match": len(state1.cascades) == len(state2.cascades),
        }

        # Deep verify posts if counts match
        if results["posts_match"]:
            for pid, post1 in state1.posts.items():
                post2 = state2.posts.get(pid)
                if post2 is None:
                    results["posts_match"] = False
                    break
                if (post1.view_count != post2.view_count or
                    post1.like_count != post2.like_count or
                    post1.share_count != post2.share_count):
                    results["posts_match"] = False
                    break

        results["all_match"] = all(results.values())
        return results
