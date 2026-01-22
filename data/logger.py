"""Structured append-only data logging for simulation events.

Provides efficient logging of:
- Interactions
- Posts
- Cascade events
- User state changes
- Metrics
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from collections import deque
from enum import Enum
import json
import gzip
from datetime import datetime

from models import Post, Interaction, Cascade
from models.enums import InteractionType


class LogEventType(Enum):
    """Types of loggable events."""

    INTERACTION = "interaction"
    POST_CREATED = "post_created"
    POST_UPDATED = "post_updated"
    CASCADE_STARTED = "cascade_started"
    CASCADE_SHARE = "cascade_share"
    CASCADE_ENDED = "cascade_ended"
    USER_STATE_CHANGE = "user_state_change"
    EVENT_STARTED = "event_started"
    EVENT_ENDED = "event_ended"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"


@dataclass
class LogEntry:
    """A single log entry.

    Attributes:
        event_type: Type of event
        step: Simulation step
        timestamp: When entry was created
        data: Event-specific data
        metadata: Additional context
    """

    event_type: LogEventType
    step: int
    timestamp: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "step": self.step,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        return cls(
            event_type=LogEventType(data["event_type"]),
            step=data["step"],
            timestamp=data["timestamp"],
            data=data["data"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "LogEntry":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class SimulationLogger:
    """Structured append-only logger for simulation data.

    Features:
    - Append-only logging for audit trail
    - Efficient batch writing
    - Compressed storage options
    - Query by event type or step range
    """

    def __init__(
        self,
        log_dir: str | Path,
        simulation_id: str | None = None,
        buffer_size: int = 1000,
        use_compression: bool = True,
        rotate_at_entries: int = 100000,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory to store log files
            simulation_id: Unique identifier for this simulation
            buffer_size: Number of entries to buffer before writing
            use_compression: Whether to use gzip compression
            rotate_at_entries: Rotate log file after this many entries
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.simulation_id = simulation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.buffer_size = buffer_size
        self.use_compression = use_compression
        self.rotate_at_entries = rotate_at_entries

        # Write buffer
        self._buffer: deque[LogEntry] = deque(maxlen=buffer_size * 2)
        self._entries_in_current_file = 0
        self._current_file_index = 0

        # Statistics
        self._total_entries = 0
        self._entries_by_type: dict[LogEventType, int] = {t: 0 for t in LogEventType}

    def _get_current_log_path(self) -> Path:
        """Get path to current log file."""
        ext = ".jsonl.gz" if self.use_compression else ".jsonl"
        return self.log_dir / f"{self.simulation_id}_{self._current_file_index:04d}{ext}"

    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        return self._entries_in_current_file >= self.rotate_at_entries

    def _rotate_file(self) -> None:
        """Rotate to a new log file."""
        self._current_file_index += 1
        self._entries_in_current_file = 0

    def _write_entries(self, entries: list[LogEntry]) -> None:
        """Write entries to current log file.

        Args:
            entries: Entries to write
        """
        if not entries:
            return

        log_path = self._get_current_log_path()

        # Check if rotation needed
        if self._should_rotate():
            self._rotate_file()
            log_path = self._get_current_log_path()

        # Write entries
        if self.use_compression:
            mode = "ab"
            with gzip.open(log_path, mode) as f:
                for entry in entries:
                    f.write((entry.to_json() + "\n").encode("utf-8"))
        else:
            with open(log_path, "a") as f:
                for entry in entries:
                    f.write(entry.to_json() + "\n")

        self._entries_in_current_file += len(entries)
        self._total_entries += len(entries)

    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        if self._buffer:
            entries = list(self._buffer)
            self._buffer.clear()
            self._write_entries(entries)

    def log(
        self,
        event_type: LogEventType,
        step: int,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an event.

        Args:
            event_type: Type of event
            step: Current simulation step
            data: Event data
            metadata: Optional metadata
        """
        entry = LogEntry(
            event_type=event_type,
            step=step,
            timestamp=datetime.now().isoformat(),
            data=data,
            metadata=metadata or {},
        )

        self._buffer.append(entry)
        self._entries_by_type[event_type] += 1

        # Flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self._flush_buffer()

    def log_interaction(
        self,
        interaction: Interaction,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log an interaction event.

        Args:
            interaction: Interaction to log
            additional_data: Additional context
        """
        data = interaction.to_dict()
        if additional_data:
            data.update(additional_data)

        self.log(
            LogEventType.INTERACTION,
            interaction.step,
            data,
        )

    def log_post_created(
        self,
        post: Post,
        author_id: str,
        step: int,
    ) -> None:
        """Log a post creation event.

        Args:
            post: Created post
            author_id: Author's user ID
            step: Creation step
        """
        self.log(
            LogEventType.POST_CREATED,
            step,
            {
                "post_id": post.post_id,
                "author_id": author_id,
                "topics": list(post.content.topics),
                "quality_score": post.content.quality_score,
                "controversy_score": post.content.controversy_score,
                "is_misinformation": post.content.is_misinformation,
                "virality_score": post.virality_score,
            },
        )

    def log_cascade_event(
        self,
        cascade: Cascade,
        event_type: LogEventType,
        step: int,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log a cascade-related event.

        Args:
            cascade: Cascade involved
            event_type: Type of cascade event
            step: Current step
            additional_data: Additional context
        """
        data = {
            "cascade_id": cascade.cascade_id,
            "post_id": cascade.post_id,
            "total_shares": cascade.total_shares,
            "total_reach": cascade.total_reach,
            "max_depth": cascade.max_depth,
        }
        if additional_data:
            data.update(additional_data)

        self.log(event_type, step, data)

    def log_metric(
        self,
        step: int,
        metric_name: str,
        metric_value: float | int,
        category: str = "general",
    ) -> None:
        """Log a metric value.

        Args:
            step: Current step
            metric_name: Name of metric
            metric_value: Metric value
            category: Metric category
        """
        self.log(
            LogEventType.METRIC,
            step,
            {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "category": category,
            },
        )

    def log_batch_interactions(
        self,
        interactions: list[Interaction],
    ) -> None:
        """Log multiple interactions efficiently.

        Args:
            interactions: List of interactions to log
        """
        for interaction in interactions:
            self.log_interaction(interaction)

    def flush(self) -> None:
        """Force flush all buffered entries to disk."""
        self._flush_buffer()

    def close(self) -> None:
        """Close logger and flush remaining entries."""
        self._flush_buffer()

    def get_statistics(self) -> dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "simulation_id": self.simulation_id,
            "total_entries": self._total_entries,
            "entries_by_type": {
                t.value: c for t, c in self._entries_by_type.items() if c > 0
            },
            "current_file_index": self._current_file_index,
            "entries_in_current_file": self._entries_in_current_file,
            "buffer_size": len(self._buffer),
        }

    def iterate_entries(
        self,
        event_types: list[LogEventType] | None = None,
        step_range: tuple[int, int] | None = None,
    ) -> Iterator[LogEntry]:
        """Iterate over logged entries with optional filtering.

        Args:
            event_types: Filter by event types (None for all)
            step_range: Filter by step range (min, max) inclusive

        Yields:
            Matching LogEntry objects
        """
        # Flush buffer first
        self._flush_buffer()

        # Find all log files
        pattern = f"{self.simulation_id}_*.jsonl"
        if self.use_compression:
            pattern += ".gz"

        log_files = sorted(self.log_dir.glob(pattern))

        for log_path in log_files:
            if self.use_compression:
                opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
            else:
                opener = lambda p: open(p, "r", encoding="utf-8")

            with opener(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry = LogEntry.from_json(line)

                    # Apply filters
                    if event_types and entry.event_type not in event_types:
                        continue

                    if step_range:
                        if entry.step < step_range[0] or entry.step > step_range[1]:
                            continue

                    yield entry

    def get_entries_by_step(self, step: int) -> list[LogEntry]:
        """Get all entries for a specific step.

        Args:
            step: Step to query

        Returns:
            List of entries for that step
        """
        return list(self.iterate_entries(step_range=(step, step)))

    def get_interaction_counts(
        self,
        step_range: tuple[int, int] | None = None,
    ) -> dict[str, int]:
        """Get interaction counts by type.

        Args:
            step_range: Optional step range filter

        Returns:
            Dictionary of interaction type -> count
        """
        counts: dict[str, int] = {}

        for entry in self.iterate_entries(
            event_types=[LogEventType.INTERACTION],
            step_range=step_range,
        ):
            int_type = entry.data.get("interaction_type", "unknown")
            counts[int_type] = counts.get(int_type, 0) + 1

        return counts


class LogReader:
    """Read-only interface for simulation logs.

    Use this for analysis without risk of modifying logs.
    """

    def __init__(self, log_dir: str | Path, simulation_id: str):
        """Initialize log reader.

        Args:
            log_dir: Directory containing logs
            simulation_id: Simulation ID to read
        """
        self.log_dir = Path(log_dir)
        self.simulation_id = simulation_id

        # Detect compression
        self.use_compression = bool(
            list(self.log_dir.glob(f"{simulation_id}_*.jsonl.gz"))
        )

    def iterate_entries(
        self,
        event_types: list[LogEventType] | None = None,
        step_range: tuple[int, int] | None = None,
    ) -> Iterator[LogEntry]:
        """Iterate over logged entries with optional filtering.

        Args:
            event_types: Filter by event types (None for all)
            step_range: Filter by step range (min, max) inclusive

        Yields:
            Matching LogEntry objects
        """
        pattern = f"{self.simulation_id}_*.jsonl"
        if self.use_compression:
            pattern += ".gz"

        log_files = sorted(self.log_dir.glob(pattern))

        for log_path in log_files:
            if self.use_compression:
                opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
            else:
                opener = lambda p: open(p, "r", encoding="utf-8")

            with opener(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry = LogEntry.from_json(line)

                    if event_types and entry.event_type not in event_types:
                        continue

                    if step_range:
                        if entry.step < step_range[0] or entry.step > step_range[1]:
                            continue

                    yield entry

    def count_entries(self) -> int:
        """Count total entries in logs.

        Returns:
            Total entry count
        """
        return sum(1 for _ in self.iterate_entries())

    def get_step_range(self) -> tuple[int, int] | None:
        """Get the range of steps in the logs.

        Returns:
            Tuple of (min_step, max_step) or None if no entries
        """
        min_step = float("inf")
        max_step = float("-inf")
        found = False

        for entry in self.iterate_entries():
            found = True
            min_step = min(min_step, entry.step)
            max_step = max(max_step, entry.step)

        if not found:
            return None

        return (int(min_step), int(max_step))
