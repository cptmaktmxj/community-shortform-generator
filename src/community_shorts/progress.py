"""Structured pipeline progress events and terminal rendering."""

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal, Protocol, TextIO


ProgressStage = Literal["ingest", "curate", "generate"]
ProgressStatus = Literal["running", "completed", "failed"]
_STAGE_LABELS: dict[ProgressStage, str] = {
    "ingest": "수집",
    "curate": "선별",
    "generate": "생성",
}
_STATUS_COLORS: dict[ProgressStatus, str] = {
    "running": "\x1b[36m",
    "completed": "\x1b[32m",
    "failed": "\x1b[31m",
}


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One user-visible pipeline state transition."""

    stage: ProgressStage
    status: ProgressStatus
    message: str
    current: int | None = None
    total: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete positions and empty progress messages."""

        if not self.message.strip():
            raise ValueError("progress message must not be blank")
        if (self.current is None) != (self.total is None):
            raise ValueError("progress current and total must be supplied together")
        if self.current is not None and not 0 <= self.current <= self.total:
            raise ValueError("progress position must be inside zero and total")


class ProgressSink(Protocol):
    """Boundary used by pipeline services to publish progress events."""

    def emit(self, event: ProgressEvent) -> None:
        """Publish one event."""


class NullProgress:
    """Discard progress for programmatic service usage."""

    def emit(self, event: ProgressEvent) -> None:
        """Ignore one event."""

        del event


class ConsoleProgress:
    """Render compact Korean progress lines with colors only on a real terminal."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._stream = stream or sys.stdout
        self._clock = clock

    def emit(self, event: ProgressEvent) -> None:
        """Write and flush one consistently formatted progress line."""

        timestamp = self._clock().strftime("%H:%M:%S")
        position = (
            f" [{event.current}/{event.total}]"
            if event.current is not None and event.total is not None
            else ""
        )
        detail = f" · {event.detail}" if event.detail else ""
        line = (
            f"[{timestamp}] [{_STAGE_LABELS[event.stage]}]{position} "
            f"{event.message}{detail}"
        )
        use_color = bool(getattr(self._stream, "isatty", lambda: False)())
        if use_color:
            line = f"{_STATUS_COLORS[event.status]}{line}\x1b[0m"
        self._stream.write(line + "\n")
        self._stream.flush()
