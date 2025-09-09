from __future__ import annotations

"""
Transparency scaffolding.

Responsibilities (future):
- Capture tool run metadata and events
- Provide simple in-memory store (replace with persistent backend later)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass
class ToolEvent:
    timestamp_ms: int
    event_type: str  # started | partial_result | final_result | error
    payload: Dict[str, Any]


@dataclass
class ToolRun:
    tool_name: str
    run_id: str
    status: str  # success | failure | running
    started_ms: int
    ended_ms: Optional[int]
    metadata: Dict[str, Any]
    events: List[ToolEvent]


class TransparencyStore:
    """In-memory store for WS1 scaffolding (replace later)."""

    def __init__(self) -> None:
        self._runs: Dict[str, ToolRun] = {}

    def start_run(self, tool_name: str, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> ToolRun:
        now = int(time.time() * 1000)
        run = ToolRun(
            tool_name=tool_name,
            run_id=run_id,
            status="running",
            started_ms=now,
            ended_ms=None,
            metadata=metadata or {},
            events=[],
        )
        self._runs[run_id] = run
        return run

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        evt = ToolEvent(timestamp_ms=int(time.time() * 1000), event_type=event_type, payload=payload)
        run.events.append(evt)

    def end_run(self, run_id: str, success: bool, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        run.status = "success" if success else "failure"
        run.ended_ms = int(time.time() * 1000)
        if extra_metadata:
            run.metadata.update(extra_metadata)

    def get_run(self, run_id: str) -> Optional[ToolRun]:
        return self._runs.get(run_id)

    def list_runs(self) -> List[ToolRun]:
        # Most-recent first
        return sorted(self._runs.values(), key=lambda r: r.started_ms, reverse=True)


_global_store: Optional[TransparencyStore] = None


def get_transparency_store() -> TransparencyStore:
    """Return a process-wide transparency store (in-memory)."""
    global _global_store
    if _global_store is None:
        _global_store = TransparencyStore()
    return _global_store
