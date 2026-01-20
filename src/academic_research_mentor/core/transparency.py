from __future__ import annotations

"""
Transparency scaffolding with optional persistence and simple event streaming.

- In-memory store of ToolRun and ToolEvent
- Optional JSON persistence controlled by FF_TRANSPARENCY_PERSIST and ARM_RUNLOG_DIR
- Simple in-process pub/sub via add_listener/remove_listener for streaming
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time
import os
import json
from pathlib import Path

from ..session_logging import log_transparency_event


@dataclass
class ToolEvent:
    timestamp_ms: int
    event_type: str  # started | partial_result | final_result | error | ended
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
    """In-memory store with best-effort persistence and streaming hooks."""

    def __init__(self) -> None:
        self._runs: Dict[str, ToolRun] = {}
        self._listeners: List[Any] = []  # callables receiving event dicts
        # Persistence toggles
        self._persist_enabled = os.getenv("FF_TRANSPARENCY_PERSIST", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._persist_dir = Path(
            os.getenv(
                "ARM_RUNLOG_DIR",
                os.path.join(Path.home(), ".cache", "academic-research-mentor", "runs"),
            )
        )

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
        self._emit(
            {
                "type": "started",
                "run_id": run_id,
                "tool": tool_name,
                "timestamp_ms": now,
                "metadata": run.metadata,
            }
        )
        return run

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        evt = ToolEvent(timestamp_ms=int(time.time() * 1000), event_type=event_type, payload=payload)
        run.events.append(evt)
        self._emit(
            {
                "type": event_type,
                "run_id": run_id,
                "tool": run.tool_name,
                "timestamp_ms": evt.timestamp_ms,
                "payload": payload,
            }
        )

    def end_run(self, run_id: str, success: bool, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        run.status = "success" if success else "failure"
        run.ended_ms = int(time.time() * 1000)
        if extra_metadata:
            run.metadata.update(extra_metadata)
        # Persist final run snapshot if enabled
        if self._persist_enabled:
            try:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                out_path = self._persist_dir / f"{run.run_id}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(self._serialize_run(run), f, ensure_ascii=False, indent=2)
            except Exception:
                # Persistence is best-effort; ignore errors
                pass
        self._emit(
            {
                "type": "ended",
                "run_id": run_id,
                "tool": run.tool_name,
                "timestamp_ms": run.ended_ms,
                "success": success,
            }
        )

    def get_run(self, run_id: str) -> Optional[ToolRun]:
        return self._runs.get(run_id)

    def list_runs(self) -> List[ToolRun]:
        # Most-recent first
        return sorted(self._runs.values(), key=lambda r: r.started_ms, reverse=True)

    def clear_runs(self) -> None:
        self._runs.clear()

    # --- Convenience helpers for export ---
    def _serialize_run(self, run: ToolRun) -> Dict[str, Any]:
        return {
            "tool_name": run.tool_name,
            "run_id": run.run_id,
            "status": run.status,
            "started_ms": run.started_ms,
            "ended_ms": run.ended_ms,
            "duration_ms": (run.ended_ms - run.started_ms) if (run.ended_ms and run.started_ms) else None,
            "metadata": dict(run.metadata or {}),
            "events": [asdict(evt) for evt in (run.events or [])],
        }

    def as_dicts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        runs = self.list_runs()
        if limit is not None:
            runs = runs[: max(0, int(limit))]
        return [self._serialize_run(r) for r in runs]

    def persisted_as_dicts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self._persist_enabled:
            return []
        try:
            if not self._persist_dir.exists():
                return []
            files = sorted(self._persist_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if limit is not None:
                files = files[: max(0, int(limit))]
            out: List[Dict[str, Any]] = []
            for fp in files:
                try:
                    with fp.open("r", encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    # --- Listener management (simple in-process pub/sub) ---
    def add_listener(self, callback: Any) -> None:
        try:
            if callback not in self._listeners:
                self._listeners.append(callback)
        except Exception:
            pass

    def remove_listener(self, callback: Any) -> None:
        try:
            if callback in self._listeners:
                self._listeners.remove(callback)
        except Exception:
            pass

    def _emit(self, event: Dict[str, Any]) -> None:
        # Fire-and-forget; guard each listener
        log_transparency_event(event)
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception:
                continue


_global_store: Optional[TransparencyStore] = None


def get_transparency_store() -> TransparencyStore:
    """Return a process-wide transparency store (in-memory)."""
    global _global_store
    if _global_store is None:
        _global_store = TransparencyStore()
    return _global_store
