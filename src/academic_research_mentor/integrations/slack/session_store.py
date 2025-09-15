from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...chat_logger import ChatLogger


@dataclass
class SlackSession:
    team_id: str
    channel_id: str
    thread_ts: str
    agent: Any | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    chat_logger: ChatLogger = field(default_factory=ChatLogger)
    last_active_ts: float = field(default_factory=lambda: time.time())


class InMemorySessionStore:
    """Simple in-memory session store with TTL-based pruning.

    Keyed by f"{team}:{channel}:{thread_ts}". Not process-safe.
    """

    def __init__(self, ttl_seconds: int = 60 * 60 * 8, max_sessions: int = 512) -> None:
        self._store: Dict[str, SlackSession] = {}
        self._ttl = ttl_seconds
        self._max = max_sessions

    @staticmethod
    def make_key(team_id: str, channel_id: str, thread_ts: str) -> str:
        return f"{team_id}:{channel_id}:{thread_ts}"

    def get(self, team_id: str, channel_id: str, thread_ts: str) -> Optional[SlackSession]:
        key = self.make_key(team_id, channel_id, thread_ts)
        sess = self._store.get(key)
        if not sess:
            return None
        # Expire
        if time.time() - sess.last_active_ts > self._ttl:
            try:
                del self._store[key]
            except Exception:
                pass
            return None
        return sess

    def create_or_get(self, team_id: str, channel_id: str, thread_ts: str) -> SlackSession:
        key = self.make_key(team_id, channel_id, thread_ts)
        sess = self._store.get(key)
        if sess is None:
            # Prune if over capacity (simple LRU-ish based on last_active_ts)
            if len(self._store) >= self._max:
                try:
                    oldest_key = min(self._store, key=lambda k: self._store[k].last_active_ts)
                    del self._store[oldest_key]
                except Exception:
                    self._store.clear()
            sess = SlackSession(team_id=team_id, channel_id=channel_id, thread_ts=thread_ts)
            self._store[key] = sess
        sess.last_active_ts = time.time()
        return sess

    def touch(self, team_id: str, channel_id: str, thread_ts: str) -> None:
        sess = self.get(team_id, channel_id, thread_ts)
        if sess:
            sess.last_active_ts = time.time()

    def clear(self, team_id: str, channel_id: str, thread_ts: str) -> None:
        key = self.make_key(team_id, channel_id, thread_ts)
        try:
            del self._store[key]
        except Exception:
            pass


# Global dev store instance
GLOBAL_SESSION_STORE = InMemorySessionStore()


