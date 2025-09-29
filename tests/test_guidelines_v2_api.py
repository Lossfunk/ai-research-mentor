from __future__ import annotations

import os
from typing import Any, Dict


class _StubSearch:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def run(self, q: str) -> str:
        # Return a deterministic payload per query
        return f"{self.payload} | {q}"


class _CapturingCache:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.last_get_key: str | None = None
        self.last_set_key: str | None = None

        class _Cost:
            def __init__(self) -> None:
                self.hits = 0
                self.misses = 0

            def record_cache_hit(self) -> None:
                self.hits += 1

            def record_cache_miss(self) -> None:
                self.misses += 1

            def get_stats(self) -> Dict[str, Any]:
                return {"cache_hits": self.hits, "cache_misses": self.misses}

            def get_cache_hit_rate(self) -> float:
                total = self.hits + self.misses
                return self.hits / total if total else 0.0

        self.cost_tracker = _Cost()

    def get(self, key: str) -> Dict[str, Any] | None:
        self.last_get_key = key
        return self.store.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.last_set_key = key
        self.store[key] = value


def _make_tool(monkeypatch, payload: str = "stub-result"):
    # Ensure FF default is ON for tests
    monkeypatch.setenv("FF_GUIDELINES_V2", "1")

    from academic_research_mentor.tools.guidelines.tool import GuidelinesTool

    t = GuidelinesTool()
    t.initialize()
    # Overwrite network-dependent parts
    t._search_tool = _StubSearch(payload)
    t._cache = _CapturingCache()
    t._cost_tracker = t._cache.cost_tracker
    return t


def test_v2_returns_structured_evidence(monkeypatch):
    tool = _make_tool(monkeypatch)
    out = tool.execute({"query": "mentorship", "topic": "choose research problem"})
    assert "evidence" in out
    assert isinstance(out["evidence"], list)
    # Check evidence item shape
    if out["evidence"]:
        item = out["evidence"][0]
        assert {"evidence_id", "domain", "url", "title", "snippet", "query_used"} <= set(item.keys())


def test_v2_pagination_next_token(monkeypatch):
    tool = _make_tool(monkeypatch)
    first = tool.execute({
        "query": "mentorship",
        "topic": "research taste",
        "page_size": 1
    })
    assert first["pagination"]["has_more"] is True
    tok = first["pagination"]["next_token"]
    assert tok is not None

    second = tool.execute({
        "query": "mentorship",
        "topic": "research taste",
        "page_size": 1,
        "next_token": tok
    })
    assert second["pagination"]["has_more"] in (True, False)
    # If both pages have items, ensure they differ
    if first["evidence"] and second["evidence"]:
        assert first["evidence"][0]["evidence_id"] != second["evidence"][0]["evidence_id"]


def test_cache_key_includes_pagination_and_params(monkeypatch):
    tool = _make_tool(monkeypatch)
    out1 = tool.execute({
        "query": "mentorship",
        "topic": "problem selection",
        "page_size": 2
    })
    key1 = tool._cache.last_set_key
    assert key1 is not None

    out2 = tool.execute({
        "query": "mentorship",
        "topic": "problem selection",
        "page_size": 3
    })
    key2 = tool._cache.last_set_key
    assert key2 is not None
    assert key1 != key2, "Changing page_size should change cache key"

