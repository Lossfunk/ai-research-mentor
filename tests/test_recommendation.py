from __future__ import annotations

import os
from academic_research_mentor.core.recommendation import score_tools
from academic_research_mentor.tools import auto_discover, list_tools
from academic_research_mentor.core.orchestrator import Orchestrator


def test_recommendation_prefers_o3_over_legacy() -> None:
    auto_discover()
    tools = list_tools()
    scored = score_tools("find papers on transformers", tools)
    assert scored, "Expected at least one tool"
    names = [n for n, _s, _r in scored]
    assert names[0] == "o3_search"
    assert any(n.startswith("legacy_") for n in names)


def test_orchestrator_uses_flagged_recommender(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FF_AGENT_RECOMMENDATION", "1")
    auto_discover()
    orch = Orchestrator()
    out = orch.run_task("literature_search", context={"goal": "find papers on transformers"})
    cands = out.get("candidates", [])
    assert cands, "no candidates produced"
    assert cands[0][0] == "o3_search", f"expected o3_search first, got {cands}"
