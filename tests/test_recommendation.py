from __future__ import annotations

import os
from academic_research_mentor.core.recommendation import score_tools
from academic_research_mentor.tools import auto_discover, list_tools
from academic_research_mentor.core.orchestrator import Orchestrator


def test_recommendation_prefers_semantic_over_legacy() -> None:
    auto_discover()
    tools = list_tools()
    scored = score_tools("find papers on transformers", tools)
    assert scored, "Expected at least one tool"
    names = [n for n, _s, _r in scored]
    # For natural language queries, searchthearxiv_search should be preferred
    assert names[0] == "searchthearxiv_search"
    assert any(n.startswith("legacy_") for n in names)


def test_orchestrator_uses_flagged_recommender(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FF_AGENT_RECOMMENDATION", "1")
    auto_discover()
    orch = Orchestrator()
    out = orch.run_task("literature_search", context={"goal": "find papers on transformers"})
    cands = out.get("candidates", [])
    assert cands, "no candidates produced"
    # For natural language queries, searchthearxiv_search should be preferred
    assert cands[0][0] == "searchthearxiv_search", f"expected searchthearxiv_search first, got {cands}"


def test_recommendation_prefers_web_search_for_keyword_search() -> None:
    auto_discover()
    tools = list_tools()
    # For keyword-only queries with literature terms, web_search should be preferred
    scored = score_tools("arxiv search transformer", tools)
    assert scored, "Expected at least one tool"
    names = [n for n, _s, _r in scored]
    assert names[0] == "web_search"
