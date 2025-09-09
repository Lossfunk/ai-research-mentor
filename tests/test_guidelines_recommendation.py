from __future__ import annotations

from academic_research_mentor.tools import auto_discover, list_tools
from academic_research_mentor.core.recommendation import score_tools


def test_guidelines_preferred_for_mentorship_queries() -> None:
    auto_discover()
    tools = list_tools()
    goal = "phd guidance and research methodology advice"
    scored = score_tools(goal, tools)
    assert scored, "Expected at least one tool"
    top = scored[0][0]
    assert top in {"research_guidelines", "o3_search"}
    # Should strongly prefer guidelines; allow o3_search if guidelines not available
    assert any(name == "research_guidelines" for name, _s, _r in scored)
