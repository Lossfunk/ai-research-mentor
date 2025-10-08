from __future__ import annotations

from academic_research_mentor.tools import auto_discover, list_tools, get_tool


def test_auto_discover_registers_web_search_tool() -> None:
    auto_discover()
    registry = list_tools()
    assert "web_search" in registry
    t = get_tool("web_search")
    assert t is not None
    # Check minimal metadata presence
    meta = t.get_metadata()
    assert meta.get("identity", {}).get("name") == "web_search"
