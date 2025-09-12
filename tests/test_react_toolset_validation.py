from __future__ import annotations


def test_react_toolset_excludes_openreview_includes_arxiv(monkeypatch):
    """Verify that the ReAct toolset excludes openreview and includes arxiv_search."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.runtime import get_langchain_tools
    
    tools = get_langchain_tools()
    
    # Extract tool names
    tool_names = [tool.name for tool in tools]
    
    # Verify arxiv_search is included (required tool)
    assert "arxiv_search" in tool_names, "arxiv_search should be in the ReAct toolset"
    
    # Verify openreview is NOT included
    assert "openreview" not in tool_names, "openreview should not be in the ReAct toolset"
    assert "openreview_search" not in tool_names, "openreview_search should not be in the ReAct toolset"
    
    # Verify other expected tools are included
    expected_tools = [
        "o3_search",
        "research_guidelines", 
        "searchthearxiv_search"
    ]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"{expected_tool} should be in the ReAct toolset"
    
    # Verify no tool contains "openreview" in its name
    openreview_tools = [name for name in tool_names if "openreview" in name.lower()]
    assert len(openreview_tools) == 0, f"Found tools containing 'openreview': {openreview_tools}"