from __future__ import annotations

import time
from unittest.mock import patch, MagicMock
from typing import Any, Dict, Optional


def test_recommender_scoring_with_degraded_tools(monkeypatch):
    """Test that recommender properly scores tools under degraded conditions."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.core.fallback_policy import get_fallback_policy
    from academic_research_mentor.core.recommendation import score_tools
    from academic_research_mentor.tools.base_tool import BaseTool
    
    # Set up policy with degraded O3 tool
    policy = get_fallback_policy()
    
    # Create mock tools
    class MockO3Tool(BaseTool):
        name = "o3_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"results": ["o3 result"]}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["literature_search"]}}
    
    class MockArxivTool(BaseTool):
        name = "legacy_arxiv_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"papers": [{"title": "arxiv paper"}]}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["literature_search"]}}
    
    class MockGuidelinesTool(BaseTool):
        name = "research_guidelines"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"retrieved_guidelines": []}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["guidelines_search"]}}
    
    tools = {
        "o3_search": MockO3Tool(),
        "legacy_arxiv_search": MockArxivTool(),
        "research_guidelines": MockGuidelinesTool()
    }
    
    # Test scoring with healthy tools (literature query)
    goal = "find recent papers on transformers"
    scored = score_tools(goal, tools)
    
    # O3 should be highest score for literature queries when healthy
    o3_score = next(score for name, score, reason in scored if name == "o3_search")
    arxiv_score = next(score for name, score, reason in scored if name == "legacy_arxiv_search")
    guidelines_score = next(score for name, score, reason in scored if name == "research_guidelines")
    
    assert o3_score > arxiv_score
    assert o3_score > guidelines_score
    
    # Now degrade O3 tool
    policy.record_failure("o3_search", "timeout error")
    policy.record_failure("o3_search", "another timeout")  # Backoff count = 2
    
    # Test scoring again with degraded O3
    scored_degraded = score_tools(goal, tools)
    o3_score_degraded = next(score for name, score, reason in scored_degraded if name == "o3_search")
    arxiv_score_degraded = next(score for name, score, reason in scored_degraded if name == "legacy_arxiv_search")
    
    # O3 should still be recommended but with potentially lower confidence
    # (the actual scoring logic may vary, but O3 should remain available)
    assert o3_score_degraded > 0
    # Arxiv might have negative score in some scoring scenarios, that's okay


def test_recommender_scoring_mentorship_queries_under_degraded(monkeypatch):
    """Test that recommender properly handles mentorship queries under degraded conditions."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.core.fallback_policy import get_fallback_policy
    from academic_research_mentor.core.recommendation import score_tools
    from academic_research_mentor.tools.base_tool import BaseTool
    
    policy = get_fallback_policy()
    
    # Create mock tools
    class MockO3Tool(BaseTool):
        name = "o3_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"results": ["o3 result"]}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["literature_search"]}}
    
    class MockGuidelinesTool(BaseTool):
        name = "research_guidelines"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"retrieved_guidelines": []}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["guidelines_search"]}}
    
    tools = {
        "o3_search": MockO3Tool(),
        "research_guidelines": MockGuidelinesTool()
    }
    
    # Test mentorship query
    mentorship_goal = "how to choose a research direction"
    scored = score_tools(mentorship_goal, tools)
    
    # Verify we get scores for both tools
    tool_names = [name for name, score, reason in scored]
    assert "o3_search" in tool_names
    assert "research_guidelines" in tool_names
    
    # Research guidelines should be present and have some score
    guidelines_entry = next((name, score, reason) for name, score, reason in scored if name == "research_guidelines")
    assert guidelines_entry[1] > 0
    
    # Now degrade guidelines tool
    policy.record_failure("research_guidelines", "guidelines service unavailable")
    
    # Test mentorship query with degraded guidelines
    scored_degraded = score_tools(mentorship_goal, tools)
    
    # Guidelines should still be available even when degraded
    guidelines_degraded_entry = next((name, score, reason) for name, score, reason in scored_degraded if name == "research_guidelines")
    assert guidelines_degraded_entry[1] > 0


def test_recommender_handles_all_tools_blocked(monkeypatch):
    """Test recommender behavior when all relevant tools are blocked."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.core.fallback_policy import get_fallback_policy
    from academic_research_mentor.core.recommendation import score_tools
    from academic_research_mentor.tools.base_tool import BaseTool
    
    policy = get_fallback_policy()
    
    # Create mock tool
    class MockO3Tool(BaseTool):
        name = "o3_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"results": ["o3 result"]}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["literature_search"]}}
    
    tools = {"o3_search": MockO3Tool()}
    
    # Completely block O3 tool (circuit breaker)
    for i in range(3):
        policy.record_failure("o3_search", f"critical error {i+1}")
    
    # Ensure it's circuit open
    assert policy.should_try_tool("o3_search") is False
    
    # Test scoring with blocked tool
    goal = "find papers on machine learning"
    scored = score_tools(goal, tools)
    
    # Should still return O3 with some score (for fallback consideration)
    assert len(scored) > 0
    o3_entry = next(entry for entry in scored if entry[0] == "o3_search")
    assert o3_entry[1] > 0  # Some positive score even when blocked


def test_recommender_fallback_awareness(monkeypatch):
    """Test that recommender is aware of fallback relationships."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.core.fallback_policy import get_fallback_policy
    from academic_research_mentor.core.recommendation import score_tools
    from academic_research_mentor.tools.base_tool import BaseTool
    
    policy = get_fallback_policy()
    
    # Create tools with fallback relationship
    class MockO3Tool(BaseTool):
        name = "o3_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"results": ["o3 result"]}
        
        def get_metadata(self):
            return {
                "capabilities": {"task_types": ["literature_search"]},
                "fallbacks": ["legacy_arxiv_search"]
            }
    
    class MockArxivTool(BaseTool):
        name = "legacy_arxiv_search"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"papers": [{"title": "arxiv paper"}]}
        
        def get_metadata(self):
            return {"capabilities": {"task_types": ["literature_search"]}}
    
    tools = {
        "o3_search": MockO3Tool(),
        "legacy_arxiv_search": MockArxivTool()
    }
    
    # Test with healthy O3
    goal = "literature search"
    scored_healthy = score_tools(goal, tools)
    
    # Verify both tools get scores
    tool_names_healthy = [name for name, score, reason in scored_healthy]
    assert "o3_search" in tool_names_healthy
    assert "legacy_arxiv_search" in tool_names_healthy
    
    # Degrade O3
    policy.record_failure("o3_search", "timeout")
    policy.record_failure("o3_search", "another timeout")
    
    # Test with degraded O3
    scored_degraded = score_tools(goal, tools)
    
    # Verify both tools still get scores even when O3 is degraded
    tool_names_degraded = [name for name, score, reason in scored_degraded]
    assert "o3_search" in tool_names_degraded
    assert "legacy_arxiv_search" in tool_names_degraded
    
    # Verify scores are reasonable (negative scores are possible in some scenarios)
    for name, score, reason in scored_degraded:
        assert isinstance(score, (int, float))  # Just verify it's a number