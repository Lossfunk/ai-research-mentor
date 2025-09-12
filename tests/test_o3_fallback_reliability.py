from __future__ import annotations

import time
from unittest.mock import patch, MagicMock
from typing import Any, Dict, Optional


def test_o3_timeout_falls_back_to_arxiv(monkeypatch):
    """Test that O3 timeout triggers fallback to arxiv_search with degraded mode note."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.tools.o3_search.tool import _O3SearchTool
    
    # Create O3 tool 
    tool = _O3SearchTool()
    
    # Mock the timeout context to raise TimeoutError
    mock_timeout_context = MagicMock()
    mock_timeout_context.__enter__ = MagicMock()
    mock_timeout_context.__exit__ = MagicMock(side_effect=lambda exc_type, exc_val, exc_tb: None)
    
    with patch.object(tool, '_timeout_context', return_value=mock_timeout_context):
        with patch.object(tool, '_execute_o3_search_with_timeout', side_effect=TimeoutError("Operation timed out")):
            with patch.object(tool, '_execute_fallback_arxiv_search') as mock_fallback:
                # Setup mock fallback result
                mock_fallback.return_value = {
                    "papers": [
                        {"title": "Test Paper 1", "url": "http://arxiv.org/abs/1"},
                        {"title": "Test Paper 2", "url": "http://arxiv.org/abs/2"}
                    ],
                    "_fallback_reason": "O3 timeout: Operation timed out after 15 seconds",
                    "_fallback_from": "o3_search",
                    "_degraded_mode": True,
                    "note": "Fallback to arXiv search (O3 unavailable: O3 timeout: Operation timed out after 15 seconds)"
                }
                
                # Execute O3 search - should timeout and fall back to arxiv
                result = tool.execute(
                    inputs={"query": "machine learning", "limit": 5},
                    context={"goal": "find papers on machine learning"}
                )
    
    # Verify fallback metadata is present
    assert result["_fallback_reason"] is not None
    assert "timeout" in result["_fallback_reason"].lower()
    assert result["_fallback_from"] == "o3_search"
    assert result["_degraded_mode"] is True
    
    # Verify fallback to arxiv results
    assert len(result["papers"]) == 2
    assert result["papers"][0]["title"] == "Test Paper 1"
    
    # Verify note indicates fallback
    assert "Fallback to arXiv search" in result["note"]
    assert "O3 unavailable" in result["note"]


def test_o3_exception_falls_back_to_arxiv(monkeypatch):
    """Test that O3 exception triggers fallback to arxiv_search with degraded mode note."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.tools.o3_search.tool import _O3SearchTool
    
    tool = _O3SearchTool()
    
    # Mock the timeout context
    mock_timeout_context = MagicMock()
    mock_timeout_context.__enter__ = MagicMock()
    mock_timeout_context.__exit__ = MagicMock(side_effect=lambda exc_type, exc_val, exc_tb: None)
    
    with patch.object(tool, '_timeout_context', return_value=mock_timeout_context):
        with patch.object(tool, '_execute_o3_search_with_timeout', side_effect=Exception("O3 processing error")):
            with patch.object(tool, '_execute_fallback_arxiv_search') as mock_fallback:
                # Setup mock fallback result
                mock_fallback.return_value = {
                    "papers": [{"title": "Exception Test Paper", "url": "http://arxiv.org/abs/123"}],
                    "_fallback_reason": "O3 error: O3 processing error",
                    "_fallback_from": "o3_search",
                    "_degraded_mode": True,
                    "note": "Fallback to arXiv search (O3 unavailable: O3 error: O3 processing error)"
                }
                
                # Execute O3 search with exception
                result = tool.execute(
                    inputs={"query": "test query", "limit": 3},
                    context={"goal": "test"}
                )
    
    # Verify fallback occurred
    assert "_fallback_reason" in result
    assert result["_fallback_from"] == "o3_search"
    assert result["_degraded_mode"] is True
    
    # Verify the fallback result contains arxiv data
    assert len(result["papers"]) == 1
    assert result["papers"][0]["title"] == "Exception Test Paper"


def test_o3_fallback_failure_handling(monkeypatch):
    """Test handling when both O3 and arxiv fallback fail."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.tools.o3_search.tool import _O3SearchTool
    
    tool = _O3SearchTool()
    tool._timeout_seconds = 0.1
    
    # Mock arxiv_search to raise an exception
    with patch('academic_research_mentor.mentor_tools.arxiv_search', side_effect=Exception("arXiv failed")):
        result = tool.execute(
            inputs={"query": "test", "limit": 5},
            context={"goal": "test"}
        )
    
    # Verify complete failure is handled gracefully
    assert result["_fallback_failed"] is True
    assert "_fallback_reason" in result
    assert "Complete failure" in result["note"]
    assert "arXiv fallback failed" in result["note"]
    assert len(result["results"]) == 0


def test_o3_successful_execution_no_fallback(monkeypatch):
    """Test that successful O3 execution doesn't trigger fallback."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.tools.o3_search.tool import _O3SearchTool
    
    tool = _O3SearchTool()
    tool._timeout_seconds = 30  # Long enough to avoid timeout
    
    # Mock successful arxiv_search (this simulates O3 working)
    mock_arxiv_result = {
        "papers": [{"title": "Successful O3 Paper", "url": "http://arxiv.org/abs/success"}],
        "total": 1
    }
    
    with patch('academic_research_mentor.mentor_tools.arxiv_search', return_value=mock_arxiv_result):
        result = tool.execute(
            inputs={"query": "successful query", "limit": 1},
            context={"goal": "test"}
        )
    
    # Verify no fallback occurred
    assert "_fallback_reason" not in result
    assert "_fallback_from" not in result
    assert "_degraded_mode" not in result
    
    # Verify normal O3 result
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Successful O3 Paper"
    assert "O3-powered literature search completed" in result["note"]