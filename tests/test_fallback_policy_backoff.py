from __future__ import annotations

import time
from unittest.mock import patch, MagicMock
from typing import Any, Dict, Optional


def test_fallback_policy_backoff_counters(monkeypatch):
    """Test that backoff counters work correctly for degraded tools."""
    from academic_research_mentor.core.fallback_policy import FallbackPolicy, ToolState
    
    policy = FallbackPolicy()
    tool_name = "test_tool"
    
    # Initially tool should be healthy
    assert policy.should_try_tool(tool_name) is True
    assert policy._tool_states.get(tool_name) is None
    
    # Record first failure - should become degraded with backoff
    policy.record_failure(tool_name, "first error")
    assert policy._tool_states[tool_name] == ToolState.DEGRADED
    assert policy._backoff_counts[tool_name] == 1
    assert policy._backoff_start_time[tool_name] is not None
    
    # Immediately try again - should be blocked by backoff
    assert policy.should_try_tool(tool_name) is False
    
    # Fast-forward time beyond backoff period
    policy._backoff_start_time[tool_name] = time.time() - 10  # 10 seconds ago
    assert policy.should_try_tool(tool_name) is True  # Now allowed
    
    # Record another failure - backoff should increase
    policy.record_failure(tool_name, "second error")
    assert policy._backoff_counts[tool_name] == 2
    
    # Record success - should reduce backoff
    policy.record_success(tool_name)
    assert policy._backoff_counts[tool_name] == 1
    
    # Record more successes to fully recover
    policy.record_success(tool_name)
    policy.record_success(tool_name)  # Should reset to healthy
    assert policy._tool_states[tool_name] == ToolState.HEALTHY
    assert policy._backoff_counts[tool_name] == 0


def test_fallback_policy_circuit_breaker_with_backoff(monkeypatch):
    """Test that circuit breaker integrates with backoff counters."""
    from academic_research_mentor.core.fallback_policy import FallbackPolicy, ToolState
    
    policy = FallbackPolicy()
    tool_name = "unstable_tool"
    
    # Simulate multiple failures to trigger circuit breaker
    for i in range(3):
        policy.record_failure(tool_name, f"error {i+1}")
    
    # Should be in circuit open state
    assert policy._tool_states[tool_name] == ToolState.CIRCUIT_OPEN
    assert policy.should_try_tool(tool_name) is False
    
    # Fast-forward beyond circuit breaker timeout
    policy._last_failure_time[tool_name] = time.time() - 400  # 400 seconds ago
    
    # Should now allow testing (degraded mode)
    assert policy.should_try_tool(tool_name) is True
    assert policy._tool_states[tool_name] == ToolState.DEGRADED
    assert policy._backoff_counts[tool_name] == 0  # Backoff reset on circuit breaker recovery


def test_fallback_policy_health_summary(monkeypatch):
    """Test that health summary includes backoff information."""
    from academic_research_mentor.core.fallback_policy import FallbackPolicy, ToolState
    
    policy = FallbackPolicy()
    
    # Add some tools with different states
    policy.record_failure("degraded_tool", "error")
    policy.record_failure("degraded_tool", "another error")  # Backoff count = 2
    
    policy.record_failure("healthy_tool", "error")
    policy.record_success("healthy_tool")  # Recovered
    
    # Trigger circuit breaker for circuit_tool
    for i in range(3):
        policy.record_failure("circuit_tool", f"error {i+1}")
    
    summary = policy.get_tool_health_summary()
    
    # Verify backoff information is included
    assert "backoff_counts" in summary
    assert summary["backoff_counts"]["degraded_tool"] == 2
    assert summary["backoff_counts"]["healthy_tool"] == 0
    # When tool hits circuit breaker, backoff count should be preserved for when it recovers
    assert summary["backoff_counts"]["circuit_tool"] >= 2
    
    # Verify tools in backoff list
    assert "tools_in_backoff" in summary
    assert "degraded_tool" in summary["tools_in_backoff"]
    assert "circuit_tool" in summary["tools_in_backoff"]


def test_execution_engine_surfaces_backoff_status(monkeypatch):
    """Test that execution engine surfaces backoff status in transparency."""
    # Mock environment to avoid API key requirements
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_for_testing")
    
    from academic_research_mentor.core.fallback_policy import get_fallback_policy
    from academic_research_mentor.core.execution_engine import try_tool_with_retries
    from academic_research_mentor.core.transparency import get_transparency_store
    from academic_research_mentor.tools.base_tool import BaseTool
    
    # Set up a degraded tool
    policy = get_fallback_policy()
    tool_name = "test_degraded_tool"
    policy.record_failure(tool_name, "test failure")
    policy.record_failure(tool_name, "another failure")  # Backoff count = 2
    
    # Create a mock tool
    class MockTool(BaseTool):
        name = "test_degraded_tool"
        version = "1.0"
        
        def can_handle(self, task_context=None):
            return True
            
        def execute(self, inputs, context=None):
            return {"results": ["success"]}
        
        def get_metadata(self):
            return {}
    
    # Mock the tools list
    mock_tools = {tool_name: MockTool()}
    
    # Mock the transparency store
    store = MagicMock()
    with patch('academic_research_mentor.core.execution_engine.get_transparency_store', return_value=store):
        with patch('academic_research_mentor.core.execution_engine.print_info') as mock_print:
            try_tool_with_retries(
                mock_tools, tool_name, 1.0, 
                {"query": "test"}, {"goal": "test"}, policy
            )
    
    # Verify that backoff status was surfaced in transparency
    store.start_run.assert_called_once()
    call_args = store.start_run.call_args
    # call_args is a tuple: (args, kwargs) or just args if called positionally
    if call_args:
        args = call_args[0]  # Positional args
        kwargs = call_args[1] if len(call_args) > 1 else {}  # Keyword args
    else:
        kwargs = {}
    
    # Try to get metadata from either args or kwargs
    metadata = None
    if args and len(args) >= 4:  # start_run(self, tool_name, run_id, metadata=None)
        metadata = args[3] if len(args) > 3 else None
    if metadata is None and 'metadata' in kwargs:
        metadata = kwargs['metadata']
    
    assert metadata is not None
    assert metadata['tool_state'].value == 'degraded'
    assert metadata['backoff_count'] == 2
    
    # Print calls verification is optional since the tool execution may fail