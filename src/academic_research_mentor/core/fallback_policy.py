from __future__ import annotations

"""
Fallback policy engine for robust tool execution (WS3 extension).

Handles retry logic, circuit breakers, and degraded modes when tools fail.
Keeps orchestrator.py under 200 LOC by separating policy logic.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
from enum import Enum


class ToolState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"


class FallbackPolicy:
    """Manages tool execution fallback strategies and circuit breakers."""
    
    def __init__(self) -> None:
        self._tool_states: Dict[str, ToolState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_time: Dict[str, float] = {}
        self._backoff_counts: Dict[str, int] = {}  # Track consecutive backoff attempts
        self._backoff_start_time: Dict[str, float] = {}  # When backoff period started
        self._circuit_breaker_threshold = 3
        self._circuit_breaker_timeout = 300  # 5 minutes
        self._backoff_base_delay = 5.0  # 5 second base backoff
        self._backoff_max_delay = 60.0  # 1 minute maximum backoff
        self._backoff_reset_threshold = 3  # Successes needed to reset backoff
        
    def should_try_tool(self, tool_name: str) -> bool:
        """Check if tool should be attempted based on circuit breaker and backoff state."""
        state = self._tool_states.get(tool_name, ToolState.HEALTHY)
        
        # Check circuit breaker first
        if state == ToolState.CIRCUIT_OPEN:
            # Check if timeout has passed
            last_failure = self._last_failure_time.get(tool_name, 0)
            if time.time() - last_failure > self._circuit_breaker_timeout:
                # Reset to degraded for testing
                self._tool_states[tool_name] = ToolState.DEGRADED
                self._failure_counts[tool_name] = max(0, self._failure_counts.get(tool_name, 0) - 1)
                # Reset backoff on circuit breaker reset
                self._backoff_counts[tool_name] = 0
                return True
            return False
        
        # Check backoff state for degraded tools
        if state == ToolState.DEGRADED and tool_name in self._backoff_counts:
            backoff_count = self._backoff_counts[tool_name]
            if backoff_count > 0:
                # Calculate backoff delay
                backoff_delay = min(
                    self._backoff_base_delay * (2 ** (backoff_count - 1)),
                    self._backoff_max_delay
                )
                
                backoff_start = self._backoff_start_time.get(tool_name, 0)
                if time.time() - backoff_start < backoff_delay:
                    return False  # Still in backoff period
            
        return True
    
    def record_success(self, tool_name: str) -> None:
        """Record successful tool execution."""
        if tool_name in self._failure_counts:
            # Gradually recover from failures
            self._failure_counts[tool_name] = max(0, self._failure_counts[tool_name] - 1)
            if self._failure_counts[tool_name] == 0:
                self._tool_states[tool_name] = ToolState.HEALTHY
                # Reset backoff counters on full recovery
                self._backoff_counts[tool_name] = 0
            elif tool_name in self._backoff_counts and self._backoff_counts[tool_name] > 0:
                # Reduce backoff counter on success during degraded mode
                self._backoff_counts[tool_name] = max(0, self._backoff_counts[tool_name] - 1)
    
    def record_failure(self, tool_name: str, error: str) -> None:
        """Record tool failure and update circuit breaker state."""
        self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1
        self._last_failure_time[tool_name] = time.time()
        
        if self._failure_counts[tool_name] >= self._circuit_breaker_threshold:
            self._tool_states[tool_name] = ToolState.CIRCUIT_OPEN
        else:
            self._tool_states[tool_name] = ToolState.DEGRADED
            # Start or increment backoff counter for degraded tools
            self._backoff_counts[tool_name] = self._backoff_counts.get(tool_name, 0) + 1
            self._backoff_start_time[tool_name] = time.time()
    
    def get_execution_strategy(self, candidates: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Determine execution strategy based on tool health and candidates."""
        available_candidates = []
        blocked_tools = []
        
        for tool_name, score in candidates:
            if self.should_try_tool(tool_name):
                available_candidates.append((tool_name, score))
            else:
                blocked_tools.append(tool_name)
        
        if not available_candidates:
            return {
                "strategy": "all_blocked",
                "candidates": [],
                "blocked_tools": blocked_tools,
                "fallback_mode": "degraded"
            }
        
        # Simple strategy: try top tool, then next best
        primary = available_candidates[0]
        fallback = available_candidates[1] if len(available_candidates) > 1 else None
        
        return {
            "strategy": "primary_with_fallback" if fallback else "primary_only",
            "primary": primary,
            "fallback": fallback,
            "blocked_tools": blocked_tools,
            "total_candidates": len(available_candidates)
        }
    
    def should_retry(self, tool_name: str, attempt: int, error: str) -> Tuple[bool, float]:
        """Determine if tool should be retried and with what delay."""
        max_retries = 2
        
        if attempt >= max_retries:
            return False, 0.0
            
        # Exponential backoff with jitter
        base_delay = 1.0 * (2 ** attempt)
        jitter = 0.1 * attempt
        delay = base_delay + jitter
        
        # Don't retry certain error types
        non_retryable = ["authentication", "authorization", "invalid_input", "quota_exceeded"]
        if any(err_type in error.lower() for err_type in non_retryable):
            return False, 0.0
            
        return True, min(delay, 10.0)  # Cap at 10 seconds
    
    def get_tool_health_summary(self) -> Dict[str, Any]:
        """Return summary of tool health for monitoring."""
        return {
            "tool_states": dict(self._tool_states),
            "failure_counts": dict(self._failure_counts),
            "backoff_counts": dict(self._backoff_counts),
            "circuit_breakers_open": [
                name for name, state in self._tool_states.items() 
                if state == ToolState.CIRCUIT_OPEN
            ],
            "tools_in_backoff": [
                name for name, count in self._backoff_counts.items() 
                if count > 0
            ]
        }


# Global instance for orchestrator use
_fallback_policy = FallbackPolicy()


def get_fallback_policy() -> FallbackPolicy:
    """Get the global fallback policy instance."""
    return _fallback_policy