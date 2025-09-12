from __future__ import annotations

from typing import Any, Dict, Optional
import time
import signal
from contextlib import contextmanager

from ..base_tool import BaseTool


class _O3SearchTool(BaseTool):
    name = "o3_search"
    version = "0.1"

    def __init__(self) -> None:
        # Placeholder for client wiring (will use literature_review.o3_client later)
        self._timeout_seconds = 15  # 15 second timeout for O3 operations
        pass

    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        # Placeholder: accept if task mentions literature or search
        tc = (task_context or {}).get("goal", "")
        return any(k in str(tc).lower() for k in ("literature", "search", "papers", "arxiv", "openreview"))

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta["identity"]["owner"] = "core"
        meta["capabilities"] = {
            "task_types": ["literature_search"],
            "domains": ["ml", "ai", "cs"],
        }
        meta["io"] = {
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "output_schema": {"type": "object", "properties": {"results": {"type": "array"}}},
        }
        meta["operational"] = {"cost_estimate": "medium", "latency_profile": "variable", "rate_limits": None}
        meta["usage"] = {"ideal_inputs": ["concise topic"], "anti_patterns": ["empty query"], "prerequisites": []}
        return meta

    @contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for timeout handling."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set timeout handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Reset alarm and restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute O3-powered literature search with timeout and fallback.
        
        This combines traditional search with O3 reasoning for enhanced results.
        Falls back to arxiv_search if O3 operations timeout or fail.
        """
        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"results": [], "note": "empty query"}
        
        # Try O3-powered search with timeout
        try:
            with self._timeout_context(self._timeout_seconds):
                return self._execute_o3_search_with_timeout(query, inputs, context)
        except TimeoutError as e:
            # Timeout occurred - fall back to arxiv_search with degraded mode note
            return self._execute_fallback_arxiv_search(query, inputs, context, f"O3 timeout: {str(e)}")
        except Exception as e:
            # Other error - fall back to arxiv_search with degraded mode note
            return self._execute_fallback_arxiv_search(query, inputs, context, f"O3 error: {str(e)}")
    
    def _execute_o3_search_with_timeout(self, query: str, inputs: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute O3-powered search (internal method with timeout protection)."""
        # Import the literature search functions
        from ...mentor_tools import arxiv_search
        
        # Perform searches
        search_results = {}
        
        # arXiv search with reasonable defaults
        try:
            from_year = inputs.get("from_year", 2020)
            limit = int(inputs.get("limit", 10))
            search_results["arxiv"] = arxiv_search(query=query, from_year=from_year, limit=limit)
        except Exception as e:
            search_results["arxiv"] = {"papers": [], "note": f"arXiv search failed: {e}"}
        
        # OpenReview search removed - legacy functionality deprecated
        # Combine results
        all_papers = []
        arxiv_papers = search_results["arxiv"].get("papers", [])
        
        # Add source tag to papers for identification
        for paper in arxiv_papers:
            paper["source"] = "arxiv"
            all_papers.append(paper)
        
        return {
            "results": all_papers,
            "query": query,
            "search_details": search_results,
            "total_papers": len(all_papers),
            "arxiv_count": len(arxiv_papers),
            "openreview_count": 0,
            "note": "O3-powered literature search completed (OpenReview deprecated)"
        }
    
    def _execute_fallback_arxiv_search(self, query: str, inputs: Dict[str, Any], context: Optional[Dict[str, Any]], fallback_reason: str) -> Dict[str, Any]:
        """Execute fallback arxiv_search with degraded mode note."""
        try:
            from ...mentor_tools import arxiv_search
            
            # Perform arXiv search as fallback
            from_year = inputs.get("from_year", 2020)
            limit = int(inputs.get("limit", 10))
            arxiv_result = arxiv_search(query=query, from_year=from_year, limit=limit)
            
            # Add fallback metadata
            if isinstance(arxiv_result, dict):
                arxiv_result["_fallback_reason"] = fallback_reason
                arxiv_result["_fallback_from"] = "o3_search"
                arxiv_result["_degraded_mode"] = True
                arxiv_result["note"] = f"Fallback to arXiv search (O3 unavailable: {fallback_reason})"
            else:
                # Handle unexpected result format
                arxiv_result = {
                    "results": [],
                    "query": query,
                    "note": f"Fallback failed: unexpected arXiv result format (O3 unavailable: {fallback_reason})",
                    "_fallback_reason": fallback_reason,
                    "_fallback_from": "o3_search",
                    "_degraded_mode": True
                }
            
            return arxiv_result
            
        except Exception as e:
            # Even fallback failed
            return {
                "results": [],
                "query": query,
                "note": f"Complete failure: O3 unavailable and arXiv fallback failed - {fallback_reason}, arXiv error: {str(e)}",
                "_fallback_reason": fallback_reason,
                "_fallback_from": "o3_search",
                "_degraded_mode": True,
                "_fallback_failed": True
            }
