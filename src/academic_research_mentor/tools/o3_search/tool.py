from __future__ import annotations

from typing import Any, Dict, Optional

from ..base_tool import BaseTool


class O3SearchTool(BaseTool):
    name = "o3_search"
    version = "0.1"

    def __init__(self) -> None:
        # Placeholder for client wiring (will use literature_review.o3_client later)
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

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute O3-powered literature search using both arXiv and OpenReview.
        
        This combines traditional search with O3 reasoning for enhanced results.
        """
        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"results": [], "note": "empty query"}
        
        try:
            # Import the literature search functions
            from ...mentor_tools import arxiv_search, openreview_fetch
            
            # Perform searches
            search_results = {}
            
            # arXiv search with reasonable defaults
            try:
                from_year = inputs.get("from_year", 2020)
                limit = int(inputs.get("limit", 10))
                search_results["arxiv"] = arxiv_search(query=query, from_year=from_year, limit=limit)
            except Exception as e:
                search_results["arxiv"] = {"papers": [], "note": f"arXiv search failed: {e}"}
            
            # OpenReview search
            try:
                or_limit = int(inputs.get("or_limit", 8))
                search_results["openreview"] = openreview_fetch(query=query, limit=or_limit)
            except Exception as e:
                search_results["openreview"] = {"threads": [], "note": f"OpenReview search failed: {e}"}
            
            # Combine results
            all_papers = []
            arxiv_papers = search_results["arxiv"].get("papers", [])
            openreview_threads = search_results["openreview"].get("threads", [])
            
            # Add source tag to papers for identification
            for paper in arxiv_papers:
                paper["source"] = "arxiv"
                all_papers.append(paper)
            
            for thread in openreview_threads:
                thread["source"] = "openreview"
                all_papers.append(thread)
            
            return {
                "results": all_papers,
                "query": query,
                "search_details": search_results,
                "total_papers": len(all_papers),
                "arxiv_count": len(arxiv_papers),
                "openreview_count": len(openreview_threads),
                "note": "O3-powered literature search completed"
            }
            
        except Exception as e:
            return {
                "results": [],
                "query": query,
                "note": f"O3SearchTool execution failed: {e}",
                "error": str(e)
            }
