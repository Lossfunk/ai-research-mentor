"""
Unified research tool implementation.

Handles combining papers and guidelines with [P#] and [G#] citations.
"""

from __future__ import annotations

from typing import Any

from .tool_helpers import print_agent_reasoning


def unified_research_tool_fn(query: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    """Unified research tool that combines papers and guidelines with [P#] and [G#] citations."""
    begin, end = internal_delimiters or ("", "")
    try:
        from ..core.orchestrator import Orchestrator
        from ..tools import auto_discover
        from ..citations import CitationMerger

        # Ensure tools are discovered
        auto_discover()

        orch = Orchestrator()
        
        # Collect papers from arXiv and O3 search
        paper_results = []
        
        # Try arXiv search
        try:
            arxiv_result = orch.execute_task(
                task="legacy_arxiv_search",
                inputs={"query": query, "limit": 8},
                context={"goal": f"find papers about {query}"}
            )
            if arxiv_result["execution"]["executed"] and arxiv_result["results"]:
                paper_results.append(arxiv_result["results"])
        except Exception:
            pass
        
        # Try web search
        try:
            web_result = orch.execute_task(
                task="web_search",
                inputs={"query": query, "limit": 8},
                context={"goal": f"find papers about {query}"}
            )
            if web_result["execution"]["executed"] and web_result["results"]:
                paper_results.append(web_result["results"])
        except Exception:
            pass
        
        # Get guidelines
        guidelines_result = None
        try:
            guidelines_result = orch.execute_task(
                task="research_guidelines",
                inputs={
                    "query": query,
                    "topic": query,
                    "response_format": "concise",
                    "page_size": 20,
                    "mode": "fast",
                },
                context={"goal": f"research mentorship guidance about {query}"}
            )
        except Exception:
            pass

        # Extract papers and guidelines
        merger = CitationMerger()
        papers = merger.extract_papers_from_tool_results(paper_results)
        guidelines = []
        
        if guidelines_result and guidelines_result["execution"]["executed"] and guidelines_result["results"]:
            tool_result = guidelines_result["results"]
            evidence_items = tool_result.get("evidence") or []
            guideline_items = tool_result.get("retrieved_guidelines", [])
            guidelines = evidence_items + guideline_items

        if not papers and not guidelines:
            return f"{begin}No relevant papers or guidelines found for this query. Try rephrasing or asking more specific questions.{end}" if begin or end else "No relevant papers or guidelines found for this query. Try rephrasing or asking more specific questions."

        # Merge citations with [P#] and [G#] IDs
        merged_result = merger.merge_citations(
            papers=papers,
            guidelines=guidelines,
            max_papers=10,
            max_guidelines=20
        )

        reasoning_block = merged_result["context"]
        # Print as Agent's reasoning panel for TUI differentiation
        print_agent_reasoning(reasoning_block)
        return f"{begin}{reasoning_block}{end}" if begin or end else reasoning_block

    except Exception as e:
        return f"Error in unified research search: {str(e)}"
