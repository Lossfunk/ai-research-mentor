"""
Citation merger for combining papers and guidelines with stable IDs.

Provides unified citation context with [P1..Pn] for papers and [G1..Gm] for guidelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .models import Citation
from .aggregator import CitationAggregator


class CitationMerger:
    """Merges citations from multiple sources with stable ID assignment."""
    
    def __init__(self) -> None:
        self._aggregator = CitationAggregator()
        self._paper_counter = 0
        self._guideline_counter = 0
    
    def merge_citations(self, 
                       papers: List[Dict[str, Any]], 
                       guidelines: List[Dict[str, Any]],
                       max_papers: int = 10,
                       max_guidelines: int = 20) -> Dict[str, Any]:
        """Merge papers and guidelines with stable [P#] and [G#] IDs.
        
        Args:
            papers: List of paper dictionaries from arXiv/O3 tools
            guidelines: List of guideline dictionaries from guidelines tool
            max_papers: Maximum number of papers to include
            max_guidelines: Maximum number of guidelines to include
            
        Returns:
            Dictionary with merged citations and formatted context
        """
        # Convert papers to Citation objects
        paper_citations = []
        for paper in papers[:max_papers]:
            if not isinstance(paper, dict):
                continue
                
            url = str(paper.get("url", "")).strip()
            title = str(paper.get("title", "")).strip() or "Untitled"
            cid = f"paper_{abs(hash(url or title)) & 0xfffffff:x}"
            
            citation = Citation(
                id=cid,
                title=title,
                url=url or "https://arxiv.org",
                source=paper.get("source", "arxiv"),
                authors=[str(a) for a in paper.get("authors", []) if a],
                year=paper.get("year"),
                venue=paper.get("venue", "arXiv"),
                snippet=(paper.get("summary") or paper.get("abstract", ""))[:300] or None,
                extra={
                    "source_type": "paper",
                    "original_data": paper
                }
            )
            paper_citations.append(citation)
        
        # Convert guidelines to Citation objects
        guideline_citations = []
        for guideline in guidelines[:max_guidelines]:
            if not isinstance(guideline, dict):
                continue
                
            url = str(guideline.get("url", "")).strip()
            title = str(guideline.get("title", "")).strip() or "Research Guidance"
            cid = f"guideline_{abs(hash(url or title)) & 0xfffffff:x}"
            
            citation = Citation(
                id=cid,
                title=title,
                url=url or "https://unknown",
                source=guideline.get("domain", "research_guidance"),
                authors=[],  # Guidelines typically don't have authors
                year=None,   # Guidelines typically don't have years
                venue=guideline.get("domain", "research_guidance"),
                snippet=(guideline.get("snippet") or guideline.get("content", ""))[:300] or None,
                extra={
                    "source_type": "guideline",
                    "original_data": guideline
                }
            )
            guideline_citations.append(citation)
        
        # Assign stable IDs
        merged_papers = []
        for i, citation in enumerate(paper_citations, 1):
            citation.extra["stable_id"] = f"P{i}"
            merged_papers.append(citation)
        
        merged_guidelines = []
        for i, citation in enumerate(guideline_citations, 1):
            citation.extra["stable_id"] = f"G{i}"
            merged_guidelines.append(citation)
        
        # Generate formatted context for agent
        context_lines = []
        citations_list = []
        
        if merged_papers:
            context_lines.append(f"Found {len(merged_papers)} relevant papers:")
            for citation in merged_papers:
                stable_id = citation.extra["stable_id"]
                title = citation.title
                year = f" ({citation.year})" if citation.year else ""
                url = citation.url
                snippet = citation.snippet or ""
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                
                context_lines.append(f"PAPER [{stable_id}] {title}{year}")
                if snippet:
                    context_lines.append(f"Snippet: {snippet}")
                context_lines.append(f"Link: {url}")
                context_lines.append("---")
                
                citations_list.append(f"[{stable_id}] {title} — {url}")
        
        if merged_guidelines:
            context_lines.append(f"Found {len(merged_guidelines)} research guidelines:")
            for citation in merged_guidelines:
                stable_id = citation.extra["stable_id"]
                title = citation.title
                domain = citation.source
                snippet = citation.snippet or ""
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                
                context_lines.append(f"GUIDELINE [{stable_id}] {title} — {domain}")
                if snippet:
                    context_lines.append(f"Snippet: {snippet}")
                if citation.url != "https://unknown":
                    context_lines.append(f"Link: {citation.url}")
                    citations_list.append(f"[{stable_id}] {title} — {citation.url}")
                context_lines.append("---")
        
        # Add soft citation instructions
        context_lines.append(
            "\nUse these sources to ground your response. "
            "Embed inline bracketed citations [P#] for papers and [G#] for guidelines immediately after specific claims. "
            "Prefer [P#] citations when discussing specific papers or research directions. "
            "Include [G#] citations for methodology and general research advice. "
            "At the end, include a 'Citations' section listing [ID] Title — URL."
        )
        
        # Add soft enforcement
        context_lines.append(
            "Soft guidance: When making recommendations, try to cite relevant papers [P#] when available. "
            "If no relevant papers exist, use guidelines [G#] for methodology advice. "
            "Focus on being helpful rather than rigid citation requirements."
        )
        
        # Add replicate and extend suggestions if papers exist
        if merged_papers:
            context_lines.append("\nReplicate and extend suggestions:")
            for citation in merged_papers[:3]:  # Top 3 papers
                stable_id = citation.extra["stable_id"]
                title = citation.title
                context_lines.append(f"• Based on {title} [{stable_id}]: Consider replicating their key experiments, then extending with additional datasets or evaluation metrics")
            context_lines.append("---")
        
        if citations_list:
            context_lines.append("\nCitations:")
            for citation in citations_list:
                context_lines.append(f"• {citation}")
        
        return {
            "context": "\n".join(context_lines),
            "papers": merged_papers,
            "guidelines": merged_guidelines,
            "citations": citations_list,
            "paper_count": len(merged_papers),
            "guideline_count": len(merged_guidelines)
        }
    
    def extract_papers_from_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract papers from various tool results."""
        papers = []
        for result in tool_results:
            if not isinstance(result, dict):
                continue
                
            # Handle different tool result formats
            if "papers" in result:
                papers.extend(result["papers"])
            elif "results" in result:
                papers.extend(result["results"])
            elif "citations" in result and "citations" in result["citations"]:
                # Handle structured citations
                for citation in result["citations"]["citations"]:
                    if citation.get("source") in ["arxiv", "openreview"]:
                        papers.append(citation)
        
        return papers
    
    def extract_guidelines_from_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract guidelines from tool results."""
        guidelines = []
        for result in tool_results:
            if not isinstance(result, dict):
                continue
                
            # Handle different guideline formats
            if "evidence" in result:
                guidelines.extend(result["evidence"])
            elif "retrieved_guidelines" in result:
                guidelines.extend(result["retrieved_guidelines"])
            elif "citations" in result and "citations" in result["citations"]:
                # Handle structured citations
                for citation in result["citations"]["citations"]:
                    if citation.get("source") not in ["arxiv", "openreview"]:
                        guidelines.append(citation)
        
        return guidelines
