"""
Citation extraction utilities for literature review.

Handles extraction of Citation objects from various literature sources.
"""

from __future__ import annotations

from typing import List, Dict, Any
from ..citations import Citation


def extract_citations_from_arxiv(arxiv_results: Dict[str, Any]) -> List[Citation]:
    """Extract Citation objects from arXiv results."""
    citations = []
    papers = arxiv_results.get("papers", []) if isinstance(arxiv_results, dict) else []
    
    for paper in papers:
        if not isinstance(paper, dict):
            continue
            
        url = str(paper.get("url", "")).strip()
        title = str(paper.get("title", "")).strip() or "Untitled"
        cid = f"arxiv_{abs(hash(url or title)) & 0xfffffff:x}"
        
        citation = Citation(
            id=cid,
            title=title,
            url=url or "https://arxiv.org",
            source="arxiv",
            authors=[str(a) for a in paper.get("authors", []) if a],
            year=paper.get("year"),
            venue=paper.get("venue", "arXiv"),
            snippet=(paper.get("summary") or paper.get("abstract", ""))[:300] or None,
            extra={
                "source_type": "arxiv_paper",
                "original_data": paper
            }
        )
        citations.append(citation)
    
    return citations


def extract_citations_from_openreview(openreview_results: Dict[str, Any]) -> List[Citation]:
    """Extract Citation objects from OpenReview results."""
    citations = []
    threads = openreview_results.get("threads", []) if isinstance(openreview_results, dict) else []
    
    for thread in threads:
        if not isinstance(thread, dict):
            continue
            
        url = str(thread.get("urls", {}).get("paper", "") if thread.get("urls") else "").strip()
        title = str(thread.get("paper_title", "")).strip() or "Untitled"
        cid = f"openreview_{abs(hash(url or title)) & 0xfffffff:x}"
        
        citation = Citation(
            id=cid,
            title=title,
            url=url or "https://openreview.net",
            source="openreview",
            authors=[str(a) for a in thread.get("authors", []) if a],
            year=thread.get("year"),
            venue=thread.get("venue", "OpenReview"),
            snippet=(thread.get("abstract", ""))[:300] or None,
            extra={
                "source_type": "openreview_thread",
                "original_data": thread
            }
        )
        citations.append(citation)
    
    return citations
