"""
Citation aggregation utilities.

Handles collection, deduplication, and merging of citations
from multiple sources across tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from .models import Citation


class CitationAggregator:
    """Aggregates and manages citations from multiple sources."""
    
    def __init__(self) -> None:
        self._seen_urls: Set[str] = set()
        self._seen_titles: Set[str] = set()
    
    def add_citations(self, citations: List[Citation], source: str = "unknown") -> List[Citation]:
        """Add citations from a source, deduplicating against existing ones."""
        new_citations = []
        
        for citation in citations:
            # Check for duplicates by URL and title similarity
            if self._is_duplicate(citation):
                continue
            
            # Mark as seen
            if citation.url:
                self._seen_urls.add(citation.url.lower())
            if citation.title:
                self._seen_titles.add(citation.title.lower().strip())
            
            # Add source tracking
            citation.extra["aggregated_from"] = source
            new_citations.append(citation)
        
        return new_citations
    
    def merge_citations(self, citation_lists: List[List[Citation]]) -> List[Citation]:
        """Merge multiple lists of citations, deduplicating across all."""
        all_citations = []
        
        for i, citations in enumerate(citation_lists):
            source = f"source_{i}"
            new_citations = self.add_citations(citations, source)
            all_citations.extend(new_citations)
        
        return all_citations
    
    def group_by_source(self, citations: List[Citation]) -> Dict[str, List[Citation]]:
        """Group citations by their source."""
        groups = defaultdict(list)
        
        for citation in citations:
            source = citation.source
            groups[source].append(citation)
        
        return dict(groups)
    
    def group_by_year(self, citations: List[Citation]) -> Dict[Optional[int], List[Citation]]:
        """Group citations by publication year."""
        groups = defaultdict(list)
        
        for citation in citations:
            year = citation.year
            groups[year].append(citation)
        
        return dict(groups)
    
    def get_top_citations(self, citations: List[Citation], limit: int = 10, 
                         sort_by: str = "relevance") -> List[Citation]:
        """Get top citations based on sorting criteria."""
        if sort_by == "relevance" and any(c.relevance_score is not None for c in citations):
            sorted_citations = sorted(citations, 
                                    key=lambda c: c.relevance_score or 0, 
                                    reverse=True)
        elif sort_by == "year":
            sorted_citations = sorted(citations, 
                                    key=lambda c: c.year or 0, 
                                    reverse=True)
        else:
            # Default: keep original order
            sorted_citations = citations
        
        return sorted_citations[:limit]
    
    def get_citation_stats(self, citations: List[Citation]) -> Dict[str, Any]:
        """Get statistics about a collection of citations."""
        if not citations:
            return {"total": 0, "sources": 0, "years": 0, "with_doi": 0}
        
        sources = set(c.source for c in citations)
        years = set(c.year for c in citations if c.year)
        with_doi = sum(1 for c in citations if c.doi)
        
        return {
            "total": len(citations),
            "sources": len(sources),
            "years": len(years),
            "with_doi": with_doi,
            "completeness_avg": sum(self._calculate_completeness(c) for c in citations) / len(citations)
        }
    
    def _is_duplicate(self, citation: Citation) -> bool:
        """Check if citation is a duplicate of already seen citations."""
        # Check URL
        if citation.url and citation.url.lower() in self._seen_urls:
            return True
        
        # Check title similarity (simple approach)
        if citation.title:
            title_lower = citation.title.lower().strip()
            for seen_title in self._seen_titles:
                if self._titles_similar(title_lower, seen_title):
                    return True
        
        return False
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar enough to be considered duplicates."""
        if not title1 or not title2:
            return False
        
        # Simple similarity based on common words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    def _calculate_completeness(self, citation: Citation) -> float:
        """Calculate completeness score for a citation."""
        fields = [
            bool(citation.title and len(citation.title.strip()) > 3),
            bool(citation.url),
            bool(citation.authors and len(citation.authors) > 0),
            bool(citation.year and 1900 <= citation.year <= 2030),
            bool(citation.venue),
            bool(citation.doi),
            bool(citation.snippet and len(citation.snippet.strip()) > 10)
        ]
        return (sum(fields) / len(fields)) * 100
