"""
Citation validation utilities.

Provides quality checks and validation for citations to ensure
integrity and completeness across tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .models import Citation


class CitationValidator:
    """Validates citation quality and completeness."""
    
    def __init__(self) -> None:
        self.doi_pattern = re.compile(r'10\.\d+/[^\s]+')
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def validate_citation(self, citation: Citation) -> Dict[str, Any]:
        """Validate a single citation and return quality metrics."""
        issues: List[str] = []
        score = 100.0
        
        # Check required fields
        if not citation.title or len(citation.title.strip()) < 3:
            issues.append("Title too short or missing")
            score -= 30
        
        if not citation.url or not self._is_valid_url(citation.url):
            issues.append("Invalid or missing URL")
            score -= 25
        
        if not citation.authors or len(citation.authors) == 0:
            issues.append("No authors specified")
            score -= 15
        
        if not citation.year or citation.year < 1900 or citation.year > 2030:
            issues.append("Invalid or missing year")
            score -= 10
        
        # Check optional but valuable fields
        if not citation.venue:
            issues.append("No venue specified")
            score -= 5
        
        if not citation.doi and not self._extract_doi_from_url(citation.url):
            # DOI is nice to have but not critical for validation
            issues.append("No DOI available")
            score -= 2
        
        if not citation.snippet or len(citation.snippet.strip()) < 10:
            issues.append("No meaningful snippet")
            score -= 5
        
        # Check for duplicates in title/URL
        if citation.title and len(citation.title) > 10:
            # Basic duplicate detection could be added here
            pass
        
        return {
            "valid": score >= 70,
            "score": max(0, score),
            "issues": issues,
            "completeness": self._calculate_completeness(citation)
        }
    
    def validate_citations(self, citations: List[Citation]) -> Dict[str, Any]:
        """Validate a collection of citations."""
        if not citations:
            return {"valid": False, "score": 0, "issues": ["No citations provided"]}
        
        individual_results = [self.validate_citation(c) for c in citations]
        valid_count = sum(1 for r in individual_results if r["valid"])
        avg_score = sum(r["score"] for r in individual_results) / len(individual_results)
        
        all_issues = []
        for i, result in enumerate(individual_results):
            if result["issues"]:
                all_issues.extend([f"Citation {i+1}: {issue}" for issue in result["issues"]])
        
        return {
            "valid": valid_count == len(citations),
            "score": avg_score,
            "valid_count": valid_count,
            "total_count": len(citations),
            "issues": all_issues,
            "individual_results": individual_results
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        if not url:
            return False
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and bool(self.url_pattern.match(url))
        except Exception:
            return False
    
    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL if present."""
        if not url:
            return None
        match = self.doi_pattern.search(url)
        return match.group(0) if match else None
    
    def _calculate_completeness(self, citation: Citation) -> float:
        """Calculate completeness score (0-100) for a citation."""
        fields = [
            bool(citation.title and len(citation.title.strip()) > 3),
            bool(citation.url and self._is_valid_url(citation.url)),
            bool(citation.authors and len(citation.authors) > 0),
            bool(citation.year and 1900 <= citation.year <= 2030),
            bool(citation.venue),
            bool(citation.doi or self._extract_doi_from_url(citation.url)),
            bool(citation.snippet and len(citation.snippet.strip()) > 10)
        ]
        return (sum(fields) / len(fields)) * 100
