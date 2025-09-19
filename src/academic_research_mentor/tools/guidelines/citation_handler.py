"""
Citation handling for guidelines tool.

Handles citation conversion and validation for guidelines evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...citations import Citation, CitationFormatter, CitationValidator


class GuidelinesCitationHandler:
    """Handles citation conversion and validation for guidelines tool."""
    
    def __init__(self) -> None:
        self._citation_formatter = CitationFormatter()
        self._citation_validator = CitationValidator()
    
    def evidence_to_citations(self, evidence: List[Dict[str, Any]]) -> List[Citation]:
        """Convert evidence items to Citation objects."""
        citations = []
        for item in evidence:
            url = item.get("url", "")
            title = item.get("title", "Untitled")
            cid = item.get("evidence_id", f"ev_{abs(hash(url or title)) & 0xfffffff:x}")

            citation = Citation(
                id=cid,
                title=title,
                url=url or "https://unknown",
                source=item.get("domain", "unknown"),
                authors=[],  # Guidelines evidence typically doesn't have authors
                year=None,   # Guidelines evidence typically doesn't have years
                venue=item.get("domain", "unknown"),
                snippet=item.get("snippet", "")[:300] or None,
                relevance_score=item.get("relevance_score"),
                extra={
                    "query_used": item.get("query_used"),
                    "retrieved_at": item.get("retrieved_at"),
                    "search_url": item.get("search_url")
                }
            )
            citations.append(citation)

        return citations
    
    def add_citation_metadata(self, result: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add citation validation and formatting to result."""
        citations = self.evidence_to_citations(evidence)
        if citations:
            validation_result = self._citation_validator.validate_citations(citations)
            result["citations"] = self._citation_formatter.to_output_block(citations)
            result["citation_quality"] = {
                "valid": validation_result["valid"],
                "score": validation_result["score"],
                "completeness": validation_result.get("individual_results", [{}])[0].get("completeness", 0) if validation_result.get("individual_results") else 0
            }
        return result
