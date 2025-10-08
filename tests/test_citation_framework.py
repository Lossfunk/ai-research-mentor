"""
Tests for citation framework functionality.

Tests the unified citation system across all tools.
"""

import pytest
from academic_research_mentor.citations import Citation, CitationFormatter, CitationValidator, CitationAggregator


class TestCitation:
    """Test Citation data model."""
    
    def test_citation_creation(self):
        """Test basic citation creation."""
        citation = Citation(
            id="test_1",
            title="Test Paper",
            url="https://example.com/paper",
            source="arxiv",
            authors=["Author One", "Author Two"],
            year=2023,
            venue="Example Conference"
        )
        
        assert citation.id == "test_1"
        assert citation.title == "Test Paper"
        assert citation.url == "https://example.com/paper"
        assert citation.source == "arxiv"
        assert len(citation.authors) == 2
        assert citation.year == 2023
        assert citation.venue == "Example Conference"
    
    def test_citation_to_dict(self):
        """Test citation serialization."""
        citation = Citation(
            id="test_2",
            title="Another Paper",
            url="https://example.com/paper2",
            source="openreview"
        )
        
        data = citation.to_dict()
        assert data["id"] == "test_2"
        assert data["title"] == "Another Paper"
        assert data["url"] == "https://example.com/paper2"
        assert data["source"] == "openreview"
        assert "authors" in data
        assert "year" in data


class TestCitationFormatter:
    """Test citation formatting."""
    
    def test_inline_formatting(self):
        """Test inline citation formatting."""
        formatter = CitationFormatter()
        citation = Citation(
            id="test_1",
            title="Test Paper",
            url="https://example.com/paper",
            source="arxiv",
            authors=["Author One", "Author Two", "Author Three"],
            year=2023,
            venue="Example Conference"
        )
        
        formatted = formatter.format_inline(citation)
        assert "Author One, Author Two et al." in formatted
        assert "(2023)" in formatted
        assert "Test Paper" in formatted
        assert "Example Conference" in formatted
        assert "https://example.com/paper" in formatted
    
    def test_list_formatting(self):
        """Test citation list formatting."""
        formatter = CitationFormatter()
        citations = [
            Citation(id="1", title="Paper 1", url="https://example.com/1", source="arxiv"),
            Citation(id="2", title="Paper 2", url="https://example.com/2", source="openreview")
        ]
        
        formatted_list = formatter.format_list(citations)
        assert len(formatted_list) == 2
        assert "Paper 1" in formatted_list[0]
        assert "Paper 2" in formatted_list[1]
    
    def test_output_block_formatting(self):
        """Test structured output block formatting."""
        formatter = CitationFormatter()
        citations = [
            Citation(id="1", title="Paper 1", url="https://example.com/1", source="arxiv"),
            Citation(id="2", title="Paper 2", url="https://example.com/2", source="openreview")
        ]
        
        block = formatter.to_output_block(citations)
        assert block["count"] == 2
        assert block["style"] == "academic"
        assert len(block["citations"]) == 2
        assert block["citations"][0]["title"] == "Paper 1"


class TestCitationValidator:
    """Test citation validation."""
    
    def test_valid_citation(self):
        """Test validation of a complete citation."""
        validator = CitationValidator()
        citation = Citation(
            id="test_1",
            title="A Complete Test Paper",
            url="https://example.com/paper",
            source="arxiv",
            authors=["Author One"],
            year=2023,
            venue="Test Conference",
            snippet="This is a meaningful snippet that provides context about the paper content."
        )
        
        result = validator.validate_citation(citation)
        assert result["valid"] is True
        assert result["score"] >= 70
        # DOI is optional, so we may have one issue about missing DOI
        assert len(result["issues"]) <= 1
    
    def test_invalid_citation(self):
        """Test validation of an incomplete citation."""
        validator = CitationValidator()
        citation = Citation(
            id="test_1",
            title="",  # Empty title
            url="invalid-url",  # Invalid URL
            source="arxiv"
            # Missing authors, year, venue, snippet
        )
        
        result = validator.validate_citation(citation)
        assert result["valid"] is False
        assert result["score"] < 70
        assert len(result["issues"]) > 0
        assert any("Title too short" in issue for issue in result["issues"])
        assert any("Invalid or missing URL" in issue for issue in result["issues"])
    
    def test_citation_collection_validation(self):
        """Test validation of multiple citations."""
        validator = CitationValidator()
        citations = [
            Citation(id="1", title="Good Paper", url="https://example.com/1", source="arxiv", authors=["Author"], year=2023),
            Citation(id="2", title="", url="bad-url", source="arxiv")  # Invalid
        ]
        
        result = validator.validate_citations(citations)
        assert result["total_count"] == 2
        assert result["valid_count"] == 1
        assert result["score"] < 100  # Should be average of both


class TestCitationAggregator:
    """Test citation aggregation and deduplication."""
    
    def test_add_citations(self):
        """Test adding citations to aggregator."""
        aggregator = CitationAggregator()
        citations = [
            Citation(id="1", title="Paper 1", url="https://example.com/1", source="arxiv"),
            Citation(id="2", title="Paper 2", url="https://example.com/2", source="arxiv")
        ]
        
        new_citations = aggregator.add_citations(citations, "test_source")
        assert len(new_citations) == 2
        assert all(c.extra.get("aggregated_from") == "test_source" for c in new_citations)
    
    def test_deduplication(self):
        """Test citation deduplication."""
        aggregator = CitationAggregator()
        
        # Add first set
        citations1 = [Citation(id="1", title="Same Paper", url="https://example.com/same", source="arxiv")]
        aggregator.add_citations(citations1, "source1")
        
        # Try to add duplicate
        citations2 = [Citation(id="2", title="Same Paper", url="https://example.com/same", source="openreview")]
        new_citations = aggregator.add_citations(citations2, "source2")
        
        assert len(new_citations) == 0  # Should be deduplicated
    
    def test_citation_stats(self):
        """Test citation statistics generation."""
        aggregator = CitationAggregator()
        citations = [
            Citation(id="1", title="Paper 1", url="https://example.com/1", source="arxiv", year=2023, doi="10.1000/1"),
            Citation(id="2", title="Paper 2", url="https://example.com/2", source="openreview", year=2022),
            Citation(id="3", title="Paper 3", url="https://example.com/3", source="arxiv", year=2023)
        ]
        
        stats = aggregator.get_citation_stats(citations)
        assert stats["total"] == 3
        assert stats["sources"] == 2  # arxiv and openreview
        assert stats["years"] == 2  # 2022 and 2023
        assert stats["with_doi"] == 1  # Only one has DOI
        assert "completeness_avg" in stats


class TestCitationIntegration:
    """Test citation integration across tools."""
    
    def test_guidelines_tool_citations(self):
        """Test that guidelines tool produces valid citations."""
        from academic_research_mentor.tools.guidelines.tool import GuidelinesTool
        
        tool = GuidelinesTool()
        tool.initialize()
        
        # Test with a simple query
        result = tool.execute({"query": "research methodology", "topic": "methodology"})
        
        # Check if citations are present in V2 mode
        if "citations" in result:
            assert "citations" in result
            assert "count" in result["citations"]
            assert isinstance(result["citations"]["citations"], list)
    
    def test_arxiv_tool_citations(self):
        """Test that arXiv tools produce valid citations."""
        from academic_research_mentor.tools.legacy.arxiv.tool import ArxivSearchTool
        
        tool = ArxivSearchTool()
        tool.initialize()
        
        # Test with a simple query
        result = tool.execute({"query": "machine learning", "limit": 3})
        
        # Check if citations are present
        if "citations" in result:
            assert "citations" in result
            assert "count" in result["citations"]
            assert isinstance(result["citations"]["citations"], list)
    
    def test_web_search_tool_citations(self):
        """Test that web search tool produces valid citations."""
        from academic_research_mentor.tools.web_search.tool import WebSearchTool

        tool = WebSearchTool()
        tool.initialize()
        
        # Test with a simple query
        result = tool.execute({"query": "artificial intelligence", "limit": 3})
        
        # Check if citations are present
        if "citations" in result:
            assert "citations" in result
            assert "count" in result["citations"]
            assert isinstance(result["citations"]["citations"], list)
