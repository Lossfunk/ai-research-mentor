"""Research Synthesis using O3 Deep Reasoning

Synthesizes literature search results into coherent research context
using O3's advanced reasoning capabilities.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from .o3_client import get_o3_client
from .citation_extractors import extract_citations_from_arxiv, extract_citations_from_openreview
from .synthesis_helpers import (
    prepare_literature_data, format_literature_for_analysis, parse_synthesis_response,
    fallback_synthesis, empty_synthesis
)
from ..citations import CitationAggregator, CitationFormatter


def synthesize_literature(
    topics: List[str],
    arxiv_results: Dict[str, Any],
    openreview_results: Dict[str, Any],
    research_type: str = "other"
) -> Dict[str, Any]:
    """
    Synthesize literature search results into coherent research context.
    
    Args:
        topics: Research topics extracted from user input
        arxiv_results: Results from arXiv search
        openreview_results: Results from OpenReview search
        research_type: Type of research need
        
    Returns:
        Dictionary containing:
        - summary: str - overall research landscape summary
        - key_papers: List[Dict] - most important papers identified
        - research_gaps: List[str] - potential research gaps
        - trending_topics: List[str] - current trends in the field
        - recommendations: List[str] - actionable recommendations
    """
    o3_client = get_o3_client()
    
    if not o3_client.is_available():
        return fallback_synthesis(topics, arxiv_results, openreview_results)
    
    # Prepare literature data for O3 analysis
    literature_data = prepare_literature_data(arxiv_results, openreview_results)
    
    if not literature_data["papers"]:
        return empty_synthesis(topics)
    
    # Extract and aggregate citations from all sources
    aggregator = CitationAggregator()
    citations = []
    
    # Add arXiv citations
    arxiv_citations = extract_citations_from_arxiv(arxiv_results)
    if arxiv_citations:
        citations.extend(aggregator.add_citations(arxiv_citations, "arxiv"))
    
    # Add OpenReview citations (if any)
    openreview_citations = extract_citations_from_openreview(openreview_results)
    if openreview_citations:
        citations.extend(aggregator.add_citations(openreview_citations, "openreview"))
    
    system_message = """You are an expert research analyst with deep knowledge across multiple academic fields. Your task is to synthesize literature search results into actionable research insights.

Analyze the provided papers and create a comprehensive research landscape overview that would help a researcher understand:
1. Current state of the field
2. Key contributions and breakthrough papers
3. Research gaps and opportunities
4. Emerging trends and directions
5. Practical next steps for someone interested in this area

Be concise but thorough. Focus on actionable insights rather than just summarizing papers."""

    prompt = f"""Analyze this literature search for topics: {', '.join(topics)}

Research Type: {research_type}

Literature Found:
{format_literature_for_analysis(literature_data)}

Please provide a synthesis in the following format:
1. **Field Summary**: 2-3 sentences about the current state
2. **Key Papers**: Identify 3-5 most important papers and why they matter
3. **Research Gaps**: What's missing or underexplored
4. **Trending Topics**: Current hot areas based on recent papers
5. **Recommendations**: Specific next steps for someone interested in this field

Focus on being helpful and actionable for a researcher."""

    try:
        response = o3_client.reason(prompt, system_message)
        if response:
            result = parse_synthesis_response(response, literature_data)
            # Add citation information to the result
            if citations:
                formatter = CitationFormatter()
                result["citations"] = formatter.to_output_block(citations)
                result["citation_stats"] = aggregator.get_citation_stats(citations)
            return result
    except Exception as e:
        print(f"Literature synthesis failed: {e}")
    
    fallback_result = fallback_synthesis(topics, arxiv_results, openreview_results)
    # Add citations to fallback result as well
    if citations:
        formatter = CitationFormatter()
        fallback_result["citations"] = formatter.to_output_block(citations)
        fallback_result["citation_stats"] = aggregator.get_citation_stats(citations)
    return fallback_result



