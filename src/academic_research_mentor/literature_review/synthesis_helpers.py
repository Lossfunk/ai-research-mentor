"""
Helper functions for literature synthesis.

Contains utility functions for data preparation and response parsing.
"""

from __future__ import annotations

from typing import List, Dict, Any


def prepare_literature_data(arxiv_results: Dict[str, Any], openreview_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare literature data for O3 analysis."""
    papers = []
    
    # Process arXiv papers
    arxiv_papers = arxiv_results.get("papers", []) if isinstance(arxiv_results, dict) else []
    for paper in arxiv_papers[:10]:  # Limit to prevent token overflow
        if isinstance(paper, dict):
            papers.append({
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "url": paper.get("url", ""),
                "source": "arXiv",
                "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else ""
            })
    
    # Process OpenReview papers
    openreview_threads = openreview_results.get("threads", []) if isinstance(openreview_results, dict) else []
    for thread in openreview_threads[:10]:  # Limit to prevent token overflow
        if isinstance(thread, dict):
            papers.append({
                "title": thread.get("paper_title", ""),
                "year": thread.get("year", ""),
                "venue": thread.get("venue", ""),
                "url": thread.get("urls", {}).get("paper", "") if thread.get("urls") else "",
                "source": "OpenReview"
            })
    
    return {"papers": papers}


def format_literature_for_analysis(literature_data: Dict[str, Any]) -> str:
    """Format literature data for O3 analysis."""
    papers = literature_data.get("papers", [])
    if not papers:
        return "No papers found."
    
    formatted = []
    for i, paper in enumerate(papers[:15], 1):  # Limit for token efficiency
        title = paper.get("title", "Unknown")
        year = paper.get("year", "Unknown")
        source = paper.get("source", "Unknown")
        venue = paper.get("venue", "")
        
        paper_info = f"{i}. {title} ({year}) - {source}"
        if venue:
            paper_info += f" [{venue}]"
        
        # Add abstract if available (arXiv papers)
        abstract = paper.get("abstract", "")
        if abstract:
            paper_info += f"\n   Abstract: {abstract[:200]}..."
        
        formatted.append(paper_info)
    
    return "\n\n".join(formatted)


def parse_synthesis_response(response: str, literature_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse O3 synthesis response into structured format."""
    # Extract key papers mentioned in the response
    papers = literature_data.get("papers", [])
    key_papers = papers[:5]  # Take first 5 as key papers
    
    # Simple parsing - in a more sophisticated version, we could use NLP
    lines = response.split('\n')
    summary = ""
    recommendations = []
    research_gaps = []
    trending_topics = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "field summary" in line.lower() or "summary" in line.lower():
            current_section = "summary"
        elif "recommendation" in line.lower():
            current_section = "recommendations"
        elif "gap" in line.lower():
            current_section = "gaps"
        elif "trend" in line.lower():
            current_section = "trending"
        elif line.startswith('- ') or line.startswith('• '):
            item = line[2:].strip()
            if current_section == "recommendations":
                recommendations.append(item)
            elif current_section == "gaps":
                research_gaps.append(item)
            elif current_section == "trending":
                trending_topics.append(item)
        elif current_section == "summary" and len(line) > 20:
            summary += line + " "
    
    return {
        "summary": summary.strip() or "Research field analysis completed.",
        "key_papers": key_papers,
        "research_gaps": research_gaps or ["Further analysis needed"],
        "trending_topics": trending_topics or ["Contemporary research directions"],
        "recommendations": recommendations or ["Explore recent literature", "Identify specific research questions"],
        "raw_synthesis": response
    }


def fallback_synthesis(topics: List[str], arxiv_results: Dict[str, Any], openreview_results: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback synthesis when O3 is unavailable."""
    papers = []
    
    # Collect papers from both sources
    if isinstance(arxiv_results, dict):
        arxiv_papers = arxiv_results.get("papers", [])
        papers.extend(arxiv_papers[:5])
    
    if isinstance(openreview_results, dict):
        openreview_threads = openreview_results.get("threads", [])
        for thread in openreview_threads[:5]:
            if isinstance(thread, dict):
                papers.append({
                    "title": thread.get("paper_title", ""),
                    "year": thread.get("year", ""),
                    "url": thread.get("urls", {}).get("paper", "") if thread.get("urls") else ""
                })
    
    return {
        "summary": f"Found {len(papers)} papers related to {', '.join(topics)}. Basic literature overview available.",
        "key_papers": papers,
        "research_gaps": ["Detailed analysis requires O3 reasoning"],
        "trending_topics": topics,
        "recommendations": [
            "Review the identified papers",
            "Look for recent work in the field",
            "Identify specific research questions"
        ]
    }


def empty_synthesis(topics: List[str]) -> Dict[str, Any]:
    """Synthesis when no literature is found."""
    return {
        "summary": f"No literature found for topics: {', '.join(topics)}. This might be a very new or niche area.",
        "key_papers": [],
        "research_gaps": ["Potential opportunity for novel research"],
        "trending_topics": [],
        "recommendations": [
            "Try broader search terms",
            "Look for related fields",
            "Consider interdisciplinary approaches"
        ]
    }
