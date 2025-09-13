from __future__ import annotations

from typing import Any, Dict, List


def build_agent_context(intent: Dict[str, Any], synthesis: Dict[str, Any], topics: List[str]) -> str:
    context_parts: List[str] = []

    context_parts.append("=== RESEARCH CONTEXT ===")
    context_parts.append(f"Topics: {', '.join(topics)}")
    context_parts.append(f"Research Type: {intent.get('research_type', 'general')}")
    context_parts.append("")

    summary = synthesis.get("summary", "")
    if summary:
        context_parts.append("FIELD OVERVIEW:")
        context_parts.append(summary)
        context_parts.append("")

    key_papers = synthesis.get("key_papers", [])
    if key_papers:
        context_parts.append("KEY PAPERS FOUND:")
        for i, paper in enumerate(key_papers[:5], 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "")
            venue = paper.get("venue", "")
            source = paper.get("source", "")

            paper_line = f"{i}. {title}"
            if year:
                paper_line += f" ({year})"
            if venue:
                paper_line += f" [{venue}]"
            elif source:
                paper_line += f" [{source}]"

            context_parts.append(paper_line)
        context_parts.append("")

    gaps = synthesis.get("research_gaps", [])
    if gaps:
        context_parts.append("RESEARCH GAPS IDENTIFIED:")
        for gap in gaps[:3]:
            context_parts.append(f"- {gap}")
        context_parts.append("")

    trending = synthesis.get("trending_topics", [])
    if trending:
        context_parts.append(f"TRENDING AREAS: {', '.join(trending[:5])}")
        context_parts.append("")

    recommendations = synthesis.get("recommendations", [])
    if recommendations:
        context_parts.append("RESEARCH RECOMMENDATIONS:")
        for rec in recommendations[:3]:
            context_parts.append(f"- {rec}")
        context_parts.append("")

    context_parts.append("=== END RESEARCH CONTEXT ===")
    context_parts.append("")
    context_parts.append("Use this research context to provide informed mentoring. Ask probing questions based on the field knowledge above.")

    return "\n".join(context_parts)
