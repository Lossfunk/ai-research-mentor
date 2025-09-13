from __future__ import annotations

from typing import Any, Dict, List

from .o3_client import get_o3_client


def llm_only_overview(user_input: str, topics: List[str], research_type: str) -> Dict[str, Any]:
    o3 = get_o3_client()
    if not o3.is_available():
        return {
            "summary": f"High-level overview for topics: {', '.join(topics)}.",
            "key_papers": [],
            "research_gaps": ["Lack of grounded references due to offline retrieval"],
            "trending_topics": topics[:5],
            "recommendations": [
                "Consider re-running with network access to fetch citations.",
            ],
        }

    system_message = (
        "You are an expert research mentor. No web search is available. "
        "Produce a concise literature-style overview grounded in general knowledge only. "
        "Avoid fabricating specific citation metadata or URLs."
    )
    prompt = (
        f"User research request: {user_input}\n\n"
        f"Topics (may be partial): {', '.join(topics)}\n"
        f"Research Type: {research_type}\n\n"
        "Please provide: (1) Field summary (2-3 sentences), (2) Potential research gaps, "
        "(3) Trending sub-areas, (4) 3 concrete next steps. Do not invent paper titles or links."
    )
    try:
        content = o3.reason(prompt, system_message) or ""
    except Exception:
        content = ""

    summary = content.strip()[:800] if content else "General high-level overview produced."
    return {
        "summary": summary,
        "key_papers": [],
        "research_gaps": [],
        "trending_topics": topics[:5],
        "recommendations": [
            "Refine topics and try again",
            "Broaden/adjust keywords",
            "Add venue constraints (e.g., ICLR, NeurIPS)",
        ],
    }
