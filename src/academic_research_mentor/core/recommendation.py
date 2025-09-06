from __future__ import annotations

"""
Recommendation scoring for tool selection (WS3).

Kept under 200 LOC; simple heuristic combining:
- can_handle: filter
- base priority: o3 > others > legacy
- metadata: cost, reliability (when available)
- keyword match on goal
"""

from typing import Any, Dict, List, Tuple
import re

PRIMARY_NAMES = {"o3_search"}
LEGACY_PREFIX = "legacy_"


def _keyword_match_score(goal: str, tool_name: str) -> float:
    g = goal.lower()
    score = 0.0
    if tool_name in g:
        score += 0.5
    
    # Literature search keywords
    for kw in ("literature", "papers", "search", "review", "arxiv", "openreview"):
        if kw in g:
            score += 0.2
    
    # Research guidelines keywords
    guidelines_keywords = (
        "methodology", "advice", "guidance", "mentor", "best practices", 
        "research taste", "problem selection", "academic", "phd", "career",
        "how to", "strategy", "planning", "principles"
    )
    for kw in guidelines_keywords:
        if kw in g:
            score += 0.3  # Higher weight for guidelines-specific terms
    
    return score


def _metadata_score(meta: Dict[str, Any]) -> float:
    s = 0.0
    quality = meta.get("quality", {}) if isinstance(meta, dict) else {}
    rel = quality.get("reliability_score")
    if isinstance(rel, (int, float)):
        s += min(max(rel, 0.0), 1.0)  # clamp 0..1
    operational = meta.get("operational", {}) if isinstance(meta, dict) else {}
    cost = str(operational.get("cost_estimate", "unknown"))
    cost_penalty = {"low": 0.0, "medium": -0.1, "high": -0.3}.get(cost, -0.05)
    s += cost_penalty
    return s


def score_tools(goal: str, tools: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """Return (name, score, rationale) sorted by descending score.

    tools: name -> tool instance exposing can_handle() and get_metadata().
    """
    results: List[Tuple[str, float, str]] = []
    for name, tool in tools.items():
        try:
            if not getattr(tool, "can_handle", lambda *_: True)({"goal": goal}):
                continue
            score = 0.0
            rationale_parts: List[str] = []

            # Base priority
            if name in PRIMARY_NAMES:
                score += 2.0
                rationale_parts.append("primary")
            elif name.startswith(LEGACY_PREFIX):
                score -= 0.5
                rationale_parts.append("legacy")
            else:
                score += 0.0

            # Keyword match
            km = _keyword_match_score(goal, name)
            score += km
            if km > 0:
                rationale_parts.append("keywords")

            # Metadata
            meta = getattr(tool, "get_metadata", lambda: {})()
            ms = _metadata_score(meta)
            score += ms
            if ms != 0:
                rationale_parts.append("metadata")

            results.append((name, score, "+".join(rationale_parts) or "default"))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results
