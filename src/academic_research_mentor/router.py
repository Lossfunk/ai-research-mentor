from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .rich_formatter import print_agent_reasoning


def _run_arxiv_search_and_print(query: str) -> None:
    from .mentor_tools import arxiv_search

    result: Dict[str, Any] = arxiv_search(query=query, from_year=None, limit=5)
    papers: List[Dict[str, Any]] = result.get("papers", []) if isinstance(result, dict) else []
    if not papers:
        note = result.get("note") if isinstance(result, dict) else None
        print_agent_reasoning(f"Mentor.tools: No papers found. {note or ''}".strip())
        return

    print_agent_reasoning("Mentor.tools (arXiv):")
    for paper in papers[:5]:
        title = paper.get("title")
        year = paper.get("year")
        url = paper.get("url")
        print_agent_reasoning(f"- {title} ({year}) -> {url}")


def _run_math_ground_and_print(text: str) -> None:
    from .mentor_tools import math_ground

    result: Dict[str, Any] = math_ground(text_or_math=text, options={})
    findings = (result or {}).get("findings", {})
    print_agent_reasoning("Mentor.tools (Math Ground):")
    for key in ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]:
        items = findings.get(key) or []
        if items:
            suffix = "..." if len(items) > 3 else ""
            print_agent_reasoning(f"- {key}: {', '.join(str(x) for x in items[:3])}{suffix}")


def _run_methodology_validate_and_print(plan: str) -> None:
    from .mentor_tools import methodology_validate

    result: Dict[str, Any] = methodology_validate(plan=plan, checklist=[])
    report = (result or {}).get("report", {})
    print_agent_reasoning("Mentor.tools (Methodology Check):")
    for key in ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]:
        items = report.get(key) or []
        if items:
            print_agent_reasoning(f"- {key}: {', '.join(str(x) for x in items)}")
    sample_size_notes = report.get("sample_size_notes")
    if sample_size_notes:
        print_agent_reasoning(f"- sample_size: {sample_size_notes}")


def _run_guidelines_fallback(query: str, topic: Optional[str] = None) -> None:
    try:
        from .tools.guidelines.tool import GuidelinesTool

        tool = GuidelinesTool()
        tool.initialize()
        result = tool.execute({"query": query, "topic": topic or query})
        print_agent_reasoning("Mentor.tools (Research Guidelines):")

        guidelines = result.get("retrieved_guidelines", [])
        if guidelines:
            for guideline in guidelines[:4]:
                source = guideline.get("source_domain") or guideline.get("source") or "Unknown source"
                title = guideline.get("title") or "Guideline"
                print_agent_reasoning(f"- {title} ({source})")
        else:
            note = result.get("note", "No guidelines found")
            print_agent_reasoning(f"- {note}")
    except Exception as exc:
        print_agent_reasoning(f"Mentor.tools (Research Guidelines): Error - {exc}")


def _run_guidelines_and_print(query: str, topic: Optional[str] = None) -> None:
    try:
        from .core.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        result = orchestrator.execute_task(
            "research_guidelines",
            {"query": query, "topic": topic or query},
            {"goal": f"research mentorship guidance about {query}", "query": query},
        )

        if not result.get("execution", {}).get("executed"):
            _run_guidelines_fallback(query, topic)
            return

        guidelines_result = result.get("results", {}) or {}
        guidelines = guidelines_result.get("retrieved_guidelines", []) or guidelines_result.get("evidence", [])
        print_agent_reasoning("Mentor.tools (Research Guidelines):")
        if not guidelines:
            print_agent_reasoning("- No guidelines found for this query")
            return
        for guideline in guidelines[:4]:
            source = guideline.get("source_domain") or guideline.get("source") or "Unknown source"
            title = guideline.get("title") or "Guideline"
            print_agent_reasoning(f"- {title} ({source})")
    except Exception:
        _run_guidelines_fallback(query, topic)


def _extract_topic_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("!"):
        return None
    patterns = [
        r"\bI\s*am\s*interested\s*in\s+(.+)$",
        r"\bI'm\s*interested\s*in\s+(.+)$",
        r"\bInterested\s*in\s+(.+)$",
        r"\bMy\s*topic\s*(?:is|:)\s+(.+)$",
        r"\bTopic\s*:\s*(.+)$",
        r"\bI\s*want\s*to\s*research\s+(.+)$",
        r"\bResearch\s*(?:area|topic)\s*(?:is|:)\s+(.+)$",
        r"\bI\s*(?:need|want)\s*(?:to\s*)?(?:learn|understand)\s*(?:about|more\s*about)\s+(.+)$",
        r"\bI'm\s*(?:working|looking)\s*(?:on|into)\s+(.+)$",
        r"\bI\s*am\s*(?:working|looking)\s*(?:on|into)\s+(.+)$",
        r"\bCan\s*you\s*help\s*(?:me\s*)?(?:with|understand)\s+(.+)$",
        r"\bTell\s*me\s*about\s+(.+)$",
        r"\bWhat\s*(?:do\s*you\s*know\s*)?about\s+(.+)$",
        r"^(.+?)(?:\s*research|\s*papers|\s*literature)(?:\s*field|\s*area)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if not match:
            continue
        topic = re.sub(r"[.?!\s]+$", "", match.group(1).strip())
        if 2 <= len(topic) <= 200:
            return topic
    return None


def route_and_maybe_run_tool(user: str) -> Optional[Dict[str, str]]:
    s = user.strip()
    if not s:
        return None

    guidelines_patterns = [
        r"\b(?:research\s+)?guidelines?\s+(?:for|on|about)?\s+(.+)$",
        r"\b(?:how\s+to\s+)?(?:choose|select|pick)\s+(?:a\s+)?(?:good\s+)?(?:research\s+)?(?:problem|project|topic)\b",
        r"\b(?:research\s+)?(?:methodology|approach|process)\s+(?:advice|guidance|tips)\b",
        r"\b(?:develop|improve)\s+(?:research\s+)?taste\s+(?:and\s+judgment)?\b",
        r"\b(?:phd|graduate|academic)\s+(?:advice|guidance|career)\s+(?:planning|strategy)?\b",
        r"\b(?:what\s+)?(?:makes\s+)?(?:a\s+)?(?:good\s+)?(?:research\s+)?(?:problem|project|question)\b",
        r"\b(?:effective|good)\s+(?:research\s+)?principles?\b",
        r"\b(?:research\s+)?(?:best\s+)?practices?\b",
        r"\b(?:hamming|lesswrong|colah|nielsen)\s+(?:research\s+)?(?:advice|guidance)\b",
    ]
    for pattern in guidelines_patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            query = match.group(1) if match.groups() else s
            topic = query.strip() if query else s.strip()
            _run_guidelines_and_print(s, topic)
            return {"tool_name": "research_guidelines", "query": topic}

    if re.search(r"\$|\\\(|\\\[|\\begin\{equation\}|\\int|\\sum|\\frac|^\s*math\s*:\s*", s, flags=re.IGNORECASE):
        text = re.sub(r"^\s*math\s*:\s*", "", s, flags=re.IGNORECASE)
        _run_math_ground_and_print(text or s)
        return {"tool_name": "math_ground", "text": text}

    if re.search(r"\b(experiment|evaluation)\s+plan\b|\bmethodology\b|^\s*validate\s*:\s*", s, flags=re.IGNORECASE):
        plan = re.sub(r"^\s*validate\s*:\s*", "", s, flags=re.IGNORECASE)
        _run_methodology_validate_and_print(plan or s)
        return {"tool_name": "methodology_validate", "plan": plan}

    arxiv_patterns = [
        r"\bsearch\s+arxiv\s+for\s+(.+)$",
        r"\bfind\s+(?:recent\s+)?papers\s+(?:on|about)\s+(.+)$",
        r"\bpapers\s+(?:on|about)\s+(.+)$",
        r"\bliterature\s+(?:review|search)\s+(?:on|about|for)?\s*(.+)$",
        r"\brelated\s+work\s+(?:on|about|for)?\s*(.+)$",
        r"\bsurvey\s+(?:of|on)?\s*(.+)$",
        r"\bwhat\s+(?:are\s+)?(?:recent\s+)?(?:papers|research|work)\s+(?:on|about|in)\s+(.+)$",
        r"\bshow\s+me\s+(?:papers|research)\s+(?:on|about|in)\s+(.+)$",
        r"\bcan\s+you\s+find\s+(?:papers|research)\s+(?:on|about|in)\s+(.+)$",
    ]
    for pattern in arxiv_patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if not match:
            continue
        topic = re.sub(r"[.?!\s]+$", "", match.group(1).strip())
        if topic:
            _run_arxiv_search_and_print(topic)
            return {"tool_name": "arxiv_search", "topic": topic}

    topic = _extract_topic_from_text(s)
    if topic:
        print_agent_reasoning(f"Mentor.tools: Detected topic -> {topic}")
        _run_arxiv_search_and_print(topic)
        return {"tool_name": "arxiv_search", "topic": topic}

    return None
