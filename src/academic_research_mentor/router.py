from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _run_arxiv_search_and_print(query: str) -> None:
    from .mentor_tools import arxiv_search  # lazy import
    result: Dict[str, Any] = arxiv_search(query=query, from_year=None, limit=5)
    papers: List[Dict[str, Any]] = result.get("papers", []) if isinstance(result, dict) else []
    if not papers:
        note = result.get("note") if isinstance(result, dict) else None
        print(f"Mentor.tools: No papers found. {note or ''}")
        return
    print("Mentor.tools (arXiv):")
    for p in papers[:5]:
        title = p.get("title")
        year = p.get("year")
        url = p.get("url")
        print(f"- {title} ({year}) → {url}")


def _run_openreview_and_print(query: str) -> None:
    from .mentor_tools import openreview_fetch  # lazy import
    result: Dict[str, Any] = openreview_fetch(query=query, limit=5)
    threads: List[Dict[str, Any]] = result.get("threads", []) if isinstance(result, dict) else []
    if not threads:
        note = result.get("note") if isinstance(result, dict) else None
        print(f"Mentor.tools: No OpenReview threads found. {note or ''}")
        return
    print("Mentor.tools (OpenReview):")
    for t in threads[:5]:
        title = t.get("paper_title")
        venue = t.get("venue")
        year = t.get("year")
        url = (t.get("urls") or {}).get("paper")
        suffix = f" ({venue} {year})" if venue or year else ""
        print(f"- {title}{suffix} → {url}")


def _run_venue_guidelines_and_print(venue: str, year: Optional[int]) -> None:
    from .mentor_tools import venue_guidelines_get  # lazy import
    result: Dict[str, Any] = venue_guidelines_get(venue=venue, year=year)
    g = (result or {}).get("guidelines", {})
    urls = g.get("urls", {}) if isinstance(g, dict) else {}
    print("Mentor.tools (Venue Guidelines):")
    print(f"- Venue: {venue.upper()} {year or ''}")
    if urls.get("guide"):
        print(f"- Guide: {urls['guide']}")
    if urls.get("template"):
        print(f"- Template: {urls['template']}")
    if not urls.get("guide") and not urls.get("template"):
        print("- No known URLs. Try checking the venue website.")


def _run_math_ground_and_print(text: str) -> None:
    from .mentor_tools import math_ground  # lazy import
    result: Dict[str, Any] = math_ground(text_or_math=text, options={})
    findings = (result or {}).get("findings", {})
    print("Mentor.tools (Math Ground):")
    for key in ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]:
        items = findings.get(key) or []
        if items:
            print(f"- {key}: {', '.join(str(x) for x in items[:3])}{'...' if len(items) > 3 else ''}")


def _run_methodology_validate_and_print(plan: str) -> None:
    from .mentor_tools import methodology_validate  # lazy import
    result: Dict[str, Any] = methodology_validate(plan=plan, checklist=[])
    report = (result or {}).get("report", {})
    print("Mentor.tools (Methodology Validate):")
    for key in ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]:
        items = report.get(key) or []
        if items:
            print(f"- {key}: {', '.join(str(x) for x in items)}")


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
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            topic = re.sub(r"[.?!\s]+$", "", topic)
            if 2 <= len(topic) <= 200:
                return topic
    return None


def route_and_maybe_run_tool(user: str) -> bool:
    s = user.strip()
    if not s:
        return False

    m = re.search(r"(?:author\s+)?guidelines(?:\s+for)?\s+([A-Za-z]+)(?:\s+(\d{4}))?", s, flags=re.IGNORECASE)
    if m:
        venue = m.group(1)
        year = int(m.group(2)) if m.group(2) else None
        _run_venue_guidelines_and_print(venue, year)
        return True

    if re.search(r"\bopen\s*review\b|\bopenreview\b", s, flags=re.IGNORECASE):
        m2 = re.search(r"(?:open\s*review|openreview)\s*(?:for|about|on)?\s*(.*)$", s, flags=re.IGNORECASE)
        query = (m2.group(1) if m2 and m2.group(1) else s).strip()
        _run_openreview_and_print(query or s)
        return True

    if re.search(r"\$|\\\(|\\\[|\\begin\{equation\}|\\int|\\sum|\\frac|^\s*math\s*:\s*", s, flags=re.IGNORECASE):
        text = re.sub(r"^\s*math\s*:\s*", "", s, flags=re.IGNORECASE)
        _run_math_ground_and_print(text or s)
        return True

    if re.search(r"\b(experiment|evaluation)\s+plan\b|\bmethodology\b|^\s*validate\s*:\s*", s, flags=re.IGNORECASE):
        plan = re.sub(r"^\s*validate\s*:\s*", "", s, flags=re.IGNORECASE)
        _run_methodology_validate_and_print(plan or s)
        return True

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
    for pat in arxiv_patterns:
        m3 = re.search(pat, s, flags=re.IGNORECASE)
        if m3:
            topic = m3.group(1).strip()
            topic = re.sub(r"[.?!\s]+$", "", topic)
            if topic:
                _run_arxiv_search_and_print(topic)
                return True

    topic = _extract_topic_from_text(s)
    if topic:
        print(f"Mentor.tools: Detected topic → {topic}")
        _run_arxiv_search_and_print(topic)
        return True

    return False
