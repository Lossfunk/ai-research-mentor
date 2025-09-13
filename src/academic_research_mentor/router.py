from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from .rich_formatter import print_agent_reasoning


def _run_arxiv_search_and_print(query: str) -> None:
    from .mentor_tools import arxiv_search  # lazy import
    result: Dict[str, Any] = arxiv_search(query=query, from_year=None, limit=5)
    papers: List[Dict[str, Any]] = result.get("papers", []) if isinstance(result, dict) else []
    if not papers:
        note = result.get("note") if isinstance(result, dict) else None
        print(f"Mentor.tools: No papers found. {note or ''}")
        return
    print_agent_reasoning("Mentor.tools (arXiv):")
    for p in papers[:5]:
        title = p.get("title")
        year = p.get("year")
        url = p.get("url")
        print_agent_reasoning(f"- {title} ({year}) → {url}")

    # The following block seems out of place and references 'urls' which is undefined.
    # Commenting it out to fix indentation and undefined variable issues.
    # if not urls.get("guide") and not urls.get("template"):
    #     print("- No known URLs. Try checking the venue website.")


def _run_math_ground_and_print(text: str) -> None:
    from .mentor_tools import math_ground  # lazy import
    result: Dict[str, Any] = math_ground(text_or_math=text, options={})
    findings = (result or {}).get("findings", {})
    print_agent_reasoning("Mentor.tools (Math Ground):")
    for key in ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]:
        items = findings.get(key) or []
        if items:
            print_agent_reasoning(f"- {key}: {', '.join(str(x) for x in items[:3])}{'...' if len(items) > 3 else ''}")


def _run_guidelines_and_print(query: str, topic: Optional[str] = None) -> None:
    """Fallback direct guidelines tool usage."""
    try:
        from .tools.guidelines.tool import GuidelinesTool
        tool = GuidelinesTool()
        tool.initialize()
        
        inputs = {"query": query}
        if topic:
            inputs["topic"] = topic
            
        result = tool.execute(inputs, {"goal": f"research mentorship guidance about {query}"})
        
        guidelines = result.get("retrieved_guidelines", [])
        if guidelines:
            print_agent_reasoning("Mentor.tools (Research Guidelines):")
            for guideline in guidelines[:3]:
                source = guideline.get("source_domain", "Research guidance")
                content = guideline.get("content", "")[:100]
                print_agent_reasoning(f"- {source}: {content}...")
        else:
            print_agent_reasoning("Mentor.tools: No specific guidelines found.")
    except Exception as e:
        print_agent_reasoning(f"Mentor.tools: Guidelines search failed: {e}")


def _run_methodology_validate_and_print(plan: str) -> None:
    from .mentor_tools import methodology_validate  # lazy import
    result: Dict[str, Any] = methodology_validate(plan=plan, checklist=[])
    report = (result or {}).get("report", {})
    print_agent_reasoning("Mentor.tools (Methodology Validate):")
    for key in ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]:
        items = report.get(key) or []
        if items:
            print_agent_reasoning(f"- {key}: {', '.join(str(x) for x in items)}")


def _run_guidelines_and_print(query: str, topic: Optional[str] = None) -> None:
    """Run guidelines tool and print results."""
    # Try to use orchestrator first if available
    try:
        from .core.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        context = {"goal": query, "query": query}
        if topic:
            context["topic"] = topic
        
        result = orchestrator.execute_task("guidelines_search", {"query": query, "topic": topic or query}, context)
        
        if result.get("execution", {}).get("executed"):
            guidelines_result = result.get("results", {})
            print_agent_reasoning("Mentor.tools (Research Guidelines):")
            
            if guidelines_result.get("retrieved_guidelines"):
                for guideline in guidelines_result["retrieved_guidelines"]:
                    source_type = guideline.get("source_type", "Unknown source")
                    guide_id = guideline.get("guide_id", "unknown")
                    print_agent_reasoning(f"- {source_type} [ID: {guide_id}]")
                    
                if guidelines_result.get("formatted_content"):
                    content = guidelines_result["formatted_content"]
                    # Show first 500 chars of formatted content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print_agent_reasoning(f"\nGuidelines summary: {content}")
            else:
                print_agent_reasoning("- No guidelines found for this query")
        else:
            # Fallback to direct tool usage
            _run_guidelines_fallback(query, topic)
            
    except Exception:
        # Fallback to direct tool usage
        _run_guidelines_fallback(query, topic)


def _run_guidelines_fallback(query: str, topic: Optional[str] = None) -> None:
    """Fallback direct guidelines tool usage."""
    try:
        from .tools.guidelines.tool import GuidelinesTool
        tool = GuidelinesTool()
        tool.initialize()
        
        inputs = {"query": query}
        if topic:
            inputs["topic"] = topic
            
        result = tool.execute(inputs)
        
        print("Mentor.tools (Research Guidelines):")
        if result.get("retrieved_guidelines"):
            for guideline in result["retrieved_guidelines"]:
                source_type = guideline.get("source_type", "Unknown source")
                guide_id = guideline.get("guide_id", "unknown")
                print(f"- {source_type} [ID: {guide_id}]")
                
            if result.get("formatted_content"):
                content = result["formatted_content"]
                if len(content) > 500:
                    content = content[:500] + "..."
                print(f"\nGuidelines summary: {content}")
        else:
            print(f"- {result.get('note', 'No guidelines found')}")
            
    except Exception as e:
        print(f"Mentor.tools (Research Guidelines): Error - {e}")


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


def route_and_maybe_run_tool(user: str) -> Optional[Dict[str, str]]:
    s = user.strip()
    if not s:
        return None

    # Check for research guidelines queries first (before venue guidelines)
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
    for pat in arxiv_patterns:
        m3 = re.search(pat, s, flags=re.IGNORECASE)
        if m3:
            topic = m3.group(1).strip()
            topic = re.sub(r"[.?!\s]+$", "", topic)
            if topic:
                _run_arxiv_search_and_print(topic)
                return {"tool_name": "arxiv_search", "topic": str(topic)}
    
    topic = _extract_topic_from_text(s)
    if topic:
        print_agent_reasoning(f"Mentor.tools: Detected topic → {topic}")
        _run_arxiv_search_and_print(topic)
        return {"tool_name": "arxiv_search", "topic": topic}
    
    return None
