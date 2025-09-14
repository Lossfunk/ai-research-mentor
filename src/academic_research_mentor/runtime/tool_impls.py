from __future__ import annotations

from typing import Any

from .tool_helpers import print_summary_and_sources, registry_tool_call
from ..rich_formatter import print_agent_reasoning


def arxiv_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    # Legacy direct call (no registry). Add transparency prints.
    from ..mentor_tools import arxiv_search

    begin, end = internal_delimiters or ("", "")
    print_agent_reasoning("Using tool: legacy_arxiv_search")
    res = arxiv_search(query=q, from_year=None, limit=5)
    print_summary_and_sources(res if isinstance(res, dict) else {})
    papers = (res or {}).get("papers", [])
    if not papers:
        note = (res or {}).get("note", "No results")
        reasoning = f"Legacy arXiv search: {note}"
        print_agent_reasoning(reasoning)
        return f"{begin}{reasoning}{end}" if begin or end else reasoning
    lines = []
    for p in papers[:5]:
        title = p.get("title")
        year = p.get("year")
        url = p.get("url")
        lines.append(f"- {title} ({year}) -> {url}")
    reasoning = "\n".join(["Legacy arXiv results:"] + lines)
    print_agent_reasoning(reasoning)
    return f"{begin}{reasoning}{end}" if begin or end else reasoning


def math_tool_fn(text: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    from ..mentor_tools import math_ground

    begin, end = internal_delimiters or ("", "")
    res = math_ground(text_or_math=text, options={})
    findings = (res or {}).get("findings", {})
    keys = ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]
    lines = []
    for k in keys:
        vals = findings.get(k) or []
        if vals:
            lines.append(f"- {k}: {', '.join(str(x) for x in vals[:3])}")
    reasoning = "\n".join(["Math grounding findings:"] + (lines or ["No findings"]))
    print_agent_reasoning(reasoning)
    return f"{begin}{reasoning}{end}" if begin or end else reasoning


def method_tool_fn(text: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    from ..mentor_tools import methodology_validate

    begin, end = internal_delimiters or ("", "")
    res = methodology_validate(plan=text, checklist=[])
    report = (res or {}).get("report", {})
    keys = ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]
    lines = []
    for k in keys:
        vals = report.get(k) or []
        if vals:
            lines.append(f"- {k}: {', '.join(str(x) for x in vals)}")
    reasoning = "\n".join(["Methodology validation:"] + (lines or ["No issues detected"]))
    print_agent_reasoning(reasoning)
    return f"{begin}{reasoning}{end}" if begin or end else reasoning


def guidelines_tool_fn(query: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    """Search for research methodology and mentorship guidelines from curated sources."""
    begin, end = internal_delimiters or ("", "")
    try:
        from ..core.orchestrator import Orchestrator
        from ..tools import auto_discover

        # Ensure tools are discovered
        auto_discover()

        orch = Orchestrator()
        result = orch.execute_task(
            task="research_guidelines",
            inputs={"query": query, "topic": query},
            context={"goal": f"research mentorship guidance about {query}"}
        )

        if result["execution"]["executed"] and result["results"]:
            tool_result = result["results"]
            guidelines = tool_result.get("retrieved_guidelines", [])

            if not guidelines:
                return "No specific guidelines found for this query. Try rephrasing or ask more specific questions about research methodology."

            # Format guidelines for agent consumption
            formatted_lines = [f"Found {len(guidelines)} relevant research guidelines:"]
            sources = []

            for guideline in guidelines:
                guide_id = guideline.get("guide_id", "unknown")
                source_type = guideline.get("source_type", "Research guidance")
                source_domain = guideline.get("source_domain", "")
                content = guideline.get("content", "")[:300]  # Truncate for token efficiency

                # Extract source domain for display
                if source_domain and source_domain not in sources:
                    sources.append(source_domain)

                formatted_lines.append(f"GUIDELINE [{guide_id}]:")
                formatted_lines.append(f"Source: {source_type}")
                formatted_lines.append(f"Content: {content}")
                formatted_lines.append("---")

            # Add instruction for agent
            formatted_lines.append(
                "\nUse these guidelines to provide evidence-based research advice. "
                "Reference specific guidelines as [guide_id] in your response."
            )

            # Add sources section at the end
            if sources:
                formatted_lines.append("\n\nSource:")
                for i, source in enumerate(sources, 1):
                    formatted_lines.append(f"{i}. {source}")

            reasoning_block = "\n".join(formatted_lines)
            # Print as Agent's reasoning panel for TUI differentiation
            print_agent_reasoning(reasoning_block)
            return f"{begin}{reasoning_block}{end}" if begin or end else reasoning_block
        else:
            return "Guidelines search temporarily unavailable. Please try again later."

    except Exception as e:
        return f"Error searching guidelines: {str(e)}"


def o3_search_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    result = registry_tool_call("o3_search", {"query": q, "limit": 8})
    items = (result.get("results") if isinstance(result, dict) else []) or []
    if not items:
        note = (result or {}).get("note", "No results") if isinstance(result, dict) else "No results"
        return str(note)
    lines: list[str] = []
    for it in items[:5]:
        title = it.get("title") or it.get("paper_title") or "result"
        year = it.get("year") or it.get("published") or ""
        url = it.get("url") or (it.get("urls", {}) or {}).get("paper") or ""
        suffix = f" ({year})" if year else ""
        link = f" -> {url}" if url else ""
        lines.append(f"- {title}{suffix}{link}")
    reasoning = "\n".join(["Top literature results:"] + lines)
    print_agent_reasoning(reasoning)
    begin, end = internal_delimiters or ("", "")
    return f"{begin}{reasoning}{end}" if begin or end else reasoning


def experiment_planner_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    """Propose 3 concrete, falsifiable experiments grounded in attached snippets.

    Input: user question or goal. Reads attached snippets via attachments.search.
    Output: numbered experiments with hypothesis, variables, metric, expected outcome, and [file:page] anchors.
    """
    begin, end = internal_delimiters or ("", "")
    try:
        from ..attachments import has_attachments, search as att_search
        if not has_attachments():
            return f"{begin}No attachments loaded; cannot generate grounded experiments{end}" if begin or end else "No attachments loaded; cannot generate grounded experiments"
        detailed = "format:detailed" in (q or "").lower()
        clean_q = q.replace("format:detailed", "").replace("response:detailed", "").strip()
        snippets = att_search(clean_q, k=6)
        if not snippets:
            return f"{begin}No relevant snippets found in attachments{end}" if begin or end else "No relevant snippets found in attachments"
        lines: list[str] = ["Grounded experiment plan (3 items):"]
        for i, s in enumerate(snippets[:3], 1):
            anchor = f"[{s.get('file','file.pdf')}:{s.get('page',1)}]"
            base_text = (s.get("text") if detailed else s.get("snippet")) or s.get("text") or ""
            snippet = base_text.strip().replace("\n", " ")
            if not detailed and len(snippet) > 160:
                snippet = snippet[:160] + "…"
            lines.append(f"{i}. Hypothesis: A design that avoids response-diversity reliance improves robustness {anchor}")
            lines.append(f"   Basis: {snippet}")
            lines.append("   Variables: (a) detector signal type, (b) alignment strength, (c) sampling temperature")
            lines.append("   Metric: AUROC, FNR at fixed FPR; report CI over seeds")
            lines.append("   Expected outcome: non-diversity signals degrade less under stronger alignment")
        reasoning = "\n".join(lines)
        print_agent_reasoning(reasoning)
        return f"{begin}{reasoning}{end}" if begin or end else reasoning
    except Exception as e:
        return f"Experiment planner failed: {e}"

def searchthearxiv_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    result = registry_tool_call("searchthearxiv_search", {"query": q, "limit": 10})
    papers = (result.get("papers") if isinstance(result, dict) else []) or []
    if not papers:
        note = (result or {}).get("note", "No results") if isinstance(result, dict) else "No results"
        return str(note)
    lines: list[str] = []
    for p in papers[:5]:
        title = p.get("title") or "paper"
        year = p.get("year") or ""
        url = p.get("url") or ""
        suffix = f" ({year})" if year else ""
        link = f" -> {url}" if url else ""
        lines.append(f"- {title}{suffix}{link}")
    reasoning = "\n".join(["Semantic arXiv results:"] + lines)
    print_agent_reasoning(reasoning)
    begin, end = internal_delimiters or ("", "")
    return f"{begin}{reasoning}{end}" if begin or end else reasoning
