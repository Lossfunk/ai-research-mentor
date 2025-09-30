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
        from ..citations import CitationMerger

        # Ensure tools are discovered
        auto_discover()

        orch = Orchestrator()
        # Request a larger page to surface more curated sources with full URLs
        result = orch.execute_task(
            task="research_guidelines",
            inputs={
                "query": query,
                "topic": query,
                "response_format": "concise",
                "page_size": 30,
                "mode": "fast",
            },
            context={"goal": f"research mentorship guidance about {query}"}
        )

        if result["execution"]["executed"] and result["results"]:
            tool_result = result["results"]

            # Support both V2 structured evidence and V1 legacy output
            evidence_items = tool_result.get("evidence") or []
            guidelines = tool_result.get("retrieved_guidelines", [])

            if not evidence_items and not guidelines:
                return "No specific guidelines found for this query. Try rephrasing or ask more specific questions about research methodology."

            # Use citation merger for unified formatting
            merger = CitationMerger()
            merged_result = merger.merge_citations(
                papers=[],  # No papers from guidelines tool
                guidelines=evidence_items + guidelines,
                max_guidelines=30
            )

            reasoning_block = merged_result["context"]
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
    """List existing experiments from attached documents OR propose new ones.

    Input: user question or goal. Reads attached snippets via attachments.search.
    
    If user asks to "list", "show", "what experiments", return existing experiments from document.
    If user asks to "propose", "suggest", "design new", generate new experiment proposals.
    
    Uses document summary to provide accurate information about experiments.
    """
    begin, end = internal_delimiters or ("", "")
    try:
        from ..attachments import has_attachments, search as att_search, get_document_summary
        if not has_attachments():
            return f"{begin}No attachments loaded; cannot access experiments{end}" if begin or end else "No attachments loaded; cannot access experiments"
        
        # Get document summary for context about what's already been done
        doc_summary = get_document_summary()
        
        # Detect if user wants to LIST existing experiments or PROPOSE new ones
        query_lower = q.lower()
        wants_list = any(keyword in query_lower for keyword in [
            "list", "show", "what", "which", "all the", "describe", "summary", "summarize", "done", "conducted"
        ])
        wants_new = any(keyword in query_lower for keyword in [
            "propose", "suggest", "new", "next", "future", "design", "plan", "recommend"
        ])
        
        # If user wants list and has valid summary, return it directly
        if wants_list and not wants_new and doc_summary and "LLM unavailable" not in doc_summary:
            reasoning = f"Document contains the following experiments:\n\n{doc_summary}"
            print_agent_reasoning(reasoning)
            return f"{begin}{reasoning}{end}" if begin or end else reasoning
        
        detailed = "format:detailed" in (q or "").lower()
        clean_q = q.replace("format:detailed", "").replace("response:detailed", "").strip()
        snippets = att_search(clean_q, k=12)
        if not snippets:
            return f"{begin}No relevant snippets found in attachments{end}" if begin or end else "No relevant snippets found in attachments"
        
        # Build context with document summary and snippets
        context_lines = []
        if doc_summary and "LLM unavailable" not in doc_summary and "failed" not in doc_summary:
            context_lines.append("=== DOCUMENT CONTEXT (What's already done) ===")
            context_lines.append(doc_summary[:1500])
            context_lines.append("\n=== RELEVANT DOCUMENT EXCERPTS ===")
        
        for i, s in enumerate(snippets[:6], 1):
            anchor = f"[{s.get('file','file.pdf')}:{s.get('page',1)}]"
            base_text = (s.get("text") if detailed else s.get("snippet")) or s.get("text") or ""
            snippet = base_text.strip().replace("\n", " ")
            if not detailed and len(snippet) > 200:
                snippet = snippet[:200] + "…"
            context_lines.append(f"{i}. {anchor}: {snippet}")
        
        full_context = "\n".join(context_lines)
        
        # Prefix with clear instruction to avoid duplication
        instruction = f"""Based on the user query: "{clean_q}"

Review the document context carefully to understand what experiments have ALREADY been conducted.

Propose 3 NEW experiments that:
1. Are NOT duplicates of already-conducted experiments
2. Build on or complement the existing work
3. Address gaps or next steps

{full_context}

Format each experiment as:
Experiment N: [Brief title]
- Hypothesis: [What you're testing]
- Variables: [What you'll manipulate and measure]
- Method: [How you'll conduct it]
- Expected outcome: [What results would support/refute hypothesis]
- Grounding: [Cite relevant pages from document]"""

        reasoning = f"Generating experiments with context awareness...\n\n{instruction[:1200]}"
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


# Import unified research tool from separate module
from .unified_research import unified_research_tool_fn
