from __future__ import annotations

import re
from typing import Any

from .tool_helpers import print_summary_and_sources, registry_tool_call
from .guidelines_tool import guidelines_tool_fn
from ..rich_formatter import print_agent_reasoning


def _extract_mode(text: str) -> tuple[str, str]:
    """Return (clean_text, mode) where mode is 'concise' or 'detailed'."""
    mode = "concise"
    lowered = (text or "").lower()
    if "mode:detailed" in lowered or "response:detailed" in lowered:
        mode = "detailed"
    cleaned = re.sub(r"\b(mode|response)\s*:\s*detailed\b", "", text or "", flags=re.IGNORECASE).strip()
    return cleaned, mode


def arxiv_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    # Legacy direct call (no registry). Add transparency prints.
    from ..mentor_tools import arxiv_search

    begin, end = internal_delimiters or ("", "")
    clean_q, mode = _extract_mode(q)
    print_agent_reasoning(f"Using tool: legacy_arxiv_search (mode={mode})")
    res = arxiv_search(query=clean_q, from_year=None, limit=5)
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
        abstract = p.get("summary") or ""
        suffix = f" ({year})" if year else ""
        if mode == "detailed" and abstract:
            lines.append(f"- {title}{suffix} -> {url}\n  Abstract: {abstract[:420]}{'…' if len(abstract) > 420 else ''}")
        else:
            lines.append(f"- {title}{suffix} -> {url}")
    reasoning = "\n".join([f"Legacy arXiv results ({mode}):"] + lines)
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

    guidelines = report.get("guidelines") or []
    if guidelines:
        lines.append("Relevant guidelines:")
        for guideline in guidelines[:3]:
            title = guideline.get("title", "Guideline")
            content = guideline.get("content", "")
            lines.append(f"- {title}: {content}")

    for k in keys:
        vals = report.get(k) or []
        if vals:
            lines.append(f"- {k}: {', '.join(str(x) for x in vals)}")

    sample_size_notes = report.get("sample_size_notes")
    if sample_size_notes:
        lines.append(f"- sample_size: {sample_size_notes}")

    alignment_prompt = report.get("alignment_prompt")
    if alignment_prompt:
        lines.append(f"- llm_check: {alignment_prompt}")

    lines.append("Note: This check is heuristic and does not guarantee correctness; consult domain experts.")

    reasoning = "\n".join(["Methodology check (guideline-assisted):"] + (lines or ["No issues detected"]))
    print_agent_reasoning(reasoning)
    return f"{begin}{reasoning}{end}" if begin or end else reasoning


def web_search_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
    clean_q, mode = _extract_mode(q)
    result = registry_tool_call("web_search", {"query": clean_q, "limit": 8})
    items = (result.get("results") if isinstance(result, dict) else []) or []
    if not items:
        note = (result or {}).get("note", "No results") if isinstance(result, dict) else "No results"
        return str(note)
    lines: list[str] = []
    for it in items[:5]:
        title = it.get("title") or it.get("paper_title") or "result"
        year = it.get("year") or it.get("published") or ""
        url = it.get("url") or (it.get("urls", {}) or {}).get("paper") or ""
        snippet = it.get("snippet") or it.get("summary") or ""
        suffix = f" ({year})" if year else ""
        link = f" -> {url}" if url else ""
        if mode == "detailed" and snippet:
            snippet_txt = snippet.strip().replace("\n", " ")
            if len(snippet_txt) > 280:
                snippet_txt = snippet_txt[:280] + "…"
            lines.append(f"- {title}{suffix}{link}\n  {snippet_txt}")
        else:
            lines.append(f"- {title}{suffix}{link}")
    reasoning = "\n".join([f"Top web results ({mode}):"] + lines)
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

# Import unified research tool from separate module
from .unified_research import unified_research_tool_fn
