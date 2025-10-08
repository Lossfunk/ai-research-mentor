from __future__ import annotations

from typing import Any

from ..rich_formatter import print_agent_reasoning
from .tool_impls import (
    arxiv_tool_fn,
    web_search_tool_fn,
    searchthearxiv_tool_fn,
    math_tool_fn,
    method_tool_fn,
    guidelines_tool_fn,
    experiment_planner_tool_fn,
    unified_research_tool_fn,
)
from ..attachments import has_attachments, search as attachments_search


def get_langchain_tools() -> list[Any]:
    try:
        from langchain.tools import Tool  # type: ignore
    except Exception:
        return []

    # Internal delimiters for hiding tool reasoning when needed by agents
    internal_delimiters = ("<<<AGENT_INTERNAL_BEGIN>>>\n", "\n<<<AGENT_INTERNAL_END>>>")

    def wrap(fn):
        return lambda *args, **kwargs: fn(*args, internal_delimiters=internal_delimiters, **kwargs)

    tools: list[Any] = [
        Tool(
            name="arxiv_search",
            func=wrap(arxiv_tool_fn),
            description=(
                "Search arXiv for recent academic papers on any research topic. "
                "Use this whenever the user asks about research, papers, literature, "
                "related work, or wants to understand what's been done in a field. "
                "Input: research topic or keywords (e.g. 'transformer models', 'deep reinforcement learning'). "
                "Returns: list of relevant papers with titles, years, and URLs."
            ),
        ),
        Tool(
            name="research_guidelines",
            func=wrap(guidelines_tool_fn),
            description=(
                "Mentorship guidelines from curated sources. For novelty/methodology/experiments questions, use this "
                "IMMEDIATELY AFTER attachments_search to establish best-practice principles BEFORE any literature search."
            ),
        ),
        Tool(
            name="experiments_plan",
            func=wrap(experiment_planner_tool_fn),
            description=(
                "Propose 3 concrete, falsifiable experiments grounded in attached snippets. "
                "Use AFTER attachments_search (and optionally mentorship_guidelines); returns numbered experiments with hypothesis, variables, metrics, expected outcome, and [file:page] anchors."
            ),
        ),
        Tool(
            name="math_ground",
            func=wrap(math_tool_fn),
            description=(
                "Heuristic math grounding. Input: TeX/plain text. Returns brief findings."
            ),
        ),
        Tool(
            name="methodology_validate",
            func=wrap(method_tool_fn),
            description=(
                "Validate an experiment plan for risks/controls/ablations/reproducibility gaps."
            ),
        ),
        Tool(
            name="unified_research",
            func=wrap(unified_research_tool_fn),
            description=(
                "Unified research tool that combines papers and guidelines with [P#] and [G#] citations. "
                "Use this for comprehensive research queries that need both literature and methodology guidance. "
                "Returns papers with [P1], [P2]... and guidelines with [G1], [G2]... for proper citation."
            ),
        ),
        Tool(
            name="web_search",
            func=wrap(web_search_tool_fn),
            description=(
                "Tavily-powered web search for recent information, news, and non-arXiv sources. "
                "Use when the user asks for up-to-date context, real-world events, or resources outside academic archives. "
                "Returns top web results with titles, links, and brief summaries."
            ),
        ),
        Tool(
            name="searchthearxiv_search",
            func=wrap(searchthearxiv_tool_fn),
            description=(
                "Semantic arXiv search placeholder. Provides the historical searchthearxiv entry point; currently returns no results but keeps agent workflows consistent."
            ),
        ),
        # Tool(
        #     name="searchthearxiv_search",
        #     func=wrap(searchthearxiv_tool_fn),
        #     description=(
        #         "Semantic arXiv search via searchthearxiv.com. Use for natural language queries. "
        #         "Includes transparency logs and sources. Input: research query."
        #     ),
        # ),
    ]
    # Always add attachments_search tool (it handles empty attachments gracefully)
    def _attachments_tool_fn(q: str, *, internal_delimiters: tuple[str, str] | None = None) -> str:
        begin, end = internal_delimiters or ("", "")
        print_agent_reasoning("Using tool: attachments_search")
        if not has_attachments():
            return f"{begin}No attachments loaded. Use --attach-pdf to add documents.{end}" if begin or end else "No attachments loaded. Use --attach-pdf to add documents."
        # ResponseFormat control via inline directive
        q_lower = (q or "").lower()
        detailed = "format:detailed" in q_lower or "response:detailed" in q_lower
        # Token efficiency controls via inline directives: k:<int> page:<int> size:<int>
        def _extract_int(token: str, default_val: int) -> int:
            try:
                import re
                m = re.search(rf"{token}\s*:\s*(\d+)", q_lower)
                return int(m.group(1)) if m else default_val
            except Exception:
                return default_val
        req_k = max(1, min(_extract_int("k", 4), 8))
        page = max(1, min(_extract_int("page", 1), 50))
        size = max(1, min(_extract_int("size", req_k), 8))
        clean_q = (
            q.replace("format:detailed", "")
             .replace("response:detailed", "")
             .replace("k:", " k:")
             .replace("page:", " page:")
             .replace("size:", " size:")
        )
        # Strip directive patterns
        try:
            import re as _re
            clean_q = _re.sub(r"\b(k|page|size)\s*:\s*\d+", "", clean_q).strip()
        except Exception:
            clean_q = clean_q.strip()

        results = attachments_search(clean_q, k=req_k * page)
        # Apply pagination window
        start = (page - 1) * size
        end_idx = start + size
        window = results[start:end_idx] if start < len(results) else []
        if not results:
            return f"{begin}No relevant snippets found in attached PDFs{end}" if begin or end else "No relevant snippets found in attached PDFs"
        lines: list[str] = ["Context snippets from attachments:"]
        for r in (window or results)[:size]:
            file = r.get("file", "file.pdf")
            page = r.get("page", 1)
            snippet = (r.get("snippet") or r.get("text") or "").strip().replace("\n", " ")
            text = snippet if not detailed else (r.get("text") or snippet)
            if not detailed and len(text) > 220:
                text = text[:220] + "…"
            anchor = r.get("anchor") or f"{file}#page={page}"
            lines.append(f"- [{file}:{page}] {text} (anchor: {anchor})")
        reasoning = "\n".join(lines)
        return f"{begin}{reasoning}{end}" if begin or end else reasoning

    # Prefer attachments_search first in the tool list so agents try it before external search
    tools.insert(
        0,
        Tool(
            name="attachments_search",
            func=wrap(_attachments_tool_fn),
            description=(
                "GROUNDING FIRST: Use this FIRST to retrieve relevant snippets from attached PDFs and cite [file:page]. "
                "Defaults: concise, k=4. Controls via inline directives: k:<1-8>, page:<n>, size:<1-8>, format:detailed."
            ),
        ),
    )
    return tools
