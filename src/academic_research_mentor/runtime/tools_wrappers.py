from __future__ import annotations

from typing import Any

from ..rich_formatter import print_agent_reasoning
from .tool_impls import (
    arxiv_tool_fn,
    o3_search_tool_fn,
    searchthearxiv_tool_fn,
    math_tool_fn,
    method_tool_fn,
    guidelines_tool_fn,
)


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
            name="o3_search",
            func=wrap(o3_search_tool_fn),
            description=(
                "Consolidated literature search using O3 reasoning across arXiv and OpenReview. "
                "Prefer this over legacy arxiv_search; includes transparency logs and sources. "
                "Input: research topic. Returns key papers with links."
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
            name="research_guidelines",
            func=wrap(guidelines_tool_fn),
            description=(
                "Search curated research methodology and mentorship guidelines from expert sources. "
                "USE THIS TOOL FOR ALL RESEARCH MENTORSHIP QUESTIONS including: "
                "research advice, methodology guidance, PhD help, problem selection, research taste development, academic career guidance, "
                "research strategy decisions, publication dilemmas, research evaluation questions, and academic career planning. "
                "Specifically use when users ask about: "
                "- Research direction uncertainty ('no one else is working on this', 'red flag or opportunity', 'unique research direction') "
                "- Problem worthiness ('worth pursuing vs distraction', 'should I work on this problem', 'is this important') "
                "- Negative results ('approach doesn't work', 'should I publish negative results', 'my method failed') "
                "- Novelty concerns ('not sure novel enough', 'how to evaluate novelty', 'is this contribution significant') "
                "- Publication decisions ('should I publish this', 'where to publish', 'ready for publication') "
                "- Research taste and judgment ('developing research taste', 'how to choose problems', 'research intuition') "
                "- Academic career guidance ('career planning', 'PhD advice', 'research skills development') "
                "Input: any research mentorship question, dilemma, or uncertainty. Examples: "
                "'I found an interesting research direction but I'm worried no one else is working on it. Is that a red flag or an opportunity?', "
                "'My results are negative - my approach doesn't work. Should I publish this or try something else?', "
                "'I have some interesting results but I'm not sure they're 'novel' enough for publication. How do I evaluate this?', "
                "'How do I know if a research problem is worth pursuing vs just a distraction?' "
                "Returns: structured guidelines from authoritative sources with source attribution."
            ),
        ),
        Tool(
            name="searchthearxiv_search",
            func=wrap(searchthearxiv_tool_fn),
            description=(
                "Semantic arXiv search via searchthearxiv.com. Use for natural language queries. "
                "Includes transparency logs and sources. Input: research query."
            ),
        ),
    ]
    return tools
