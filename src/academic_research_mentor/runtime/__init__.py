"""Runtime compatibility module.

This module keeps minimal compatibility helpers used by legacy tests and CLI paths
while the codebase migrates to `llm/` + `agent/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class CompatTool:
    """Tiny compatibility wrapper with the subset expected by legacy callers."""

    name: str
    description: str
    _fn: Callable[[str], str]

    def run(self, query: str) -> str:
        return self._fn(query)


def get_langchain_tools() -> List[CompatTool]:
    from .guidelines_tool import guidelines_tool_fn
    from .tool_impls import arxiv_tool_fn, web_search_tool_fn
    from .unified_research import unified_research_tool_fn

    return [
        CompatTool("arxiv_search", "Search arXiv for papers", lambda q: arxiv_tool_fn(q)),
        CompatTool("web_search", "Search the web for recent sources", lambda q: web_search_tool_fn(q)),
        CompatTool("research_guidelines", "Retrieve research methodology guidance", lambda q: guidelines_tool_fn(q)),
        CompatTool("unified_research", "Merge paper and guideline grounding", lambda q: unified_research_tool_fn(q)),
    ]


__all__ = ["CompatTool", "get_langchain_tools"]
