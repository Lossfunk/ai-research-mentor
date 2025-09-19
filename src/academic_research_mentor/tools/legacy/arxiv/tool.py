from __future__ import annotations

from typing import Any, Dict, Optional, List

from ...base_tool import BaseTool
from ....mentor_tools import arxiv_search as legacy_arxiv_search
from ....citations import Citation, CitationFormatter


class ArxivSearchTool(BaseTool):
    name = "legacy_arxiv_search"
    version = "0.1"

    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:  # type: ignore[override]
        tc = (task_context or {}).get("goal", "")
        return any(k in str(tc).lower() for k in ("arxiv", "paper", "literature"))

    def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        meta = super().get_metadata()
        meta["identity"]["owner"] = "legacy"
        meta["capabilities"] = {"task_types": ["literature_search"], "domains": ["cs", "ml"]}
        meta["io"] = {
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "output_schema": {"type": "object", "properties": {"papers": {"type": "array"}}},
        }
        meta["operational"] = {"cost_estimate": "low", "latency_profile": "low", "rate_limits": None}
        meta["usage"] = {"ideal_inputs": ["concise topic"], "anti_patterns": [], "prerequisites": []}
        return meta

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore[override]
        q = str(inputs.get("query", "")).strip()
        if not q:
            return {"papers": [], "note": "empty query"}
        result = legacy_arxiv_search(query=q, from_year=None, limit=int(inputs.get("limit", 10)))

        # Build lightweight citations using URL strategy (consistent with guidelines tool)
        papers: List[Dict[str, Any]] = result.get("papers", []) if isinstance(result, dict) else []
        citations: List[Citation] = []
        for p in papers:
            url = str(p.get("url", "")).strip()
            title = str(p.get("title", "")).strip() or "Untitled"
            cid = f"arxiv_{abs(hash(url or title)) & 0xfffffff:x}"
            citations.append(Citation(
                id=cid,
                title=title,
                url=url or "https://arxiv.org",
                source="arxiv",
                authors=[str(a) for a in p.get("authors", []) if a],
                year=p.get("year"),
                venue=p.get("venue", "arXiv"),
                snippet=(p.get("summary") or "")[:300] or None,
            ))

        if citations:
            formatter = CitationFormatter()
            result["citations"] = formatter.to_output_block(citations)

        return result
