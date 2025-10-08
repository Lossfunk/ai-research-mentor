from __future__ import annotations

from typing import Any, Dict, Optional

from ..base_tool import BaseTool


class SearchTheArxivTool(BaseTool):
    name = "searchthearxiv_search"
    version = "0.1"

    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        goal = str((task_context or {}).get("goal", "")).lower()
        return any(keyword in goal for keyword in ("search", "find", "arxiv", "paper", "literature"))

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"query": query, "results": [], "note": "empty query"}
        return {
            "query": query,
            "results": [],
            "note": "searchthearxiv tool currently disabled; no results returned",
        }

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta["identity"].update({"name": self.name, "owner": "core", "version": self.version})
        meta["capabilities"] = {
            "task_types": ["literature_search"],
            "domains": ["ml", "ai", "cs"],
        }
        meta["io"] = {
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "note": {"type": "string"},
                },
            },
        }
        meta["operational"] = {"cost_estimate": "low", "latency_profile": "fast"}
        meta["usage"] = {
            "ideal_inputs": ["natural-language research queries"],
            "anti_patterns": ["empty query"],
            "prerequisites": [],
        }
        return meta
