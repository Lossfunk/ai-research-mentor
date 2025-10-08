from __future__ import annotations

from typing import Any, Dict, Optional
import os

from ..base_tool import BaseTool
from .providers import HTTPX_AVAILABLE, execute_openrouter_search, execute_tavily_search


class WebSearchTool(BaseTool):
    name = "web_search"
    version = "0.1"

    def __init__(self) -> None:
        self._client: Any = None
        self._config: Dict[str, Any] = {}
        self._init_error: Optional[str] = None
        self._default_limit = 8

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().initialize(config)
        self._config = dict(config or {})
        client = self._config.get("client")
        if client is not None:
            self._client = client
            self._init_error = None

    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        goal = str((task_context or {}).get("goal", "")).lower()
        keywords = (
            "web",
            "search",
            "recent",
            "news",
            "article",
            "updates",
            "blog",
            "resource",
        )
        return any(k in goal for k in keywords)

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception as exc:  # pragma: no cover - import guards
            self._init_error = f"tavily import failed: {exc}"
            return False

        api_key = str(self._config.get("api_key") or os.getenv("TAVILY_API_KEY", "")).strip()
        if not api_key:
            self._init_error = "TAVILY_API_KEY not configured"
            return False

        try:
            self._client = TavilyClient(api_key=api_key)
            self._init_error = None
            return True
        except Exception as exc:  # pragma: no cover - network/init errors
            self._client = None
            self._init_error = f"tavily client init failed: {exc}"
            return False

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta["identity"].update({"owner": "core", "name": self.name, "version": self.version})
        meta["capabilities"] = {
            "task_types": ["web_search", "literature_search"],
            "domains": ["ml", "ai", "cs", "news", "technology"],
        }
        meta["io"] = {
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Plain-language search query"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                    "mode": {"type": "string", "enum": ["fast", "focused", "exhaustive"]},
                    "domain": {"type": "string", "description": "Optional domain filter"},
                    "include_answer": {"type": "boolean"},
                },
                "required": ["query"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "summary": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                },
            },
        }
        meta["operational"] = {"cost_estimate": "medium", "latency_profile": "variable"}
        meta["usage"] = {
            "ideal_inputs": [
                "Recent events or emerging topics",
                "Queries requiring web sources or non-arXiv material",
            ],
            "anti_patterns": ["Empty query", "Requests for proprietary data"],
            "prerequisites": ["TAVILY_API_KEY or OPENROUTER_API_KEY"],
        }
        return meta

    def _normalize_mode(self, mode: str) -> str:
        m = mode.lower()
        if m in {"exhaustive", "deep", "advanced"}:
            return "advanced"
        return "basic"

    def is_available(self) -> bool:
        if self._client is not None:
            return True
        if self._ensure_client():
            return True
        api_key = str(self._config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if api_key and HTTPX_AVAILABLE:
            return True
        return False

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"results": [], "note": "empty query"}

        limit_raw = inputs.get("limit", self._default_limit)
        try:
            limit = max(1, min(int(limit_raw), 12))
        except Exception:
            limit = self._default_limit

        mode = str(inputs.get("mode", "fast") or "fast")
        search_depth = self._normalize_mode(mode)
        include_answer = bool(inputs.get("include_answer", True))
        domain = str(inputs.get("domain", "")).strip() or None

        tavily_result: Optional[Dict[str, Any]] = None
        tavily_error: Optional[str] = None
        if self._ensure_client():
            tavily_result, tavily_error = execute_tavily_search(
                self._client,
                query=query,
                limit=limit,
                search_depth=search_depth,
                include_answer=include_answer,
                domain=domain,
                mode=mode,
            )
        else:
            tavily_error = self._init_error or "tavily client unavailable"

        if tavily_result is not None:
            return tavily_result

        openrouter_result, openrouter_error = execute_openrouter_search(
            query=query,
            limit=limit,
            domain=domain,
            mode=mode,
            config=self._config,
        )
        if openrouter_result is not None:
            return openrouter_result

        note_parts = []
        if tavily_error:
            note_parts.append(f"Tavily: {tavily_error}")
        if openrouter_error:
            note_parts.append(f"OpenRouter: {openrouter_error}")
        note = "; ".join(note_parts) or "No providers available"

        return {
            "results": [],
            "query": query,
            "note": f"Web search unavailable ({note})",
            "metadata": {"provider": "unavailable", "mode": mode, "limit": limit, "domain": domain},
            "_degraded_mode": True,
        }
