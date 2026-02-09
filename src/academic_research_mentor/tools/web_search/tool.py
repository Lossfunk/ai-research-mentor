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
        self._dotenv_cache: Optional[Dict[str, str]] = None
        self._default_limit = 8

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().initialize(config)
        self._config = dict(config or {})
        self._dotenv_cache = None
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

    def _resolve_api_key(self, config_key: str, env_key: str) -> str:
        configured = str(self._config.get(config_key, "")).strip()
        if configured:
            return configured

        from_env = os.getenv(env_key, "").strip()
        if from_env:
            return from_env

        cached = self._load_dotenv_cache().get(env_key, "").strip()
        if cached and not os.getenv(env_key):
            os.environ[env_key] = cached
        return cached

    def _load_dotenv_cache(self) -> Dict[str, str]:
        if self._dotenv_cache is not None:
            return self._dotenv_cache

        values: Dict[str, str] = {}
        dotenv_path = self._find_dotenv_path()
        if dotenv_path:
            try:
                with open(dotenv_path, "r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[7:].strip()
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if not key:
                            continue
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                            value = value[1:-1]
                        values[key] = value
            except Exception:
                values = {}

        self._dotenv_cache = values
        return values

    def _find_dotenv_path(self) -> Optional[str]:
        if bool(self._config.get("disable_dotenv_lookup")):
            return None
        configured = str(self._config.get("dotenv_path", "")).strip()
        if configured and os.path.isfile(configured):
            return configured

        current = os.path.abspath(os.getcwd())
        while True:
            candidate = os.path.join(current, ".env")
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception as exc:  # pragma: no cover - import guards
            self._init_error = f"tavily import failed: {exc}"
            return False

        api_key = self._resolve_api_key("api_key", "TAVILY_API_KEY")
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
        api_key = self._resolve_api_key("openrouter_api_key", "OPENROUTER_API_KEY")
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

        openrouter_key = self._resolve_api_key("openrouter_api_key", "OPENROUTER_API_KEY")
        config = dict(self._config)
        if openrouter_key and not config.get("openrouter_api_key"):
            config["openrouter_api_key"] = openrouter_key

        openrouter_result, openrouter_error = execute_openrouter_search(
            query=query,
            limit=limit,
            domain=domain,
            mode=mode,
            config=config,
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
