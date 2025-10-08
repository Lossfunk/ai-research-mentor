"""
Guidelines tool for research mentoring advice.

Searches curated research guidance sources to provide evidence-based
academic mentoring advice on methodology, problem selection, and research taste.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..base_tool import BaseTool
from .cache import CostTracker, GuidelinesCache
from .config import GuidelinesConfig
from .citation_handler import GuidelinesCitationHandler
from .evidence_collector import EvidenceCollector
from .executors import GuidelinesV1Executor, GuidelinesV2Executor
from .formatter import GuidelinesFormatter
from .query_builder import QueryBuilder
from .search_providers import (
    BaseSearchProvider,
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
)
from .tool_metadata import ToolMetadata
from .utils import enforce_domain_cap, matches_guidelines_query


class GuidelinesTool(BaseTool):
    """Tool for searching research guidelines and providing mentoring advice."""
    
    name = "research_guidelines"
    version = "1.0"
    
    def __init__(self) -> None:
        self.config = GuidelinesConfig()
        self._search_tool: Optional[BaseSearchProvider] = None
        self._cache: Optional[GuidelinesCache] = None
        self._cost_tracker: Optional[CostTracker] = None
        self._evidence_collector: Optional[EvidenceCollector] = None
        self._query_builder: Optional[QueryBuilder] = None
        self._formatter: Optional[GuidelinesFormatter] = None
        self._metadata_handler: Optional[ToolMetadata] = None
        self._citation_handler: Optional[GuidelinesCitationHandler] = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the guidelines tool with optional configuration."""
        search_provider: Optional[BaseSearchProvider] = None
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if api_key:
            try:
                search_provider = TavilySearchProvider(api_key=api_key)
            except Exception:
                search_provider = None

        if search_provider is None:
            try:
                search_provider = DuckDuckGoSearchProvider()
            except Exception:
                search_provider = None

        self._search_tool = search_provider
        
        # Initialize caching and cost tracking
        self._cache = GuidelinesCache(self.config)
        self._cost_tracker = self._cache.cost_tracker
        
        # Initialize helper classes
        self._evidence_collector = EvidenceCollector(self.config, self._search_tool, self._cost_tracker)
        self._query_builder = QueryBuilder(self.config)
        self._formatter = GuidelinesFormatter(self.config)
        self._metadata_handler = ToolMetadata(self.config, self._cache, self._cost_tracker)
        self._citation_handler = GuidelinesCitationHandler()

    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this tool can handle research guidelines queries."""
        if not task_context:
            return False
            
        # Check for task goal or query text
        text = ""
        goal = task_context.get("goal", "")
        query = task_context.get("query", "")
        text = f"{goal} {query}".strip().lower()
        
        if not text:
            return False
            
        # Detect research guidance patterns
        return matches_guidelines_query(text)
    
    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search research guidelines and return evidence or v1 formatted content.

        V2 (preferred when FF_GUIDELINES_V2): returns structured evidence with pagination
        and provenance fields. V1 returns a formatted string and a list of truncated blobs.
        """
        query = str(inputs.get("query", "")).strip()
        topic = str(inputs.get("topic", query)).strip()
        response_format = str(inputs.get("response_format", self.config.RESPONSE_FORMAT_DEFAULT)).strip().lower()
        mode = str(inputs.get("mode", self.config.DEFAULT_MODE)).strip().lower()
        max_per_source = int(inputs.get("max_per_source", self.config.DEFAULT_MAX_PER_SOURCE))
        page_size = int(inputs.get("page_size", getattr(self.config, "RESPONSE_PAGE_SIZE_DEFAULT", 10)))
        next_token = str(inputs.get("next_token", "")).strip() or None
        
        if not topic:
            return {
                "retrieved_guidelines": [],
                "formatted_content": "No topic provided for guidelines search.",
                "total_guidelines": 0,
                "note": "Empty query provided"
            }
        
        # In V2 mode we can operate without a search tool using curated sources only.
        if (not self._search_tool or not getattr(self._search_tool, "supports_text", False)) and not self.config.FF_GUIDELINES_V2:
            return {
                "retrieved_guidelines": [],
                "formatted_content": f"Guidelines search unavailable for topic: {topic}",
                "total_guidelines": 0,
                "note": "Search tool not available"
            }
        
        # Check cache first
        cache_key = f"{query}:{topic}:{mode}:{response_format}:{max_per_source}:{page_size}:{next_token or ''}"
        cached_result = self._cache.get(cache_key) if self._cache else None
        
        if cached_result:
            # Add cache note and return cached result
            cached_result["cached"] = True
            cached_result["cache_note"] = "Result served from cache"
            return enforce_domain_cap(cached_result, max_per_source)
        
        # Record cache miss
        if self._cost_tracker:
            self._cost_tracker.record_cache_miss()
        
        try:
            if self.config.FF_GUIDELINES_V2:
                if not (
                    self._evidence_collector
                    and self._formatter
                    and self._citation_handler
                ):
                    raise RuntimeError("Guidelines V2 dependencies not initialized")
                executor = GuidelinesV2Executor(
                    self._evidence_collector,
                    self._formatter,
                    self._citation_handler,
                    self._cache,
                    self._search_tool,
                )
                return enforce_domain_cap(
                    executor.run(
                    topic,
                    mode,
                    max_per_source,
                    response_format,
                    page_size,
                    next_token,
                        cache_key,
                    ),
                    max_per_source,
                )
            if not (self._query_builder and self._formatter):
                raise RuntimeError("Guidelines V1 dependencies not initialized")
            executor = GuidelinesV1Executor(
                self.config,
                self._search_tool,
                self._query_builder,
                self._formatter,
                self._cache,
                self._cost_tracker,
            )
            return enforce_domain_cap(executor.run(topic, cache_key), max_per_source)
        except Exception as e:
            error_result = {
                "retrieved_guidelines": [],
                "formatted_content": f"Error searching guidelines: {e}",
                "total_guidelines": 0,
                "error": str(e),
                "note": "Error in guidelines search",
                "cached": False
            }
            return error_result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for selection and usage."""
        meta = self._metadata_handler.get_metadata()
        provider = self._search_tool
        if provider:
            operational = meta.setdefault("operational", {})
            operational["search_provider"] = provider.__class__.__name__
            operational["supports_structured_search"] = bool(getattr(provider, "supports_structured", False))
        else:
            meta.setdefault("operational", {})["search_provider"] = "none"
            meta["operational"]["supports_structured_search"] = False
        return meta
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and cost statistics."""
        return self._metadata_handler.get_cache_stats()
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all cached results."""
        return self._metadata_handler.clear_cache()
    