"""
Guidelines tool for research mentoring advice.

Searches curated research guidance sources to provide evidence-based
academic mentoring advice on methodology, problem selection, and research taste.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..base_tool import BaseTool
from .config import GuidelinesConfig
from .cache import GuidelinesCache, CostTracker
from .evidence_collector import EvidenceCollector
from .query_builder import QueryBuilder
from .formatter import GuidelinesFormatter
from .tool_metadata import ToolMetadata
from .citation_handler import GuidelinesCitationHandler


class GuidelinesTool(BaseTool):
    """Tool for searching research guidelines and providing mentoring advice."""
    
    name = "research_guidelines"
    version = "1.0"
    
    def __init__(self) -> None:
        self.config = GuidelinesConfig()
        self._search_tool = None
        self._cache = None
        self._cost_tracker = None
        self._evidence_collector = None
        self._query_builder = None
        self._formatter = None
        self._metadata_handler = None
        self._citation_handler = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the guidelines tool with optional configuration."""
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self._search_tool = DuckDuckGoSearchRun()
        except ImportError:
            # Graceful fallback if DuckDuckGo search not available
            self._search_tool = None
        
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
        guidelines_patterns = [
            r'\b(research\s+methodology|problem\s+selection|research\s+taste)\b',
            r'\b(academic\s+advice|phd\s+guidance|research\s+strategy)\b',
            r'\b(how\s+to\s+choose|develop\s+taste|research\s+skills)\b',
            r'\b(research\s+best\s+practices|methodology\s+advice)\b',
            r'\b(academic\s+career|research\s+planning|project\s+selection)\b',
            r'\b(hamming|effective\s+research|research\s+principles)\b',
            r'\bphd\b|\bcareer\s+guidance\b|\bmentoring\b|\bacademic\s+guidance\b',
            r'\bresearch\s+advice\b|\bgraduate\s+school\b|\bacademic\s+career\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in guidelines_patterns)
    
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
        if not self._search_tool and not self.config.FF_GUIDELINES_V2:
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
            return cached_result
        
        # Record cache miss
        if self._cost_tracker:
            self._cost_tracker.record_cache_miss()
        
        try:
            if self.config.FF_GUIDELINES_V2:
                return self._execute_v2(topic, mode, max_per_source, response_format, page_size, next_token, cache_key)
            else:
                return self._execute_v1(topic, cache_key)
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

    def _execute_v2(self, topic: str, mode: str, max_per_source: int, 
                   response_format: str, page_size: int, next_token: Optional[str], 
                   cache_key: str) -> Dict[str, Any]:
        """Execute V2 structured evidence flow."""
        # Always include curated evidence; optionally add search-based evidence
        curated = self._evidence_collector.collect_curated_evidence(topic)
        evidence: List[Dict[str, Any]] = list(curated)
        sources_covered = sorted(list({e.get("domain", "") for e in curated if e.get("domain")}))
        
        if self._search_tool:
            searched, covered = self._evidence_collector.collect_structured_evidence(topic, mode, max_per_source)
            evidence.extend(searched)
            for d in covered:
                if d not in sources_covered:
                    sources_covered.append(d)
        
        if not evidence:
            result_v2 = {
                "topic": topic,
                "total_evidence": 0,
                "sources_covered": sources_covered,
                "evidence": [],
                "pagination": {"has_more": False, "next_token": None},
                "cached": False,
                "note": "No evidence found"
            }
            if self._cache:
                self._cache.set(cache_key, result_v2)
            return result_v2

        result_v2 = self._formatter.format_v2_response(topic, evidence, sources_covered, response_format, page_size, next_token)
        
        # Add citation validation and formatting
        result_v2 = self._citation_handler.add_citation_metadata(result_v2, evidence)
        
        if self._cache:
            self._cache.set(cache_key, result_v2)
        return result_v2

    def _execute_v1(self, topic: str, cache_key: str) -> Dict[str, Any]:
        """Execute V1 fallback flow."""
        search_queries = self._query_builder.get_prioritized_queries(topic)[:self.config.MAX_SEARCH_QUERIES]
        retrieved_guidelines = []
        
        for query_str in search_queries:
            try:
                results = self._search_tool.run(query_str)
                if results and len(results) > 20:
                    source_type = self._query_builder.identify_source_type(query_str)
                    guide_id = f"guide_{hash(results) & 0xfffff:05x}"
                    source_domain = self._query_builder.extract_domain(query_str)
                    retrieved_guidelines.append({
                        "guide_id": guide_id,
                        "source_type": source_type,
                        "source_domain": source_domain,
                        "content": results[:500],
                        "search_query": query_str
                    })
                    if self._cost_tracker:
                        self._cost_tracker.record_search_query(0.01)
            except Exception:
                continue

        if not retrieved_guidelines:
            result = {
                "retrieved_guidelines": [],
                "formatted_content": f"No guidelines found for '{topic}' in curated sources. Try rephrasing your query.",
                "total_guidelines": 0,
                "note": "No guidelines retrieved from search",
                "cached": False
            }
            if self._cache:
                self._cache.set(cache_key, result)
            return result

        formatted_content = self._formatter.format_guidelines_for_agent(topic, retrieved_guidelines)
        result = {
            "retrieved_guidelines": retrieved_guidelines,
            "formatted_content": formatted_content,
            "total_guidelines": len(retrieved_guidelines),
            "topic": topic,
            "note": f"Retrieved {len(retrieved_guidelines)} relevant guidelines for agent reasoning",
            "cached": False
        }
        if self._cache:
            self._cache.set(cache_key, result)
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for selection and usage."""
        return self._metadata_handler.get_metadata()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and cost statistics."""
        return self._metadata_handler.get_cache_stats()
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all cached results."""
        return self._metadata_handler.clear_cache()
    