"""
Tool metadata and cache management utilities for guidelines tool.

Handles metadata generation and cache operations.
"""

from typing import Any, Dict

from .config import GuidelinesConfig


class ToolMetadata:
    """Handles tool metadata and cache operations."""
    
    def __init__(self, config: GuidelinesConfig, cache: Any, cost_tracker: Any):
        self.config = config
        self._cache = cache
        self._cost_tracker = cost_tracker
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for selection and usage."""
        meta = {
            "identity": {
                "name": "research_guidelines",
                "version": "1.0",
                "owner": "guidelines"
            },
            "capabilities": {
                "task_types": ["research_advice", "methodology_guidance", "academic_mentoring"],
                "domains": ["research_methodology", "academic_career", "problem_selection", "research_taste"]
            },
            "io": {
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Research question or topic"},
                        "topic": {"type": "string", "description": "Specific area for guidance"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "retrieved_guidelines": {"type": "array"},
                        "formatted_content": {"type": "string"},
                        "total_guidelines": {"type": "integer"},
                        "cached": {"type": "boolean"},
                        "cache_note": {"type": "string"},
                        "citations": {"type": "object", "description": "Structured citations with validation"},
                        "citation_quality": {"type": "object", "description": "Citation quality metrics"}
                    }
                }
            },
            "operational": {
                "cost_estimate": "low-medium",
                "latency_profile": "5-10 seconds",
                "rate_limits": "3 searches per query",
                "caching_enabled": self.config.ENABLE_CACHING,
                "cache_ttl_hours": self.config.CACHE_TTL_HOURS if self.config.ENABLE_CACHING else None
            },
            "usage": {
                "ideal_inputs": ["research methodology questions", "academic career advice", "problem selection guidance"],
                "anti_patterns": ["very broad questions", "non-research topics"],
                "prerequisites": ["internet connection for search"]
            },
            "citations": {
                "supports_citations": True,
                "citation_format": "structured",
                "citation_validation": True,
                "citation_aggregation": True,
                "citation_sources": ["curated", "search"],
                "citation_quality_metrics": ["completeness", "validity", "relevance"]
            }
        }
        return meta
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and cost statistics."""
        if not self._cost_tracker:
            return {"error": "Cost tracker not initialized"}
        
        stats = self._cost_tracker.get_stats()
        stats["cache_hit_rate"] = self._cost_tracker.get_cache_hit_rate()
        return stats
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all cached results."""
        if not self._cache:
            return {"error": "Cache not initialized"}
        
        old_stats = self._cost_tracker.get_stats() if self._cost_tracker else {}
        self._cache.clear()
        
        return {
            "message": "Cache cleared successfully",
            "old_stats": old_stats,
            "cache_enabled": self.config.ENABLE_CACHING
        }
