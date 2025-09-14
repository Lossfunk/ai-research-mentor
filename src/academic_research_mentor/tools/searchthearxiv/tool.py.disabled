"""
SearchTheArXiv tool for semantic literature search.

Provides natural language search capabilities for arXiv papers using
searchthearxiv.com's semantic search interface.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..base_tool import BaseTool
from .scraper import SearchTheArxivScraper
from .parser import SearchTheArxivParser


class SearchTheArxivTool(BaseTool):
    """Tool for searching arXiv papers using semantic search via searchthearxiv.com."""
    
    name = "searchthearxiv_search"
    version = "1.0"
    
    def __init__(self) -> None:
        self.scraper = SearchTheArxivScraper()
        self.parser = SearchTheArxivParser()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the scraper."""
        self.scraper.initialize()
    
    def cleanup(self) -> None:
        """Clean up scraper."""
        self.scraper.cleanup()
    
    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this tool can handle literature search queries with semantic/natural language focus."""
        if not task_context:
            return False
            
        # Check for task goal or query text
        text = ""
        goal = task_context.get("goal", "")
        query = task_context.get("query", "")
        text = f"{goal} {query}".strip().lower()
        
        if not text:
            return False
        
        # Detect semantic/natural language search patterns
        semantic_patterns = [
            # Natural language indicators
            r'\b(find|search|look for|papers about|research on)\b.*\b(using|with|that|which)\b',
            r'\b(explain|describe|what is|how does)\b.*\b(research|paper|study)\b',
            r'\b(similar to|like|related to)\b.*\b(paper|research|work)\b',
            r'\b(recent|latest|current)\b.*\b(research|papers|studies)\b',
            r'\b(semantic|meaning|concept|idea)\b.*\b(search|find)\b',
            
            # Conversational search patterns
            r'\b(i want|i need|i\'m looking for)\b.*\b(papers|research|information)\b',
            r'\b(can you|help me)\b.*\b(find|search)\b',
            r'\b(papers that|research that|studies that)\b.*\b(deal with|address|discuss)\b',
            
            # Literature search specific
            r'\b(literature|papers|research|articles|publications)\b',
            r'\b(arxiv|arxiv)\b',
        ]
        
        # Check if query matches semantic patterns
        semantic_match = any(re.search(pattern, text, re.IGNORECASE) for pattern in semantic_patterns)
        
        # Check if query is conversational/natural language (longer, more descriptive)
        word_count = len(text.split())
        is_natural_language = word_count > 4 and any(
            indicator in text for indicator in [
                "find", "search", "look for", "papers about", "research on",
                "explain", "describe", "what is", "how does", "similar to",
                "i want", "i need", "can you", "help me"
            ]
        )
        
        return semantic_match or is_natural_language
    
    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute semantic search on searchthearxiv.com."""
        query = str(inputs.get("query", "")).strip()
        limit = int(inputs.get("limit", 10))
        
        if not query:
            return {
                "papers": [],
                "query": query,
                "total_papers": 0,
                "note": "Empty query provided"
            }
        
        try:
            # Perform the search
            html_content = self.scraper.search(query)
            
            if not html_content:
                return {
                    "papers": [],
                    "query": query,
                    "total_papers": 0,
                    "note": "No results found or search failed"
                }
            
            # Parse results
            papers = self.parser.parse_results(html_content, limit)
            
            return {
                "papers": papers,
                "query": query,
                "total_papers": len(papers),
                "source": "searchthearxiv",
                "note": f"Semantic search completed with {len(papers)} results"
            }
            
        except Exception as e:
            return {
                "papers": [],
                "query": query,
                "total_papers": 0,
                "note": f"SearchTheArxivTool execution failed: {e}",
                "error": str(e)
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for selection and usage."""
        meta = super().get_metadata()
        meta["identity"]["owner"] = "searchthearxiv"
        meta["capabilities"] = {
            "task_types": ["semantic_literature_search", "natural_language_search"],
            "domains": ["cs", "physics", "math", "q-bio", "q-fin", "stat", "eess"],
            "sources": ["arxiv"]
        }
        meta["io"] = {
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query for semantic search"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "papers": {"type": "array"},
                    "query": {"type": "string"},
                    "total_papers": {"type": "integer"},
                    "source": {"type": "string"},
                    "note": {"type": "string"}
                }
            }
        }
        meta["operational"] = {
            "cost_estimate": "low",
            "latency_profile": "5-15 seconds",
            "rate_limits": "Respectful scraping with delays",
            "reliability": "Variable (depends on external website availability)",
            "service_status": "Currently experiencing technical issues"
        }
        meta["quality"] = {
            "reliability_score": 0.4,  # Lower due to current service issues
            "confidence_estimation": True
        }
        meta["usage"] = {
            "ideal_inputs": [
                "Natural language queries about research topics",
                "Conversational search requests",
                "Semantic similarity searches"
            ],
            "anti_patterns": [
                "Very short keyword-only queries",
                "Non-research topics",
                "High-volume automated searches"
            ],
            "prerequisites": [
                "Internet connection",
                "Access to searchthearxiv.com"
            ]
        }
        return meta