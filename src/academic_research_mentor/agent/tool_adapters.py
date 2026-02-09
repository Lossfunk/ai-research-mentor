"""Adapters to bridge legacy BaseTool implementations to the new ToolRegistry."""

from __future__ import annotations

from typing import Any

from .tools import Tool


class WebSearchToolAdapter(Tool):
    """Adapter for the WebSearchTool."""
    
    def __init__(self):
        from academic_research_mentor.tools.web_search.tool import WebSearchTool
        self._tool = WebSearchTool()
        self._tool.initialize()
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return (
            "Search the web for recent information, news, articles, and resources. "
            "Use this for current events, recent developments, blog posts, or when you need "
            "up-to-date information that may not be in academic papers."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (1-12)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)
        
        if not query:
            return "Error: No search query provided"
        
        result = self._tool.execute({"query": query, "limit": limit})
        
        # Format results for the LLM
        results = result.get("results", [])
        if not results:
            note = result.get("note", "No results found")
            return f"Web search returned no results. {note}"
        
        formatted = []
        for i, r in enumerate(results[:limit], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", r.get("snippet", ""))[:300]
            formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")
        
        return "\n\n".join(formatted)


class ArxivSearchToolAdapter(Tool):
    """Adapter for the ArxivSearchTool."""
    
    def __init__(self):
        from academic_research_mentor.tools.legacy.arxiv.tool import ArxivSearchTool
        self._tool = ArxivSearchTool()
        self._tool.initialize()
    
    @property
    def name(self) -> str:
        return "arxiv_search"
    
    @property
    def description(self) -> str:
        return (
            "Search arXiv for academic papers in computer science, machine learning, AI, "
            "physics, mathematics, and other scientific fields. Use this when looking for "
            "research papers, preprints, or academic literature."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for academic papers"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-20)",
                    "default": 5
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "date"],
                    "description": "Sort order. Use 'date' to find the absolute latest papers.",
                    "default": "relevance"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)
        sort_by = kwargs.get("sort_by", "relevance")
        
        if not query:
            return "Error: No search query provided"
        
        result = self._tool.execute({"query": query, "limit": limit, "sort_by": sort_by})
        
        # Format results for the LLM
        papers = result.get("papers", [])
        if not papers:
            note = result.get("note", "No papers found")
            return f"arXiv search returned no results. {note}"
        
        formatted = []
        for i, p in enumerate(papers[:limit], 1):
            title = p.get("title", "Untitled")
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            url = p.get("url", "")
            year = p.get("year", "")
            published = p.get("published", "")
            summary = (p.get("summary", "") or "")[:400]
            
            # Show full published date if available, otherwise year
            date_str = f"Published: {published}" if published else f"Year: {year}"
            
            formatted.append(
                f"{i}. **{title}**\n"
                f"   Authors: {authors}\n"
                f"   {date_str} | URL: {url}\n"
                f"   {summary}"
            )
        
        return "\n\n".join(formatted)


class GuidelinesToolAdapter(Tool):
    """Adapter for the GuidelinesTool (research guidelines)."""
    
    def __init__(self):
        from academic_research_mentor.tools.guidelines.tool import GuidelinesTool
        self._tool = GuidelinesTool()
        self._tool.initialize()
    
    @property
    def name(self) -> str:
        return "research_guidelines"
    
    @property
    def description(self) -> str:
        return (
            "Search for research guidelines, methodology advice, and best practices in academic research. "
            "Use this when the user needs advice on research methods, experimental design, writing practices, "
            "or general guidance on how to conduct research effectively."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic or methodology question"
                },
                "topic": {
                    "type": "string",
                    "description": "Specific topic area (optional, defaults to query)"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        topic = kwargs.get("topic", query)
        
        if not query:
            return "Error: No query provided for guidelines search"
        
        result = self._tool.execute({"query": query, "topic": topic})
        
        # Format results for the LLM
        formatted_content = result.get("formatted_content", "")
        if formatted_content:
            return formatted_content
        
        guidelines = result.get("retrieved_guidelines", [])
        if not guidelines:
            note = result.get("note", "No guidelines found")
            return f"No research guidelines found. {note}"
        
        formatted = []
        for i, g in enumerate(guidelines[:5], 1):
            title = g.get("title", "Guideline")
            content = g.get("content", g.get("snippet", ""))[:500]
            source = g.get("source", "")
            formatted.append(f"{i}. **{title}**\n   Source: {source}\n   {content}")
        
        return "\n\n".join(formatted)


class CitationIntegrityToolAdapter(Tool):
    """Audits references for broken links and malformed metadata."""

    @property
    def name(self) -> str:
        return "citation_integrity_audit"

    @property
    def description(self) -> str:
        return (
            "Audit citations/references for integrity. Extracts URLs/DOIs/arXiv IDs, "
            "checks for dead links, validates DOI/arXiv existence, and flags weak BibTeX entries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reference_text": {
                    "type": "string",
                    "description": "References section, BibTeX, or text containing citations/links.",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional additional URLs to verify.",
                },
                "check_urls": {"type": "boolean", "default": True},
                "verify_doi": {"type": "boolean", "default": True},
                "verify_arxiv": {"type": "boolean", "default": False},
            },
            "required": ["reference_text"],
        }

    def execute(self, **kwargs: Any) -> str:
        text = str(kwargs.get("reference_text", "")).strip()
        if not text:
            return "Citation audit failed: no reference_text provided."

        from academic_research_mentor.citations.integrity import audit_reference_text, format_integrity_report

        report = audit_reference_text(
            text,
            extra_urls=kwargs.get("urls") if isinstance(kwargs.get("urls"), list) else None,
            check_urls=bool(kwargs.get("check_urls", True)),
            verify_doi=bool(kwargs.get("verify_doi", True)),
            verify_arxiv=bool(kwargs.get("verify_arxiv", False)),
        )
        return format_integrity_report(report)


def create_default_tools() -> list[Tool]:
    """Create all default tools for the mentor agent."""
    tools = []
    
    # Try to create each tool, skip if it fails to initialize
    try:
        tools.append(WebSearchToolAdapter())
    except Exception as e:
        print(f"Warning: Could not initialize web_search tool: {e}")
    
    try:
        tools.append(ArxivSearchToolAdapter())
    except Exception as e:
        print(f"Warning: Could not initialize arxiv_search tool: {e}")
    
    try:
        tools.append(GuidelinesToolAdapter())
    except Exception as e:
        print(f"Warning: Could not initialize research_guidelines tool: {e}")

    try:
        tools.append(CitationIntegrityToolAdapter())
    except Exception as e:
        print(f"Warning: Could not initialize citation_integrity_audit tool: {e}")
    
    return tools
