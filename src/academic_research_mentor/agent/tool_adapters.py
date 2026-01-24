"""Adapters to bridge legacy BaseTool implementations to the new ToolRegistry."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .tools import Tool

# Import attachments at module level to preserve state
try:
    from academic_research_mentor.attachments import (
        has_attachments,
        search as search_attachments,
        get_document_summary,
    )
    _ATTACHMENTS_AVAILABLE = True
except ImportError:
    _ATTACHMENTS_AVAILABLE = False


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


class AttachmentsSearchToolAdapter(Tool):
    """Adapter for searching attached PDF documents."""

    @property
    def name(self) -> str:
        return "attachments_search"

    @property
    def description(self) -> str:
        return (
            "Search through attached PDF documents (papers, reports, etc.) that the user has provided. "
            "Use this to find specific information, methods, results, or details from the attached documents. "
            "This tool searches the full text of all attached PDFs and returns relevant excerpts with page numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant sections in attached documents"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-20)",
                    "default": 8
                }
            },
            "required": ["query"]
        }

    def execute(self, **kwargs: Any) -> str:
        if not _ATTACHMENTS_AVAILABLE:
            return "Error: Attachments module is not available. Please check installation."

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 8)

        if not query:
            return "Error: No search query provided"

        # Check if attachments are loaded
        if not has_attachments():
            return (
                "No documents are currently attached. Please ask the user to provide "
                "the PDF documents you need to search."
            )

        # Get document summary for context
        doc_summary = get_document_summary()

        # Search attachments
        results = search_attachments(query, k=limit)

        if not results:
            return (
                f"No results found in attached documents for query: '{query}'\n\n"
                f"Document Overview:\n{doc_summary}" if doc_summary else
                f"No results found in attached documents for query: '{query}'"
            )

        # Format results for the LLM
        formatted = []

        # Add document summary if available (gives context about what's in the docs)
        if doc_summary:
            formatted.append(f"**Document Overview:**\n{doc_summary}\n")

        formatted.append(f"**Search Results for '{query}':**\n")

        for i, result in enumerate(results[:limit], 1):
            file_name = result.get("file", "document.pdf")
            page = result.get("page", 1)
            snippet = result.get("snippet", result.get("text", ""))[:400]
            anchor = result.get("anchor", f"{file_name}#page={page}")

            formatted.append(
                f"{i}. **{file_name}** (Page {page})\n"
                f"   Location: {anchor}\n"
                f"   {snippet}"
            )

        return "\n\n".join(formatted)


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
        tools.append(AttachmentsSearchToolAdapter())
    except Exception as e:
        print(f"Warning: Could not initialize attachments_search tool: {e}")

    return tools
