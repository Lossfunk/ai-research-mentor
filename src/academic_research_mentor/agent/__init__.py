"""Agent module - Simple agent with tool calling."""

from .agent import MentorAgent
from .tools import Tool, ToolRegistry
from .tool_adapters import (
    WebSearchToolAdapter,
    ArxivSearchToolAdapter,
    GuidelinesToolAdapter,
    CitationIntegrityToolAdapter,
    create_default_tools,
)

__all__ = [
    "MentorAgent",
    "Tool",
    "ToolRegistry",
    "WebSearchToolAdapter",
    "ArxivSearchToolAdapter",
    "GuidelinesToolAdapter",
    "CitationIntegrityToolAdapter",
    "create_default_tools",
]
