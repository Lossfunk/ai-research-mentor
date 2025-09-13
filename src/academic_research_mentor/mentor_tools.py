from __future__ import annotations

from typing import Any, Dict, List

# Re-exports for compatibility
from .tools.legacy.arxiv.client import arxiv_search as arxiv_search  # noqa: F401
from .tools.utils.math import math_ground as math_ground  # noqa: F401
from .tools.utils.methodology import methodology_validate as methodology_validate  # noqa: F401


GEMINI_FUNCTION_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "arxiv_search",
        "description": "Search arXiv for relevant papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "from_year": {"type": "number", "description": "Minimum publication year"},
                "limit": {"type": "number", "description": "Max results (≤25)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "math_ground",
        "description": "Heuristic math grounding: assumptions, glossary, proof skeleton.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_or_math": {"type": "string", "description": "TeX or plain text"},
                "options": {
                    "type": "object",
                    "properties": {
                        "dimensional_check": {"type": "boolean"},
                        "assumptions_check": {"type": "boolean"},
                    },
                },
            },
            "required": ["text_or_math"],
        },
    },
    {
        "name": "methodology_validate",
        "description": "Heuristic validation of experiment plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "Design summary"},
                "checklist": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["plan"],
        },
    },
]


def get_gemini_tools_block() -> List[Dict[str, Any]]:
    return [{"function_declarations": GEMINI_FUNCTION_DECLARATIONS}]


def handle_mentor_function_call(function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
    if function_name == "arxiv_search":
        return arxiv_search(
            query=str(function_args.get("query", "")),
            from_year=function_args.get("from_year"),
            limit=int(function_args.get("limit", 10)),
        )
    if function_name == "math_ground":
        return math_ground(
            text_or_math=str(function_args.get("text_or_math", "")),
            options=function_args.get("options"),
        )
    if function_name == "methodology_validate":
        return methodology_validate(
            plan=str(function_args.get("plan", "")),
            checklist=function_args.get("checklist"),
        )
    return {"error": f"Unknown function: {function_name}"}
