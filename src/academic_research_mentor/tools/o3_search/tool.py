from __future__ import annotations

from typing import Any, Dict, Optional

from ..base_tool import BaseTool


class O3SearchTool(BaseTool):
    name = "o3_search"

    def __init__(self) -> None:
        # Placeholder for client wiring (will use literature_review.o3_client later)
        pass

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Scaffold implementation only
        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"results": [], "note": "empty query"}
        return {
            "results": [],
            "query": query,
            "note": "O3SearchTool scaffold; real search to be implemented in WS2/WS3",
        }
