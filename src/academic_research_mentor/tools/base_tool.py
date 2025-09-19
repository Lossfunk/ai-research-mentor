from __future__ import annotations

"""
Base tool interface (WS2).

Contract (kept compact and under 200 LOC):
- Attributes: `name`, `version`
- Lifecycle: `initialize(config)`, `cleanup()`
- Selection: `can_handle(task_context) -> bool`
- Execution: `execute(inputs, context) -> {}`
- Metadata: `get_metadata() -> {identity, capabilities, io, operational, quality, usage}`
"""

from typing import Any, Dict, Optional


class BaseTool:
    """Common interface for all tools.

    Subclasses should provide sensible defaults and override as needed.
    """

    name: str = "tool"
    version: str = "0.1"

    # ---- Lifecycle ----
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:  # noqa: ARG002
        """Optional initialization hook (e.g., create clients)."""

    def cleanup(self) -> None:
        """Optional cleanup hook (e.g., close sessions)."""

    # ---- Selection ----
    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:  # noqa: ARG002
        """Return True when the tool is a plausible match for the task."""
        return True

    # ---- Execution ----
    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # noqa: ARG002
        raise NotImplementedError

    # ---- Metadata ----
    def get_metadata(self) -> Dict[str, Any]:
        """Return capability metadata used for selection and UI.

        Minimal defaults; concrete tools should augment these.
        """
        return {
            "identity": {"name": self.name, "version": self.version, "owner": "unknown"},
            "capabilities": {"task_types": [], "domains": []},
            "io": {"input_schema": {}, "output_schema": {}},
            "operational": {"cost_estimate": "unknown", "latency_profile": "unknown", "rate_limits": None},
            "quality": {"reliability_score": None, "confidence_estimation": False},
            "usage": {"ideal_inputs": [], "anti_patterns": [], "prerequisites": []},
            "citations": {
                "supports_citations": False,
                "citation_format": "none",
                "citation_validation": False,
                "citation_aggregation": False
            },
        }
