from __future__ import annotations

"""
Agent scaffolding (WS1).

This module will eventually host agent decision-making logic and integrate with
orchestrator and tools registry. For now, it provides a tiny placeholder to
avoid import errors and to make the package structure explicit.
"""

from typing import Any, Optional, Tuple


class AgentPlaceholder:
    """Non-functional agent placeholder.

    Retained to make the future migration path explicit without changing runtime
    behavior that currently lives in `runtime.build_agent`.
    """

    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.version = "0.1"

    def run(self, text: str) -> str:
        return "[AgentPlaceholder] Not yet implemented."


def build_agent_placeholder(instructions: str) -> Tuple[Optional[AgentPlaceholder], Optional[str]]:
    """Return a no-op agent placeholder and a note.

    The real agent builder remains in `runtime.build_agent` until WS3.
    """
    return AgentPlaceholder(instructions), "placeholder"
