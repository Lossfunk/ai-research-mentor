from __future__ import annotations

from typing import Any, Optional, Tuple

from .models import resolve_model
from .agents.react_agent import LangChainReActAgentWrapper
from .tools_wrappers import get_langchain_tools


def build_agent(instructions: str) -> Tuple[Optional[Any], Optional[str]]:
    llm, reason = resolve_model()
    if llm is None:
        return None, reason

    tools = get_langchain_tools()
    agent = LangChainReActAgentWrapper(llm=llm, system_instructions=instructions, tools=tools)
    return agent, None


__all__ = ["build_agent"]
