from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from .models import resolve_model
from .agents.base_chat import LangChainAgentWrapper
from .agents.react_agent import LangChainReActAgentWrapper
from .agents.router_agent import LangChainSpecialistRouterWrapper
from .tools_wrappers import get_langchain_tools


def build_agent(instructions: str) -> Tuple[Optional[Any], Optional[str]]:
    llm, reason = resolve_model()
    if llm is None:
        return None, reason

    # Toggle ReAct agent via env var; default to react for fresh clones
    mode = os.environ.get("LC_AGENT_MODE", "react").strip().lower()
    if mode in {"react", "tools"}:
        tools = get_langchain_tools()
        agent = LangChainReActAgentWrapper(llm=llm, system_instructions=instructions, tools=tools)
    elif mode in {"router", "route"}:
        agent = LangChainSpecialistRouterWrapper(llm=llm, system_instructions=instructions)
    else:
        # We keep manual tool routing in the CLI; here we just build a chat agent
        agent = LangChainAgentWrapper(llm=llm, system_instructions=instructions)
    return agent, None


__all__ = ["build_agent"]
