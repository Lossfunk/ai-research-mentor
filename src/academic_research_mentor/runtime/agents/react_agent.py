from __future__ import annotations

import os
from typing import Any, List


class LangChainReActAgentWrapper:
    """Small compatibility wrapper around langgraph's create_react_agent.

    Keeps history behavior required by legacy tests while new runtime paths use
    the modern `MentorAgent`.
    """

    def __init__(self, llm: Any, system_instructions: str, tools: list[Any]) -> None:
        from langgraph.prebuilt import create_react_agent

        self.system_instructions = system_instructions
        self._executor = create_react_agent(llm, tools)
        self._history: List[Any] = []
        self._history_enabled = os.getenv("ARM_HISTORY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        self._max_history_messages = max(0, int(os.getenv("ARM_MAX_HISTORY_MESSAGES", "10")))

    def _build_messages(self, user_text: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list[Any] = [SystemMessage(content=self.system_instructions)]
        if self._history_enabled and self._history:
            messages.extend(self._history[-self._max_history_messages :])
        messages.append(HumanMessage(content=user_text))
        return messages

    def print_response(self, user_text: str, stream: bool = False) -> str:  # noqa: ARG002
        from langchain_core.messages import AIMessage, HumanMessage

        payload = {"messages": self._build_messages(user_text)}
        result = self._executor.invoke(payload) if hasattr(self._executor, "invoke") else {"messages": []}
        messages = result.get("messages", []) if isinstance(result, dict) else []
        ai_message = messages[-1] if messages else None
        ai_content = getattr(ai_message, "content", "") if ai_message is not None else ""

        if self._history_enabled:
            self._history.append(HumanMessage(content=user_text))
            self._history.append(AIMessage(content=str(ai_content)))

        return str(ai_content)

    def run(self, user_text: str) -> str:
        return self.print_response(user_text)
