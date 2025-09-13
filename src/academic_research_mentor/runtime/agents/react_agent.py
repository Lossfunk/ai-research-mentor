from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ...rich_formatter import print_formatted_response, print_error, print_agent_reasoning


class LangChainReActAgentWrapper:
    """Wrapper around a ReAct agent (LangGraph prebuilt) using our tools.

    Keeps the same surface as LangChainAgentWrapper for CLI compatibility.
    Streaming is step-wise (not token-wise) in this minimal implementation.
    """

    def __init__(self, llm: Any, system_instructions: str, tools: list[Any]) -> None:
        from langchain_core.messages import SystemMessage  # type: ignore
        from langgraph.prebuilt import create_react_agent  # type: ignore

        self._llm = llm
        self._system_instructions = system_instructions
        self._SystemMessage = SystemMessage
        self._agent_executor = create_react_agent(llm, tools)
        self._chat_logger = None
        self._current_user_input = None
        # Delimiters for hiding internal/tool reasoning from the response display
        self._internal_begin = "<<<AGENT_INTERNAL_BEGIN>>>"
        self._internal_end = "<<<AGENT_INTERNAL_END>>>"

    def set_chat_logger(self, chat_logger: Any) -> None:
        """Set the chat logger for recording conversations."""
        self._chat_logger = chat_logger

    def _build_messages(self, user_text: str) -> list[Any]:
        from langchain_core.messages import HumanMessage  # type: ignore

        return [
            self._SystemMessage(content=self._system_instructions),
            HumanMessage(content=user_text),
        ]

    def print_response(self, user_text: str, stream: bool = True) -> None:  # noqa: ARG002
        # Render in strict order: (1) Agent's reasoning (via tool panels), then (2) Agent's response
        try:
            self._current_user_input = user_text
            content = ""
            # Invoke once synchronously so tools can print their reasoning panels first
            result = self._agent_executor.invoke({"messages": self._build_messages(user_text)})
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
            tool_calls = self._extract_tool_calls(result)

            # Log the conversation turn (after content determined)
            if self._chat_logger:
                self._chat_logger.add_turn(user_text, tool_calls, self._clean_for_display(content, user_text))

            # Always print the final, cleaned response once
            print_formatted_response(self._clean_for_display(content, user_text), "Agent's response")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")

    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from the agent execution result."""
        tool_calls = []

        if not isinstance(result, dict):
            return tool_calls

        # Look for tool calls in the messages
        messages = result.get("messages", [])
        for msg in messages:
            # Check for tool messages or AI messages with tool calls
            if hasattr(msg, 'tool_calls'):
                # LangChain tool calls format
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "tool_name": tool_call.get('name', 'unknown'),
                        "score": 3.0  # Default score as in examples
                    })
            elif hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                # Alternative format for tool calls
                for tool_call in msg.additional_kwargs['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                    tool_calls.append({
                        "tool_name": tool_name,
                        "score": 3.0
                    })

        return tool_calls

    def run(self, user_text: str) -> Any:
        class _Reply:
            def __init__(self, text: str) -> None:
                self.content = text
                self.text = text

        result = self._agent_executor.invoke({"messages": self._build_messages(user_text)})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = ""
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
        return _Reply(self._clean_for_display(content, user_text))

    def _clean_for_display(self, content: str, user_text: Optional[str]) -> str:
        """Strip internal reasoning blocks and remove user-echo prefixes for display.

        This keeps the TUI "Agent's response" focused on the final answer.
        """
        try:
            text = str(content or "")
            if not text:
                return text
            # Remove internal blocks
            pattern = re.compile(re.escape(self._internal_begin) + r"[\s\S]*?" + re.escape(self._internal_end))
            text = re.sub(pattern, "", text)
            # Remove user echo at the beginning (case-insensitive, whitespace tolerant)
            if user_text:
                ut = str(user_text).strip()
                if ut:
                    # Simple prefix strip if present
                    if text.lstrip().lower().startswith(ut.lower()):
                        # Preserve original leading whitespace before user echo
                        leading = len(text) - len(text.lstrip())
                        text = text[:leading] + text.lstrip()[len(ut):]
            # Collapse excessive blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text
        except Exception:
            return str(content or "")
