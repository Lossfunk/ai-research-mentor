from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Optional

from ...rich_formatter import print_formatted_response, print_error, print_agent_reasoning
from ...core.stage_detector import detect_stage
from ...session_logging import SessionLogManager


class LangChainReActAgentWrapper:
    """Wrapper around a ReAct agent (LangGraph prebuilt) using our tools.

    Keeps the same surface as LangChainAgentWrapper for CLI compatibility.
    Streaming is step-wise (not token-wise) in this minimal implementation.
    """

    def __init__(self, llm: Any, system_instructions: str, tools: list[Any]) -> None:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
        from langgraph.prebuilt import create_react_agent  # type: ignore

        self._llm = llm
        self._system_instructions = system_instructions
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
        self._AIMessage = AIMessage
        self._agent_executor = create_react_agent(llm, tools)
        self._chat_logger = None
        self._current_user_input = None
        self._session_logger: Optional[SessionLogManager] = None
        # Delimiters for hiding internal/tool reasoning from the response display
        self._internal_begin = "<<<AGENT_INTERNAL_BEGIN>>>"
        self._internal_end = "<<<AGENT_INTERNAL_END>>>"
        try:
            self._recursion_limit = int(os.environ.get("ARM_GRAPH_RECURSION_LIMIT", "40"))
        except Exception:
            self._recursion_limit = 40
        # Lightweight bounded conversation memory (Human/AI pairs only; System provided per turn)
        self._history: list[Any] = []
        self._history_enabled: bool = os.environ.get("ARM_HISTORY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        try:
            self._max_history_messages: int = int(os.environ.get("ARM_MAX_HISTORY_MESSAGES", "12"))
        except Exception:
            self._max_history_messages = 12

    def _invoke_with_recursion_limit(self, payload: Dict[str, Any]) -> Any:
        try:
            return self._agent_executor.invoke(payload, config={"recursion_limit": self._recursion_limit})
        except TypeError:
            return self._agent_executor.invoke(payload)

    def set_chat_logger(self, chat_logger: Any) -> None:
        """Set the chat logger for recording conversations."""
        self._chat_logger = chat_logger

    def set_session_logger(self, session_logger: SessionLogManager) -> None:
        self._session_logger = session_logger

    def _build_messages(self, user_text: str) -> list[Any]:
        from langchain_core.messages import HumanMessage  # type: ignore

        messages: list[Any] = [self._SystemMessage(content=self._system_instructions)]
        if self._history_enabled and self._history:
            # Append the tail of history within the cap
            messages.extend(self._history[-self._max_history_messages :])
        messages.append(HumanMessage(content=user_text))
        return messages

    def print_response(self, user_text: str, stream: bool = True) -> None:  # noqa: ARG002
        # Render in strict order: (1) Agent's reasoning (via tool panels), then (2) Agent's response
        try:
            self._current_user_input = user_text
            content = ""
            if self._session_logger:
                self._session_logger.log_event("agent_invoked", {"history_len": len(self._history), "input_preview": user_text[:200]})
            # Invoke once synchronously so tools can print their reasoning panels first
            result = self._invoke_with_recursion_limit({"messages": self._build_messages(user_text)})
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
            tool_calls = self._extract_tool_calls(result)
            if self._session_logger:
                self._session_logger.record_tool_calls(tool_calls)

            # Log the conversation turn (after content determined) with stage detection
            if self._chat_logger:
                try:
                    stage = detect_stage(self._current_user_input or user_text)
                except Exception:
                    stage = None
                self._chat_logger.add_turn(user_text, tool_calls, self._clean_for_display(content, user_text), stage=stage)

            # Update bounded history (Human -> AI)
            if self._history_enabled and self._HumanMessage and self._AIMessage:
                try:
                    self._history.append(self._HumanMessage(content=user_text))
                    self._history.append(self._AIMessage(content=self._clean_for_display(content, user_text)))
                    if len(self._history) > self._max_history_messages:
                        self._history = self._history[-self._max_history_messages :]
                except Exception:
                    pass

            # Always print the final, cleaned response once
            cleaned = self._clean_for_display(content, user_text)
            if self._session_logger:
                self._session_logger.log_event("agent_response", {"content": cleaned, "raw": content})
            print_formatted_response(cleaned, "Agent's response")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")
            if self._session_logger:
                self._session_logger.log_event("agent_failure", {"error": str(exc)})

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

        result = self._invoke_with_recursion_limit({"messages": self._build_messages(user_text)})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = ""
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
        cleaned = self._clean_for_display(content, user_text)
        if self._session_logger:
            self._session_logger.log_event("agent_response", {"content": cleaned, "raw": content, "mode": "run"})
        if self._history_enabled and self._HumanMessage and self._AIMessage:
            try:
                self._history.append(self._HumanMessage(content=user_text))
                self._history.append(self._AIMessage(content=cleaned))
                if len(self._history) > self._max_history_messages:
                    self._history = self._history[-self._max_history_messages :]
            except Exception:
                pass
        return _Reply(cleaned)

    def reset_history(self) -> None:
        """Clear the in-memory conversation history for this agent instance."""
        try:
            self._history = []
        except Exception:
            self._history = []

    def preload_history_from_chatlog(self, turns: list[dict]) -> int:
        """Load Human/AI message pairs from a saved chat log into memory.

        Returns the number of pairs loaded.
        """
        try:
            if not (self._HumanMessage and self._AIMessage):
                return 0
            self._history = []
            loaded = 0
            for turn in turns:
                user = (turn or {}).get("user_prompt")
                ai = (turn or {}).get("ai_response")
                if user and ai:
                    self._history.append(self._HumanMessage(content=str(user)))
                    self._history.append(self._AIMessage(content=str(ai)))
                    loaded += 1
            if self._history_enabled and len(self._history) > self._max_history_messages:
                self._history = self._history[-self._max_history_messages :]
            return loaded
        except Exception:
            return 0

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
