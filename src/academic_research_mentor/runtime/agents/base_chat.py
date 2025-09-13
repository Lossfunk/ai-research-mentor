from __future__ import annotations

from typing import Any

from ...rich_formatter import (
    print_formatted_response,
    print_streaming_chunk,
    print_error,
    start_streaming_response,
    end_streaming_response,
)


class LangChainAgentWrapper:
    """Minimal wrapper to mimic the prior Agent surface used by our CLI.

    - print_response(text, stream=True): prints a response
    - run(text): returns an object with .content
    """

    def __init__(self, llm: Any, system_instructions: str) -> None:
        self._llm = llm
        self._system_instructions = system_instructions
        try:
            # Prefer langchain-core message types when available
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # type: ignore
        except Exception:  # pragma: no cover
            HumanMessage = None  # type: ignore
            SystemMessage = None  # type: ignore
            AIMessage = None  # type: ignore
        self._HumanMessage = HumanMessage  # type: ignore[attr-defined]
        self._SystemMessage = SystemMessage  # type: ignore[attr-defined]
        self._AIMessage = AIMessage  # type: ignore[attr-defined]
        # Lightweight in-memory conversation buffer (bounded)
        self._history = []
        self._max_history_messages = 10

    def _build_messages(self, user_text: str) -> Any:
        if self._HumanMessage and self._SystemMessage:
            messages: list[Any] = [self._SystemMessage(content=self._system_instructions)]
            # Append bounded history
            if self._history:
                messages.extend(self._history[-self._max_history_messages :])
            messages.append(self._HumanMessage(content=user_text))
            return messages
        # Fallback: raw string prompt composition
        return f"{self._system_instructions}\n\n{user_text}"

    def print_response(self, user_text: str, stream: bool = True) -> None:  # noqa: ARG002
        try:
            if stream and hasattr(self._llm, "stream"):
                accumulated: list[str] = []
                start_streaming_response("Mentor")
                try:
                    from langchain_core.messages import AIMessageChunk  # type: ignore
                except Exception:  # pragma: no cover
                    AIMessageChunk = None  # type: ignore
                for chunk in self._llm.stream(self._build_messages(user_text)):
                    # Only stream model token chunks; ignore tool calls/metadata
                    if AIMessageChunk is not None and isinstance(chunk, AIMessageChunk):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            accumulated.append(piece)
                            print_streaming_chunk(piece)
                        continue
                    # Some providers yield dict-like or other chunk types; attempt safe extract
                    piece = getattr(chunk, "content", None)
                    if isinstance(piece, str) and piece:
                        accumulated.append(piece)
                        print_streaming_chunk(piece)
                end_streaming_response()
                content = "".join(accumulated)
            else:
                result = self._llm.invoke(self._build_messages(user_text))
                content = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
                print_formatted_response(content, "Mentor")
            # Update history buffer when message classes available
            if self._HumanMessage and self._AIMessage:
                self._history.append(self._HumanMessage(content=user_text))
                self._history.append(self._AIMessage(content=content))
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")

    def run(self, user_text: str) -> Any:
        class _Reply:
            def __init__(self, text: str) -> None:
                self.content = text
                self.text = text

        result = self._llm.invoke(self._build_messages(user_text))
        content = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
        if self._HumanMessage and self._AIMessage:
            self._history.append(self._HumanMessage(content=user_text))
            self._history.append(self._AIMessage(content=content))
        return _Reply(content)
