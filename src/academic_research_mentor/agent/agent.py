"""Simple agent with tool calling support."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from academic_research_mentor.llm import LLMClient, create_client, Message, ToolCall
from academic_research_mentor.llm.types import StreamChunk, ToolResult, Role
from .tools import ToolRegistry


class MentorAgent:
    """Research mentor agent with tool calling support."""

    MAX_TOOL_ITERATIONS = 5  # Prevent infinite loops

    def __init__(
        self,
        system_prompt: str,
        client: Optional[LLMClient] = None,
        tools: Optional[ToolRegistry] = None,
        max_history: int = 20
    ):
        self.system_prompt = system_prompt
        self.client = client or create_client()
        self.tools = tools or ToolRegistry()
        self.max_history = max_history
        self._history: list[Message] = []

    def _get_messages(self, user_message: Any, context: Optional[str] = None) -> list[Message]:
        """Build message list with system prompt, history, and user message."""
        messages = [Message.system(self.system_prompt)]

        if self._history:
            history_slice = self._history[-self.max_history:]
            messages.extend(history_slice)

        if isinstance(user_message, list):
            parts = []
            if context:
                parts.append({"type": "text", "text": f"Context:\n{context}"})
            parts.extend(user_message)
            messages.append(Message(Role.USER, parts))  # type: ignore[arg-type]
        else:
            full_message = f"Context:\n{context}\n\nUser message: {user_message}" if context else user_message
            messages.append(Message.user(full_message))
        return messages

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and return tool response messages."""
        tool_messages = []
        
        for tc in tool_calls:
            result = self.tools.execute(tc.name, **tc.arguments)
            result.tool_call_id = tc.id  # Set the tool call ID
            tool_messages.append(result.to_message())
        
        return tool_messages

    def chat(self, user_message: Any, context: Optional[str] = None) -> str:
        """Send a message and get a response (with automatic tool calling)."""
        messages = self._get_messages(user_message, context)
        tool_definitions = self.tools.get_definitions() if len(self.tools) > 0 else None

        # Tool calling loop
        for _ in range(self.MAX_TOOL_ITERATIONS):
            response, tool_calls = self.client.chat(messages, tools=tool_definitions)
            
            if not tool_calls:
                # No tool calls - we have the final response
                self._history.append(Message.user(user_message))
                self._history.append(response)
                return response.content
            
            # Execute tool calls
            messages.append(response)  # Add assistant message with tool calls
            tool_messages = self._execute_tool_calls(tool_calls)
            messages.extend(tool_messages)

        # Max iterations reached
        return "I apologize, but I encountered an issue processing your request. Please try again."

    async def chat_async(self, user_message: Any, context: Optional[str] = None) -> str:
        """Async version of chat."""
        messages = self._get_messages(user_message, context)
        tool_definitions = self.tools.get_definitions() if len(self.tools) > 0 else None

        for _ in range(self.MAX_TOOL_ITERATIONS):
            response, tool_calls = await self.client.chat_async(messages, tools=tool_definitions)
            
            if not tool_calls:
                self._history.append(Message.user(user_message))
                self._history.append(response)
                return response.content
            
            messages.append(response)
            tool_messages = self._execute_tool_calls(tool_calls)
            messages.extend(tool_messages)

        return "I apologize, but I encountered an issue processing your request. Please try again."

    async def stream_async(
        self,
        user_message: Any,
        context: Optional[str] = None,
        include_reasoning: bool = True
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response asynchronously with tool support.
        
        Handles streamed tool-calls inline to avoid an extra non-stream probe call.
        """
        messages = self._get_messages(user_message, context)
        tool_definitions = self.tools.get_definitions() if len(self.tools) > 0 else None

        for _ in range(self.MAX_TOOL_ITERATIONS):
            full_content = ""
            streamed_tool_calls: list[ToolCall] = []
            try:
                async for chunk in self.client.stream_async(
                    messages,
                    tools=tool_definitions,
                    include_reasoning=include_reasoning,
                ):
                    if chunk.tool_calls:
                        streamed_tool_calls.extend(chunk.tool_calls)
                        continue
                    if chunk.content:
                        full_content += chunk.content
                    yield chunk
            except Exception:
                # Fallback to non-stream completion if streaming fails before emitting text.
                if full_content.strip() or streamed_tool_calls:
                    raise
                response, fallback_tool_calls = await self.client.chat_async(messages, tools=tool_definitions)
                if fallback_tool_calls:
                    streamed_tool_calls = fallback_tool_calls
                else:
                    fallback_text = response.content or ""
                    if fallback_text:
                        full_content += fallback_text
                        yield StreamChunk(content=fallback_text)
                    self._history.append(Message.user(user_message))
                    self._history.append(Message.assistant(full_content))
                    return

            if streamed_tool_calls:
                messages.append(Message.assistant("", streamed_tool_calls))

                for tc in streamed_tool_calls:
                    yield StreamChunk(
                        tool_status="calling",
                        tool_name=tc.name,
                        content=f"Calling tool: {tc.name}",
                    )
                for tc in streamed_tool_calls:
                    yield StreamChunk(tool_status="executing", tool_name=tc.name)
                    result = self.tools.execute(tc.name, **tc.arguments)
                    result.tool_call_id = tc.id
                    messages.append(result.to_message())
                    yield StreamChunk(
                        tool_status="completed",
                        tool_name=tc.name,
                        tool_result=result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    )
                continue

            self._history.append(Message.user(user_message))
            self._history.append(Message.assistant(full_content))
            return

        yield StreamChunk(content="I apologize, but I encountered an issue processing your request. Please try again.")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self._history.copy()


def create_mentor_agent(
    system_prompt: Optional[str] = None,
    provider: str = "openrouter"
) -> MentorAgent:
    """Create a mentor agent with default configuration.
    
    Args:
        system_prompt: Custom system prompt (loads from prompt.md if not provided)
        provider: LLM provider ("openrouter" or "openai")
    """
    # Load system prompt if not provided
    if not system_prompt:
        from academic_research_mentor.prompts_loader import load_instructions_from_prompt_md
        instructions, _ = load_instructions_from_prompt_md("mentor", ascii_normalize=False)
        system_prompt = instructions or "You are a helpful research mentor."
    
    client = create_client(provider=provider)
    return MentorAgent(system_prompt=system_prompt, client=client)
