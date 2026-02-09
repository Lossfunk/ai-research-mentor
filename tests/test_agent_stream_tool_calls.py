from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from academic_research_mentor.agent.agent import MentorAgent
from academic_research_mentor.agent.tools import ToolRegistry
from academic_research_mentor.llm.types import Message, StreamChunk, ToolCall


class _FakeStreamingClient:
    def __init__(self, *, with_tool_call: bool) -> None:
        self.with_tool_call = with_tool_call
        self.calls = 0

    async def stream_async(  # type: ignore[override]
        self,
        _messages: list[Any],
        tools: Optional[list[Any]] = None,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.with_tool_call and self.calls == 1 and tools:
            yield StreamChunk(
                tool_calls=[ToolCall(id="tc1", name="echo_tool", arguments={"text": "ok"})],
                finish_reason="tool_calls",
            )
            return
        yield StreamChunk(content="final-response")


class _FailThenFallbackClient:
    def __init__(self) -> None:
        self.stream_calls = 0
        self.chat_calls = 0

    async def stream_async(  # type: ignore[override]
        self,
        _messages: list[Any],
        tools: Optional[list[Any]] = None,  # noqa: ARG002
        include_reasoning: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    async def chat_async(  # type: ignore[override]
        self,
        _messages: list[Any],
        tools: Optional[list[Any]] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[Message, Optional[list[ToolCall]]]:
        self.chat_calls += 1
        return Message.assistant("fallback-response"), None


async def _collect_chunks(agent: MentorAgent) -> list[StreamChunk]:
    chunks = []
    async for chunk in agent.stream_async("hello", include_reasoning=False):
        chunks.append(chunk)
    return chunks


def test_stream_without_tool_probe_only_calls_stream_once() -> None:
    client = _FakeStreamingClient(with_tool_call=False)
    agent = MentorAgent(system_prompt="test", client=client, tools=ToolRegistry())
    chunks = __import__("asyncio").run(_collect_chunks(agent))
    assert client.calls == 1
    assert any((c.content or "") == "final-response" for c in chunks)


def test_stream_executes_tool_calls_then_continues_stream() -> None:
    registry = ToolRegistry()
    registry.register_function(
        name="echo_tool",
        description="echoes text",
        function=lambda text="": f"echo:{text}",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    client = _FakeStreamingClient(with_tool_call=True)
    agent = MentorAgent(system_prompt="test", client=client, tools=registry)
    chunks = __import__("asyncio").run(_collect_chunks(agent))
    statuses = [c.tool_status for c in chunks if c.tool_status]
    assert client.calls == 2
    assert "calling" in statuses
    assert "executing" in statuses
    assert "completed" in statuses
    assert any((c.content or "") == "final-response" for c in chunks)


def test_stream_falls_back_to_chat_async_on_pre_content_failure() -> None:
    client = _FailThenFallbackClient()
    agent = MentorAgent(system_prompt="test", client=client, tools=ToolRegistry())
    chunks = __import__("asyncio").run(_collect_chunks(agent))
    assert client.stream_calls == 1
    assert client.chat_calls == 1
    assert any((c.content or "") == "fallback-response" for c in chunks)
