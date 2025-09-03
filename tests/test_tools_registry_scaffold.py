from __future__ import annotations

from academic_research_mentor.tools import register_tool, get_tool, list_tools, ToolBase


class _EchoTool(ToolBase):
    name = "echo"

    def execute(self, inputs, context=None):  # type: ignore[override]
        return {"echo": inputs, "ctx": sorted(list((context or {}).keys()))}


def test_registry_register_and_fetch() -> None:
    t = _EchoTool()
    register_tool(t)
    assert get_tool("echo") is t
    names = set(list_tools().keys())
    assert "echo" in names
