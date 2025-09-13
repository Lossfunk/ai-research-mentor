from __future__ import annotations

import re
from typing import Any

from ...rich_formatter import print_formatted_response, print_error
from ...mentor_tools import (
    arxiv_search,
    math_ground,
    methodology_validate,
)


class LangChainSpecialistRouterWrapper:
    """Specialist router using LangGraph StateGraph with simple triggers.

    Routes to venue, math, methodology, or default chat.
    """

    def __init__(self, llm: Any, system_instructions: str) -> None:
        from langgraph.graph import StateGraph, END  # type: ignore
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore

        self._llm = llm
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
        self._AIMessage = AIMessage
        self._system_instructions = system_instructions

        def classify(state: dict) -> str:
            msgs = state.get("messages", [])
            text = ""
            if msgs:
                last = msgs[-1]
                text = getattr(last, "content", None) or getattr(last, "text", None) or str(last)

            if re.search(r"\$|\\\(|\\\[|\\begin\{equation\}|\\int|\\sum|\\frac|\bnorm\b|\bO\(", text, flags=re.IGNORECASE):
                return "math"
            if re.search(r"\bmethodology\b|\bexperiment\b|\bevaluation\s+plan\b|^\s*validate\s*:", text, flags=re.IGNORECASE):
                return "methodology"
            if re.search(r"\barxiv\b|\bpapers\b|\bfind\b|\bliterature\b|\brelated\s+work\b|\bsurvey\b|\bresearch\s+on\b|\bresearch\s+about\b|\bresearch\s+in\b", text, flags=re.IGNORECASE):
                return "arxiv"
            return "chat"

        def node_chat(state: dict) -> dict:
            result = self._llm.invoke(state["messages"])
            content = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
            new_msgs = state["messages"] + [self._AIMessage(content=content)]
            return {"messages": new_msgs}

        def node_math(state: dict) -> dict:
            last = state["messages"][-1]
            txt = getattr(last, "content", None) or str(last)
            res = math_ground(text_or_math=txt, options={})
            findings = (res or {}).get("findings", {})
            keys = ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]
            lines = []
            for k in keys:
                vals = findings.get(k) or []
                if vals:
                    lines.append(f"- {k}: {', '.join(str(x) for x in vals[:3])}")
            content = "\n".join(lines) or "No findings"
            new_msgs = state["messages"] + [self._AIMessage(content=content)]
            return {"messages": new_msgs}

        def node_method(state: dict) -> dict:
            last = state["messages"][-1]
            txt = getattr(last, "content", None) or str(last)
            res = methodology_validate(plan=txt, checklist=[])
            report = (res or {}).get("report", {})
            keys = ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]
            lines = []
            for k in keys:
                vals = report.get(k) or []
                if vals:
                    lines.append(f"- {k}: {', '.join(str(x) for x in vals)}")
            content = "\n".join(lines) or "No issues detected"
            new_msgs = state["messages"] + [self._AIMessage(content=content)]
            return {"messages": new_msgs}

        def node_arxiv(state: dict) -> dict:
            last = state["messages"][-1]
            q = getattr(last, "content", None) or str(last)
            res = arxiv_search(query=q, from_year=None, limit=5)
            papers = (res or {}).get("papers", [])
            if not papers:
                content = (res or {}).get("note", "No results")
            else:
                lines = []
                for p in papers[:5]:
                    title = p.get("title")
                    year = p.get("year")
                    url = p.get("url")
                    lines.append(f"- {title} ({year}) -> {url}")
                content = "\n".join(lines)
            new_msgs = state["messages"] + [self._AIMessage(content=content)]
            return {"messages": new_msgs}

        builder = StateGraph(state_schema=dict)  # type: ignore[arg-type]
        builder.add_node("chat", node_chat)
        builder.add_node("math", node_math)
        builder.add_node("methodology", node_method)
        builder.add_node("arxiv", node_arxiv)

        def route_selector(state: dict) -> str:
            return classify(state)

        builder.set_entry_point("chat")
        # After chat classification of the incoming message
        builder.add_conditional_edges(
            "chat",
            route_selector,
            {"chat": "chat", "math": "math", "methodology": "methodology", "arxiv": "arxiv"},
        )
        # All terminal nodes go to END
        for node in ["math", "methodology", "arxiv"]:
            builder.add_edge(node, END)
        self._graph = builder.compile()

    def _init_state(self, user_text: str) -> dict:
        return {
            "messages": [
                self._SystemMessage(content=self._system_instructions),
                self._HumanMessage(content=user_text),
            ]
        }

    def print_response(self, user_text: str, stream: bool = True) -> None:  # noqa: ARG002
        try:
            result = self._graph.invoke(self._init_state(user_text))
            messages = result.get("messages", []) if isinstance(result, dict) else []
            content = ""
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
            print_formatted_response(content, "Mentor (Specialist Router)")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")

    def run(self, user_text: str) -> Any:
        class _Reply:
            def __init__(self, text: str) -> None:
                self.content = text
                self.text = text

        result = self._graph.invoke(self._init_state(user_text))
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = ""
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
        return _Reply(content)
