from __future__ import annotations

import os
from typing import Any, Optional, Tuple
import re

from .rich_formatter import (
    print_formatted_response, 
    print_streaming_chunk, 
    print_error,
    start_streaming_response,
    end_streaming_response
)


class _LangChainAgentWrapper:
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


class _LangChainReActAgentWrapper:
    """Wrapper around a ReAct agent (LangGraph prebuilt) using our tools.

    Keeps the same surface as _LangChainAgentWrapper for CLI compatibility.
    Streaming is step-wise (not token-wise) in this minimal implementation.
    """

    def __init__(self, llm: Any, system_instructions: str, tools: list[Any]) -> None:
        from langchain_core.messages import SystemMessage  # type: ignore
        from langgraph.prebuilt import create_react_agent  # type: ignore

        self._llm = llm
        self._system_instructions = system_instructions
        self._SystemMessage = SystemMessage
        self._agent_executor = create_react_agent(llm, tools)

    def _build_messages(self, user_text: str) -> list[Any]:
        from langchain_core.messages import HumanMessage  # type: ignore

        return [
            self._SystemMessage(content=self._system_instructions),
            HumanMessage(content=user_text),
        ]

    def print_response(self, user_text: str, stream: bool = True) -> None:  # noqa: ARG002
        # For simplicity, invoke once and print final model response (tool steps are internal)
        try:
            result = self._agent_executor.invoke({"messages": self._build_messages(user_text)})
            messages = result.get("messages", []) if isinstance(result, dict) else []
            content = ""
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
            print_formatted_response(content, "Mentor (ReAct Agent)")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")

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
        return _Reply(content)


class _LangChainSpecialistRouterWrapper:
    """Specialist router using LangGraph StateGraph with simple triggers.

    Routes to venue, openreview, math, methodology, or default chat.
    """

    def __init__(self, llm: Any, system_instructions: str) -> None:
        from langgraph.graph import StateGraph, END  # type: ignore
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
        from .mentor_tools import (
            arxiv_search,
            openreview_fetch,
            venue_guidelines_get,
            math_ground,
            methodology_validate,
        )

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
            if re.search(r"\bopenreview\b|\bopen\s*review\b", text, flags=re.IGNORECASE):
                return "openreview"
            if re.search(r"\bguidelines\b|\bICLR\b|\bNeurIPS\b|\bACL\b", text, flags=re.IGNORECASE):
                return "venue"
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

        def node_venue(state: dict) -> dict:
            last = state["messages"][-1]
            txt = getattr(last, "content", None) or str(last)
            m = re.search(r"([A-Za-z]{2,})\s*(\d{4})?", txt)
            venue = m.group(1) if m else txt
            year = int(m.group(2)) if (m and m.group(2)) else None
            res = venue_guidelines_get(venue=venue, year=year)
            g = (res or {}).get("guidelines", {})
            urls = g.get("urls", {}) if isinstance(g, dict) else {}
            parts = [f"Venue: {venue.upper()} {year or ''}".strip()]
            if urls.get("guide"):
                parts.append(f"Guide: {urls['guide']}")
            if urls.get("template"):
                parts.append(f"Template: {urls['template']}")
            content = "\n".join(parts) if len(parts) > 1 else "No known URLs"
            new_msgs = state["messages"] + [self._AIMessage(content=content)]
            return {"messages": new_msgs}

        def node_openreview(state: dict) -> dict:
            last = state["messages"][-1]
            q = getattr(last, "content", None) or str(last)
            res = openreview_fetch(query=q, limit=5)
            threads = (res or {}).get("threads", [])
            if not threads:
                content = (res or {}).get("note", "No results")
            else:
                lines = []
                for t in threads[:5]:
                    title = t.get("paper_title")
                    venue = t.get("venue")
                    year = t.get("year")
                    url = (t.get("urls") or {}).get("paper")
                    suffix = f" ({venue} {year})" if (venue or year) else ""
                    lines.append(f"- {title}{suffix} -> {url}")
                content = "\n".join(lines)
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
        builder.add_node("venue", node_venue)
        builder.add_node("openreview", node_openreview)
        builder.add_node("math", node_math)
        builder.add_node("methodology", node_method)
        builder.add_node("arxiv", node_arxiv)
        builder.add_node("end", lambda s: s)

        def route_selector(state: dict) -> str:
            return classify(state)

        builder.set_entry_point("chat")
        # After chat classification of the incoming message
        builder.add_conditional_edges(
            "chat",
            route_selector,
            {"chat": "chat", "venue": "venue", "openreview": "openreview", "math": "math", "methodology": "methodology", "arxiv": "arxiv"},
        )
        # All terminal nodes go to END
        for node in ["venue", "openreview", "math", "methodology", "arxiv"]:
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

def _import_langchain_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Lazily import LangChain chat model classes for major providers."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatOpenAI = None  # type: ignore
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatGoogleGenerativeAI = None  # type: ignore
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except Exception:  # pragma: no cover
        ChatAnthropic = None  # type: ignore
    try:
        from langchain_mistralai import ChatMistralAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatMistralAI = None  # type: ignore
    return ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic, ChatMistralAI


def _resolve_model() -> Tuple[Optional[Any], Optional[str]]:
    ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic, ChatMistralAI = _import_langchain_models()
    try:
        # Prefer OpenRouter when available (OpenAI-compatible API)
        if os.environ.get("OPENROUTER_API_KEY") and ChatOpenAI is not None:
            model_id = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            llm = ChatOpenAI(
                model=model_id,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=base_url,
                temperature=0,
            )
            return llm, None

        # OpenAI
        if os.environ.get("OPENAI_API_KEY") and ChatOpenAI is not None:
            model_id = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            # Respect optional OPENAI_BASE_URL if provided
            base_url = os.environ.get("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {"model": model_id, "temperature": 0}
            if base_url:
                kwargs["base_url"] = base_url
            llm = ChatOpenAI(**kwargs)
            return llm, None

        # Google Gemini
        if os.environ.get("GOOGLE_API_KEY") and ChatGoogleGenerativeAI is not None:
            model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-latest")
            llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0)
            return llm, None

        # Anthropic
        if os.environ.get("ANTHROPIC_API_KEY") and ChatAnthropic is not None:
            model_id = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            llm = ChatAnthropic(model=model_id, api_key=os.environ.get("ANTHROPIC_API_KEY"), temperature=0)
            return llm, None

        # Mistral
        if os.environ.get("MISTRAL_API_KEY") and ChatMistralAI is not None:
            model_id = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
            llm = ChatMistralAI(model=model_id, api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0)
            return llm, None

        return None, "No supported model API key found (set OPENROUTER_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, or MISTRAL_API_KEY)"
    except Exception as exc:  # pragma: no cover
        return None, f"Model init failed: {exc}"


def build_agent(instructions: str) -> Tuple[Optional[Any], Optional[str]]:
    llm, reason = _resolve_model()
    if llm is None:
        return None, reason

    # Toggle ReAct agent via env var; default to simple chat wrapper
    mode = os.environ.get("LC_AGENT_MODE", "chat").strip().lower()
    if mode in {"react", "tools"}:
        tools = get_langchain_tools()
        agent = _LangChainReActAgentWrapper(llm=llm, system_instructions=instructions, tools=tools)
    elif mode in {"router", "route"}:
        agent = _LangChainSpecialistRouterWrapper(llm=llm, system_instructions=instructions)
    else:
        # We keep manual tool routing in the CLI; here we just build a chat agent
        agent = _LangChainAgentWrapper(llm=llm, system_instructions=instructions)
    return agent, None


def get_langchain_tools() -> list[Any]:
    """Return LangChain Tool objects wrapping existing mentor tools.

    Non-breaking: callers can opt-in to LangChain agent-based tool use.
    """
    try:
        from langchain.tools import Tool  # type: ignore
    except Exception:
        return []

    try:
        from .mentor_tools import (
            arxiv_search,
            openreview_fetch,
            venue_guidelines_get,
            math_ground,
            methodology_validate,
        )
    except Exception:
        return []

    def _arxiv_tool_fn(q: str) -> str:
        res = arxiv_search(query=q, from_year=None, limit=5)
        papers = (res or {}).get("papers", [])
        if not papers:
            return (res or {}).get("note", "No results")
        lines = []
        for p in papers[:5]:
            title = p.get("title")
            year = p.get("year")
            url = p.get("url")
            lines.append(f"- {title} ({year}) -> {url}")
        return "\n".join(lines)

    def _openreview_tool_fn(q: str) -> str:
        res = openreview_fetch(query=q, limit=5)
        threads = (res or {}).get("threads", [])
        if not threads:
            return (res or {}).get("note", "No results")
        lines = []
        for t in threads[:5]:
            title = t.get("paper_title")
            venue = t.get("venue")
            year = t.get("year")
            url = (t.get("urls") or {}).get("paper")
            suffix = f" ({venue} {year})" if (venue or year) else ""
            lines.append(f"- {title}{suffix} -> {url}")
        return "\n".join(lines)

    def _venue_tool_fn(text: str) -> str:
        m = re.search(r"([A-Za-z]{2,})\s*(\d{4})?", text)
        venue = m.group(1) if m else text
        year = int(m.group(2)) if (m and m.group(2)) else None
        res = venue_guidelines_get(venue=venue, year=year)
        g = (res or {}).get("guidelines", {})
        urls = g.get("urls", {}) if isinstance(g, dict) else {}
        parts = [f"Venue: {venue.upper()} {year or ''}".strip()]
        if urls.get("guide"):
            parts.append(f"Guide: {urls['guide']}")
        if urls.get("template"):
            parts.append(f"Template: {urls['template']}")
        return "\n".join(parts) if len(parts) > 1 else "No known URLs"

    def _math_tool_fn(text: str) -> str:
        res = math_ground(text_or_math=text, options={})
        findings = (res or {}).get("findings", {})
        keys = ["assumptions", "symbol_glossary", "dimensional_issues", "proof_skeleton"]
        lines = []
        for k in keys:
            vals = findings.get(k) or []
            if vals:
                lines.append(f"- {k}: {', '.join(str(x) for x in vals[:3])}")
        return "\n".join(lines) or "No findings"

    def _method_tool_fn(text: str) -> str:
        res = methodology_validate(plan=text, checklist=[])
        report = (res or {}).get("report", {})
        keys = ["risks", "missing_controls", "ablation_suggestions", "reproducibility_gaps"]
        lines = []
        for k in keys:
            vals = report.get(k) or []
            if vals:
                lines.append(f"- {k}: {', '.join(str(x) for x in vals)}")
        return "\n".join(lines) or "No issues detected"

    tools: list[Any] = [
        Tool(
            name="arxiv_search",
            func=_arxiv_tool_fn,
            description=(
                "Search arXiv for recent academic papers on any research topic. "
                "Use this whenever the user asks about research, papers, literature, "
                "related work, or wants to understand what's been done in a field. "
                "Input: research topic or keywords (e.g. 'transformer models', 'deep reinforcement learning'). "
                "Returns: list of relevant papers with titles, years, and URLs."
            ),
        ),
        Tool(
            name="openreview_fetch",
            func=_openreview_tool_fn,
            description=(
                "Search OpenReview for academic papers and reviews from top-tier conferences. "
                "Use this to find papers from venues like NeurIPS, ICLR, ICML, etc. "
                "Input: research topic, keywords, or venue name. "
                "Returns: papers with reviews, venues, and forum links."
            ),
        ),
        Tool(
            name="venue_guidelines_get",
            func=_venue_tool_fn,
            description=(
                "Get likely author-guideline URLs for a venue/year. Input: 'ICLR 2025' or 'NeurIPS'."
            ),
        ),
        Tool(
            name="math_ground",
            func=_math_tool_fn,
            description=(
                "Heuristic math grounding. Input: TeX/plain text. Returns brief findings."
            ),
        ),
        Tool(
            name="methodology_validate",
            func=_method_tool_fn,
            description=(
                "Validate an experiment plan for risks/controls/ablations/reproducibility gaps."
            ),
        ),
    ]
    return tools
