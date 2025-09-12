from __future__ import annotations

import os
import re
from typing import Any, Optional, Tuple, List, Dict

from .rich_formatter import (
    print_formatted_response, 
    print_streaming_chunk, 
    print_error,
    start_streaming_response,
    end_streaming_response,
    print_info,
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
        self._chat_logger = None
        self._current_user_input = None
        
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
        # Stream step-wise when available to avoid sudden full output
        try:
            self._current_user_input = user_text
            content = ""
            if stream and hasattr(self._agent_executor, "stream"):
                start_streaming_response("Mentor (ReAct Agent)")
                try:
                    # Stream agent values and print incremental deltas of the last AI message
                    for step in self._agent_executor.stream({"messages": self._build_messages(user_text)}, stream_mode="values"):  # type: ignore[arg-type]
                        try:
                            msgs = step.get("messages", []) if isinstance(step, dict) else []
                            if msgs:
                                last_msg = msgs[-1]
                                latest = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or ""
                                if isinstance(latest, str) and len(latest) > len(content):
                                    delta = latest[len(content):]
                                    if delta:
                                        print_streaming_chunk(delta)
                                    content = latest
                        except Exception:
                            continue
                finally:
                    end_streaming_response()
                # Best-effort tool call extraction is not easily available during streaming; do once at end
                result = {"messages": [{"content": content}]}
                tool_calls = []
            else:
                result = self._agent_executor.invoke({"messages": self._build_messages(user_text)})
                messages = result.get("messages", []) if isinstance(result, dict) else []
                if messages:
                    last_msg = messages[-1]
                    content = getattr(last_msg, "content", None) or getattr(last_msg, "text", None) or str(last_msg)
                tool_calls = self._extract_tool_calls(result)

            # Log the conversation turn (after content determined)
            if self._chat_logger:
                self._chat_logger.add_turn(user_text, tool_calls, content)

            if not stream:
                print_formatted_response(content, "Mentor (ReAct Agent)")
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
        return _Reply(content)


class _LangChainSpecialistRouterWrapper:
    """Specialist router using LangGraph StateGraph with simple triggers.

    Routes to venue, math, methodology, or default chat.
    """

    def __init__(self, llm: Any, system_instructions: str) -> None:
        from langgraph.graph import StateGraph, END  # type: ignore
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
        from .mentor_tools import (
            arxiv_search,
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
        builder.add_node("end", lambda s: s)

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

    # Toggle ReAct agent via env var; default to react for fresh clones
    mode = os.environ.get("LC_AGENT_MODE", "react").strip().lower()
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
            math_ground,
            methodology_validate,
        )
    except Exception:
        return []

    # --- Helpers for transparency in ReAct tool wrappers ---
    def _print_summary_and_sources(result: dict | None) -> None:
        try:
            if not isinstance(result, dict):
                return
            summary_lines: list[str] = []
            sources: list[str] = []
            papers = result.get("papers")
            results = result.get("results")
            threads = result.get("threads")
            retrieved = result.get("retrieved_guidelines")
            if isinstance(papers, list) and papers:
                for p in papers[:3]:
                    title = p.get("title") or p.get("paper_title") or "paper"
                    url = p.get("url") or (p.get("urls", {}) or {}).get("paper")
                    if url:
                        sources.append(url)
                    summary_lines.append(f"- {title}")
            elif isinstance(results, list) and results:
                for r in results[:3]:
                    title = r.get("title") or r.get("paper_title") or "result"
                    url = r.get("url") or (r.get("urls", {}) or {}).get("paper")
                    if url:
                        sources.append(url)
                    summary_lines.append(f"- {title}")
            elif isinstance(threads, list) and threads:
                for t in threads[:3]:
                    title = t.get("paper_title") or "thread"
                    url = (t.get("urls", {}) or {}).get("paper")
                    if url:
                        sources.append(url)
                    summary_lines.append(f"- {title}")
            elif isinstance(retrieved, list) and retrieved:
                for g in retrieved[:3]:
                    src = g.get("source_domain") or g.get("search_query") or "guideline"
                    sources.append(src)
                    summary_lines.append(f"- {src}")
            if summary_lines:
                print_info("Found:\n" + "\n".join(summary_lines[:3]))
            if sources:
                print_info("Sources: " + ", ".join(sources[:5]))
        except Exception:
            # Transparency is best-effort; never fail the interaction
            pass

    def _registry_tool_call(tool_name: str, payload: dict) -> dict:
        try:
            from .tools import auto_discover as _auto, get_tool as _get
            _auto()
            tool = _get(tool_name)
            if tool is None:
                print_info(f"Using tool: {tool_name} (unavailable)")
                return {"note": f"tool {tool_name} unavailable"}
            print_info(f"Using tool: {tool_name}")
            result = tool.execute(payload, {"goal": payload.get("query", "")})
            _print_summary_and_sources(result if isinstance(result, dict) else {})
            return result if isinstance(result, dict) else {"note": "non-dict result"}
        except Exception as e:
            return {"note": f"{tool_name} failed: {e}"}

    def _arxiv_tool_fn(q: str) -> str:
        # Legacy direct call (no registry). Add transparency prints.
        print_info("Using tool: legacy_arxiv_search")
        res = arxiv_search(query=q, from_year=None, limit=5)
        _print_summary_and_sources(res if isinstance(res, dict) else {})
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

    def _guidelines_tool_fn(query: str) -> str:
        """Search for research methodology and mentorship guidelines from curated sources."""
        try:
            from .core.orchestrator import Orchestrator
            from .tools import auto_discover
            
            # Ensure tools are discovered
            auto_discover()
            
            orch = Orchestrator()
            result = orch.execute_task(
                task="research_guidelines",
                inputs={"query": query, "topic": query},
                context={"goal": f"research mentorship guidance about {query}"}
            )
            
            if result["execution"]["executed"] and result["results"]:
                tool_result = result["results"]
                guidelines = tool_result.get("retrieved_guidelines", [])
                
                if not guidelines:
                    return "No specific guidelines found for this query. Try rephrasing or ask more specific questions about research methodology."
                
                # Format guidelines for agent consumption
                formatted_lines = [f"Found {len(guidelines)} relevant research guidelines:"]
                sources = []
                
                for guideline in guidelines:
                    guide_id = guideline.get("guide_id", "unknown")
                    source_type = guideline.get("source_type", "Research guidance")
                    source_domain = guideline.get("source_domain", "")
                    content = guideline.get("content", "")[:300]  # Truncate for token efficiency
                    
                    # Extract source domain for display
                    if source_domain and source_domain not in sources:
                        sources.append(source_domain)
                    
                    formatted_lines.append(f"GUIDELINE [{guide_id}]:")
                    formatted_lines.append(f"Source: {source_type}")
                    formatted_lines.append(f"Content: {content}")
                    formatted_lines.append("---")
                
                # Add instruction for agent
                formatted_lines.append(
                    "\nUse these guidelines to provide evidence-based research advice. "
                    "Reference specific guidelines as [guide_id] in your response."
                )
                
                # Add sources section at the end
                if sources:
                    formatted_lines.append("\n\nSource:")
                    for i, source in enumerate(sources, 1):
                        formatted_lines.append(f"{i}. {source}")
                
                return "\n".join(formatted_lines)
            else:
                return "Guidelines search temporarily unavailable. Please try again later."
                
        except Exception as e:
            return f"Error searching guidelines: {str(e)}"

    def _o3_search_tool_fn(q: str) -> str:
        """Registry-backed O3 literature search with transparency printing."""
        result = _registry_tool_call("o3_search", {"query": q, "limit": 8})
        items = (result.get("results") if isinstance(result, dict) else []) or []
        if not items:
            note = (result or {}).get("note", "No results") if isinstance(result, dict) else "No results"
            return str(note)
        lines: list[str] = []
        for it in items[:5]:
            title = it.get("title") or it.get("paper_title") or "result"
            year = it.get("year") or it.get("published") or ""
            url = it.get("url") or (it.get("urls", {}) or {}).get("paper") or ""
            suffix = f" ({year})" if year else ""
            link = f" -> {url}" if url else ""
            lines.append(f"- {title}{suffix}{link}")
        return "\n".join(lines)

    def _searchthearxiv_tool_fn(q: str) -> str:
        """Registry-backed semantic arXiv search (searchthearxiv.com) with transparency printing."""
        result = _registry_tool_call("searchthearxiv_search", {"query": q, "limit": 10})
        papers = (result.get("papers") if isinstance(result, dict) else []) or []
        if not papers:
            note = (result or {}).get("note", "No results") if isinstance(result, dict) else "No results"
            return str(note)
        lines: list[str] = []
        for p in papers[:5]:
            title = p.get("title") or "paper"
            year = p.get("year") or ""
            url = p.get("url") or ""
            suffix = f" ({year})" if year else ""
            link = f" -> {url}" if url else ""
            lines.append(f"- {title}{suffix}{link}")
        return "\n".join(lines)

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
            name="o3_search",
            func=_o3_search_tool_fn,
            description=(
                "Consolidated literature search using O3 reasoning across arXiv and OpenReview. "
                "Prefer this over legacy arxiv_search; includes transparency logs and sources. "
                "Input: research topic. Returns key papers with links."
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
        Tool(
            name="research_guidelines",
            func=_guidelines_tool_fn,
            description=(
                "Search curated research methodology and mentorship guidelines from expert sources. "
                "USE THIS TOOL FOR ALL RESEARCH MENTORSHIP QUESTIONS including: "
                "research advice, methodology guidance, PhD help, problem selection, research taste development, academic career guidance, "
                "research strategy decisions, publication dilemmas, research evaluation questions, and academic career planning. "
                "Specifically use when users ask about: "
                "- Research direction uncertainty ('no one else is working on this', 'red flag or opportunity', 'unique research direction') "
                "- Problem worthiness ('worth pursuing vs distraction', 'should I work on this problem', 'is this important') "
                "- Negative results ('approach doesn\'t work', 'should I publish negative results', 'my method failed') "
                "- Novelty concerns ('not sure novel enough', 'how to evaluate novelty', 'is this contribution significant') "
                "- Publication decisions ('should I publish this', 'where to publish', 'ready for publication') "
                "- Research taste and judgment ('developing research taste', 'how to choose problems', 'research intuition') "
                "- Academic career guidance ('career planning', 'PhD advice', 'research skills development') "
                "- Methodology questions ('research methodology', 'experiment design', 'evaluation methods') "
                "Input: any research mentorship question, dilemma, or uncertainty. Examples: "
                "'I found an interesting research direction but I\'m worried no one else is working on it. Is that a red flag or an opportunity?', "
                "'My results are negative - my approach doesn\'t work. Should I publish this or try something else?', "
                "'I have some interesting results but I\'m not sure they\'re \'novel\' enough for publication. How do I evaluate this?', "
                "'How do I know if a research problem is worth pursuing vs just a distraction?' "
                "Returns: structured guidelines from authoritative sources with source attribution."
            ),
        ),
        Tool(
            name="searchthearxiv_search",
            func=_searchthearxiv_tool_fn,
            description=(
                "Semantic arXiv search via searchthearxiv.com. Use for natural language queries. "
                "Includes transparency logs and sources. Input: research query."
            ),
        ),
    ]
    return tools
