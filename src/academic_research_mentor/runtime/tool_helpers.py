from __future__ import annotations

from typing import Any

from ..rich_formatter import print_agent_reasoning
from .telemetry import record_tool_usage


def print_summary_and_sources(result: dict | None) -> None:
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
        if summary_lines or sources:
            parts: list[str] = []
            if summary_lines:
                parts.append("Found:\n" + "\n".join(summary_lines[:3]))
            if sources:
                parts.append("Sources: " + ", ".join(sources[:5]))
            print_agent_reasoning("\n".join(parts))
    except Exception:
        # Transparency is best-effort; never fail the interaction
        pass


def registry_tool_call(tool_name: str, payload: dict) -> dict:
    try:
        from ..tools import auto_discover as _auto, get_tool as _get
        _auto()
        tool = _get(tool_name)
        if tool is None:
            print_agent_reasoning(f"Using tool: {tool_name} (unavailable)")
            return {"note": f"tool {tool_name} unavailable"}
        print_agent_reasoning(f"Using tool: {tool_name}")
        record_tool_usage(tool_name)
        result = tool.execute(payload, {"goal": payload.get("query", "")})
        print_summary_and_sources(result if isinstance(result, dict) else {})
        return result if isinstance(result, dict) else {"note": "non-dict result"}
    except Exception as e:
        return {"note": f"{tool_name} failed: {e}"}
