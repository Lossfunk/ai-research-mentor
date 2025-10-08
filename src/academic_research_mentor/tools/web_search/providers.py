from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ...citations import Citation, CitationFormatter

try:  # pragma: no cover - optional dependency guard
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore
HTTPX_AVAILABLE = httpx is not None
def execute_tavily_search(
    client: Any,
    *,
    query: str,
    limit: int,
    search_depth: str,
    include_answer: bool,
    domain: Optional[str],
    mode: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if client is None:
        return None, "tavily client unavailable"

    try:
        response = client.search(
            query=query,
            max_results=limit,
            search_depth=search_depth,
            include_domains=[domain] if domain else None,
            include_raw_content=True,
            include_images=False,
            include_answer=include_answer,
        )
    except Exception as exc:  # pragma: no cover - network/init errors
        return None, str(exc)

    results_raw = response.get("results") if isinstance(response, dict) else None
    results_raw = results_raw if isinstance(results_raw, list) else []

    formatted = _format_results(
        query=query,
        entries=results_raw,
        limit=limit,
        domain=domain,
        mode=mode,
        provider="tavily",
        note_suffix=f"Tavily ({mode})",
        summary=response.get("answer") if include_answer else None,
        search_depth=search_depth,
    )
    return formatted, None


def execute_openrouter_search(
    *,
    query: str,
    limit: int,
    domain: Optional[str],
    mode: str,
    config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    api_key = str(config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        return None, "OPENROUTER_API_KEY not configured"
    if httpx is None:
        return None, "httpx unavailable for OpenRouter"

    max_results = max(1, min(int(config.get("openrouter_max_results", limit)), 6))
    raw_model = str(config.get("openrouter_model") or "").strip()
    if raw_model:
        if ":online" not in raw_model and not config.get("openrouter_disable_online_suffix"):
            model = f"{raw_model}:online"
        else:
            model = raw_model
    else:
        model = "openrouter/auto:online"
    system_message = (
        "You are a web search aggregation assistant. Return STRICT JSON with keys 'results' and optional 'summary'. "
        "Each item in 'results' must have 'title', 'url', and 'snippet'. Do not include markdown fences or additional commentary."
    )
    user_payload: Dict[str, Any] = {"query": query, "max_results": max_results}
    if domain:
        user_payload["requested_domain"] = domain

    plugins: List[Dict[str, Any]] = []
    if not config.get("openrouter_disable_web_plugin"):
        plugin_entry: Dict[str, Any] = {"id": "web", "max_results": max_results}
        search_prompt = config.get("openrouter_search_prompt")
        if search_prompt:
            plugin_entry["search_prompt"] = str(search_prompt)
        plugins.append(plugin_entry)
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "temperature": 0,
    }
    if plugins:
        body["plugins"] = plugins

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = config.get("openrouter_referer") or os.getenv("OPENROUTER_APP_URL")
    title = config.get("openrouter_title") or os.getenv("OPENROUTER_APP_NAME", "Academic Research Mentor")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    try:
        with httpx.Client(timeout=20) as client:  # type: ignore[union-attr]
            resp = client.post(  # type: ignore[union-attr]
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
            )
    except Exception as exc:  # pragma: no cover - network errors
        return None, f"OpenRouter request failed: {exc}"

    if resp.status_code >= 400:  # type: ignore[union-attr]
        return None, f"OpenRouter HTTP {resp.status_code}"  # type: ignore[union-attr]

    try:
        payload = resp.json()  # type: ignore[union-attr]
    except Exception as exc:
        return None, f"OpenRouter invalid JSON: {exc}"

    content = ""
    try:
        choices = payload.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "")
    except Exception:
        content = ""

    if not content:
        return None, "OpenRouter response missing content"

    try:
        parsed = _parse_json_block(content)
    except ValueError as exc:
        return None, f"OpenRouter content parse error: {exc}"

    results_section = parsed.get("results")
    if not isinstance(results_section, list):
        return None, "OpenRouter response missing results"

    formatted = _format_results(
        query=query,
        entries=results_section,
        limit=limit,
        domain=domain,
        mode=mode,
        provider="openrouter-web",
        note_suffix="OpenRouter web plugin",
        summary=parsed.get("summary"),
        search_depth=None,
    )
    return formatted, None


def _parse_json_block(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _format_results(
    *,
    query: str,
    entries: List[Dict[str, Any]],
    limit: int,
    domain: Optional[str],
    mode: str,
    provider: str,
    note_suffix: str,
    summary: Optional[str],
    search_depth: Optional[str],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    citations: List[Citation] = []
    formatter = CitationFormatter()

    for idx, entry in enumerate(entries[:limit], start=1):
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or entry.get("name") or "Untitled result").strip()
        url = str(entry.get("url") or entry.get("link") or "").strip()
        snippet = str(entry.get("snippet") or entry.get("content") or entry.get("description") or "").strip()
        published = entry.get("published") or entry.get("published_at")
        score = entry.get("score") or entry.get("relevance")
        source = entry.get("source") or entry.get("domain") or "web"

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet[:600] if snippet else None,
                "summary": snippet or None,
                "score": score,
                "published": published,
                "source": source,
            }
        )

        if url or snippet:
            citations.append(
                Citation(
                    id=f"{provider.replace('-', '_')}_{idx:02d}",
                    title=title,
                    url=url or "https://openrouter.ai",
                    source=source,
                    authors=[],
                    year=None,
                    venue=domain or "Web",
                    snippet=(snippet or "")[:300] or None,
                )
            )

    output: Dict[str, Any] = {
        "query": query,
        "results": results,
        "total_results": len(results),
        "note": f"Web search via {note_suffix}",
        "metadata": {
            "provider": provider,
            "mode": mode,
            "limit": limit,
            "domain": domain,
        },
    }
    if search_depth:
        output["metadata"]["search_depth"] = search_depth
    if summary:
        output["summary"] = str(summary).strip()
    if citations:
        output["citations"] = formatter.to_output_block(citations)

    return output