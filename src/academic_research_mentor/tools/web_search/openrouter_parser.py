from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_URL_RE = re.compile(r"https?://[^\s)\]]+")


def parse_openrouter_content(content: Any, *, limit: int) -> Tuple[Dict[str, Any], Optional[str]]:
    text = _normalize_content(content)
    if not text:
        return {"results": [], "summary": None}, "empty content"

    structured = _try_parse_json(text)
    if structured is not None:
        results = structured.get("results")
        summary = structured.get("summary") or structured.get("answer")
        return {
            "results": results if isinstance(results, list) else [],
            "summary": str(summary).strip() if summary else None,
        }, None

    fallback = _extract_links_and_summary(text, limit=limit)
    if fallback.get("results") or fallback.get("summary"):
        return fallback, None

    return {"results": [], "summary": text[:600]}, None


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        value = content.get("text") or content.get("content") or ""
        return str(value).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                value = block.get("text") or block.get("content") or ""
                if value:
                    parts.append(str(value))
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(content or "").strip()


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text]
    if text.startswith("```"):
        trimmed = re.sub(r"^```(?:json)?", "", text).strip()
        trimmed = re.sub(r"```$", "", trimmed).strip()
        candidates.append(trimmed)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_links_and_summary(text: str, *, limit: int) -> Dict[str, Any]:
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    summary: Optional[str] = None
    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for line in lines:
        if summary is None and "http" not in line.lower() and len(line) > 24:
            summary = line

        for title, url in _LINK_RE.findall(line):
            clean_url = url.rstrip(").,")
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            snippet = _LINK_RE.sub("", line).strip(" -:")
            results.append({"title": title.strip() or "Web result", "url": clean_url, "snippet": snippet[:400]})
            if len(results) >= limit:
                return {"results": results, "summary": summary}

    for line in lines:
        for match in _URL_RE.finditer(line):
            clean_url = match.group(0).rstrip(").,")
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            results.append({"title": clean_url, "url": clean_url, "snippet": line[:400]})
            if len(results) >= limit:
                return {"results": results, "summary": summary}

    if summary is None and lines:
        summary = lines[0][:500]
    return {"results": results, "summary": summary}
