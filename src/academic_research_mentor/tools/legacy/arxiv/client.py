from __future__ import annotations

import html
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS: float = 15.0
DEFAULT_MAX_RETRIES: int = 2

from .query import extract_phrases_and_tokens, build_arxiv_query, relevance_score


class _SimpleResponse:
    def __init__(self, text: str, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Any:
        import json as _json
        return _json.loads(self.text)


def _fetch_with_retry(url: str, params: Optional[Dict[str, Any]] = None, timeout_s: float = DEFAULT_TIMEOUT_SECONDS):
    last_exc: Optional[Exception] = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            if httpx is not None:
                with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": "AcademicResearchMentor/1.0"}) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response
            else:
                import urllib.request as _urlrequest
                full_url = url
                if params:
                    sep = '&' if ('?' in url) else '?'
                    full_url = f"{url}{sep}{urlencode(params)}"
                req = _urlrequest.Request(full_url, headers={"User-Agent": "AcademicResearchMentor/1.0"})
                with _urlrequest.urlopen(req, timeout=timeout_s) as resp:  # nosec - simple GET
                    data = resp.read()
                    try:
                        encoding = resp.headers.get_content_charset()  # type: ignore[attr-defined]
                    except Exception:
                        encoding = None
                    text = data.decode(encoding or "utf-8", errors="replace")
                    status = getattr(resp, "status", 200)
                    headers = dict(resp.headers.items()) if hasattr(resp, "headers") else {}
                    return _SimpleResponse(text=text, status_code=status, headers=headers)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < DEFAULT_MAX_RETRIES:
                import time as _time
                _time.sleep(2.0)
    LOGGER.debug("HTTP fetch failed for %s params=%s exc=%s", url, params, last_exc)
    return None


def arxiv_search(query: str, from_year: Optional[int] = None, limit: int = 10) -> Dict[str, Any]:
    if httpx is None:
        return {"papers": [], "note": "httpx unavailable; could not query arXiv."}

    base_url = "https://export.arxiv.org/api/query"
    full_query = build_arxiv_query(query, from_year)

    sort_by = "relevance"
    if from_year is not None and from_year >= 2022:
        sort_by = "submittedDate"

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max(1, min(int(max(limit * 2.5, limit + 10)), 30)),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    resp = _fetch_with_retry(base_url, params=params)
    if resp is None:
        return {"papers": [], "note": "arXiv request failed or timed out."}

    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        parsed: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title_text = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            title_text = html.unescape(" ".join(title_text.split()))
            summary_text = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            summary_text = html.unescape(" ".join(summary_text.split()))
            authors = [a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)]
            link = entry.find("atom:link[@rel='alternate']", ns)
            link_href = link.get("href") if link is not None else entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
            year_val = None
            if len(published) >= 4 and published[:4].isdigit():
                year_val = int(published[:4])
            parsed.append({
                "title": title_text,
                "summary": summary_text,
                "authors": authors,
                "year": year_val,
                "venue": "arXiv",
                "url": link_href,
            })

        if not parsed:
            phrases, tokens = extract_phrases_and_tokens(query)
            if phrases or tokens:
                relaxed_terms = []
                for phr in phrases:
                    safe = phr.replace('"', '')
                    relaxed_terms.append(f'(ti:"{safe}" OR abs:"{safe}" OR all:"{safe}")')
                important_tokens = [tok for tok in tokens if len(tok) >= 3][:3]
                for tok in important_tokens:
                    relaxed_terms.append(f'(ti:{tok} OR abs:{tok})')
                if relaxed_terms:
                    relaxed_query = " AND ".join(relaxed_terms)
                    relaxed_params = dict(params)
                    relaxed_params["search_query"] = relaxed_query
                    relaxed_params["sortBy"] = "relevance"
                    relaxed_params["max_results"] = min(20, params["max_results"])  # Smaller
                    resp2 = _fetch_with_retry(base_url, params=relaxed_params)
                    if resp2 is not None:
                        root2 = ET.fromstring(resp2.text)
                        parsed = []
                        for entry in root2.findall("atom:entry", ns):
                            title_text = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                            title_text = html.unescape(" ".join(title_text.split()))
                            summary_text = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                            summary_text = html.unescape(" ".join(summary_text.split()))
                            authors = [a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)]
                            link = entry.find("atom:link[@rel='alternate']", ns)
                            link_href = link.get("href") if link is not None else entry.findtext("atom:id", default="", namespaces=ns)
                            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
                            year_val = None
                            if len(published) >= 4 and published[:4].isdigit():
                                year_val = int(published[:4])
                            parsed.append({
                                "title": title_text,
                                "summary": summary_text,
                                "authors": authors,
                                "year": year_val,
                                "venue": "arXiv",
                                "url": link_href,
                            })

        phrases, tokens = extract_phrases_and_tokens(query)
        for item in parsed:
            item["_local_score"] = relevance_score(item.get("title", ""), item.get("summary", ""), phrases, tokens)
        parsed.sort(key=lambda x: x.get("_local_score", 0.0), reverse=True)

        non_trivial = [p for p in parsed if p.get("_local_score", 0.0) > 0.0]
        chosen = non_trivial if len(non_trivial) >= max(1, min(int(limit), 10)) // 2 else parsed

        papers = []
        for p in chosen[: max(1, int(limit))]:
            p.pop("_local_score", None)
            papers.append(p)

        note = None
        if not papers and parsed:
            note = "Relevance filter was strict; returning API results would have been off-topic."
        return {"papers": papers, "note": note}
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("arXiv parse error: %s", exc)
        return {"papers": [], "note": "Failed to parse arXiv response."}
