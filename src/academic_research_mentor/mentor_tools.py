"""
AI Research Mentor Tools

This module defines minimal, pragmatic tool functions for the AI Research Mentor.
- Availability checks per tool
- One-retry with short backoff and timeouts
- Graceful degradation when remote APIs are unavailable
- Gemini-compatible function declarations for tool-calling

Environment and setup (suggested):
- Conda env: lossfunk
- Use uv instead of pip for Python deps, e.g.:
  - conda activate lossfunk
  - uv pip install --system httpx python-dotenv

Dependencies: httpx (for HTTP requests). Everything else uses the standard library.
"""
from __future__ import annotations

import time
import html
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import re
import json as _json  # stdlib JSON fallback
import urllib.request as _urlrequest  # stdlib HTTP fallback

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # runtime availability will be checked


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS: float = 15.0
DEFAULT_MAX_RETRIES: int = 2


def is_tool_available(tool_name: str) -> bool:
    """Best-effort availability check for remote-backed tools.

    - For network tools, ensure httpx is importable.
    - For local heuristic tools, always True.
    """
    network_tools = {"arxiv_search", "openreview_fetch", "venue_guidelines_get"}
    if tool_name in network_tools:
        return httpx is not None
    return True


class _SimpleResponse:
    """Lightweight response proxy for stdlib HTTP fallback.

    Provides .text and .json() compatible with the subset used here.
    """

    def __init__(self, text: str, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Any:
        return _json.loads(self.text)


def _fetch_with_retry(url: str, params: Optional[Dict[str, Any]] = None, timeout_s: float = DEFAULT_TIMEOUT_SECONDS):  # type: ignore[override]
    """HTTP GET with tiny retry logic.

    Uses httpx when available; otherwise falls back to urllib from the stdlib.
    Returns an object exposing .text and optionally .json().
    """
    last_exc: Optional[Exception] = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            if httpx is not None:
                with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": "AcademicResearchMentor/1.0"}) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response
            else:
                full_url = url
                if params:
                    # Preserve existing querystring if any
                    sep = '&' if ('?' in url) else '?'
                    full_url = f"{url}{sep}{urlencode(params)}"
                req = _urlrequest.Request(full_url, headers={"User-Agent": "AcademicResearchMentor/1.0"})
                with _urlrequest.urlopen(req, timeout=timeout_s) as resp:  # nosec - simple GET
                    data = resp.read()
                    # Attempt to pick a reasonable encoding
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
                time.sleep(2.0)
    LOGGER.debug("HTTP fetch failed for %s params=%s exc=%s", url, params, last_exc)
    return None


# ----------------------
# Tool: arxiv_search
# ----------------------

def _extract_phrases_and_tokens(raw_query: str) -> Tuple[List[str], List[str]]:
    """Extract quoted phrases and remaining keyword tokens from a raw query string.

    - Preserves user-provided fielded terms like ti:, abs:, au:, cat:, all:
    - Returns (phrases, tokens) lowered and stripped of minimal stopwords
    """
    if not raw_query:
        return [], []

    text = raw_query.strip()

    # If the query already contains arXiv field operators, do not tokenize aggressively
    if re.search(r"\b(?:ti|abs|au|cat|all):", text, flags=re.IGNORECASE):
        return [text], []

    # Pull out quoted phrases first
    phrases = [m.group(1).strip().lower() for m in re.finditer(r'"([^"]+)"', text)]
    text_wo_quotes = re.sub(r'"[^"]+"', ' ', text)

    # Tokenize remaining text on non-alphanumeric boundaries
    raw_tokens = re.split(r"[^A-Za-z0-9_-]+", text_wo_quotes)
    raw_tokens = [tok.lower() for tok in raw_tokens if tok]

    # Minimal stopwords - keep important ML/AI terms that were previously filtered
    stopwords = {
        "a", "an", "the", "and", "or", "for", "to", "of", "on", "in", "with", "by", "from",
        "at", "as", "is", "are", "be", "being", "into", "via", "using", "use", "based",
        "towards", "toward", "new", "novel",
    }
    tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= 2]

    return phrases, tokens


def _detect_ml_domain(query: str) -> Optional[str]:
    """Detect the most relevant ML/CS domain from query text for category filtering."""
    query_lower = query.lower()
    
    # Map domain keywords to arXiv categories
    domain_keywords = {
        "cs.LG": ["machine learning", "neural network", "deep learning", "diffusion", "transformer", 
                  "gan", "vae", "reinforcement learning", "supervised learning", "unsupervised learning",
                  "training", "optimization", "gradient", "backprop", "lstm", "cnn", "rnn"],
        "cs.CV": ["computer vision", "image", "video", "visual", "detection", "segmentation", 
                  "classification", "recognition", "object detection", "face", "ocr", "opencv"],
        "cs.CL": ["natural language", "nlp", "text", "language model", "bert", "gpt", "llm",
                  "translation", "sentiment", "tokenization", "parsing", "dialogue"],
        "cs.AI": ["artificial intelligence", "planning", "reasoning", "knowledge", "expert system",
                  "agent", "multi-agent", "search algorithm", "heuristic"],
        "cs.RO": ["robot", "robotics", "manipulation", "navigation", "control", "autonomous"],
        "stat.ML": ["statistical learning", "bayesian", "mcmc", "inference", "probability", "statistics"],
        # Multimodal and vision-language terms can map to CV or CL; prefer CV here
        "cs.CV": ["multimodal", "vision-language", "vlm", "image-text", "cross-modal", "grounding", "clip"],
    }
    
    # Score each domain
    domain_scores = {}
    for category, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            domain_scores[category] = score
    
    if not domain_scores:
        return None
    
    # Return the domain with highest score
    return max(domain_scores, key=domain_scores.get)


def _build_arxiv_query(raw_query: str, from_year: Optional[int]) -> str:
    """Construct a more precise arXiv query using best practices.

    Strategy:
    1. Use field-specific searches (ti:, abs:, cat:) for better precision
    2. Apply category filtering for ML/AI domains when detected
    3. Prioritize title matches over abstract matches
    4. Use quoted phrases for exact matching
    5. Apply date filtering when specified
    """
    phrases, tokens = _extract_phrases_and_tokens(raw_query)

    clauses: List[str] = []
    
    # If user provided fielded terms, use as-is
    if phrases == [raw_query] and not tokens:
        clauses.append(raw_query.strip())
    else:
        # Detect domain for category filtering
        detected_domain = _detect_ml_domain(raw_query)
        if detected_domain:
            clauses.append(f"cat:{detected_domain}")

        # Build phrase clauses (AND across phrases)
        phrase_clauses: List[str] = []
        for phr in phrases:
            safe = phr.replace('"', '')
            # Also consider hyphen/space variants within phrases
            hyphen_variant = safe.replace('-', ' ')
            if hyphen_variant != safe and len(hyphen_variant) >= 3:
                phrase_clauses.append(f'(ti:"{safe}" OR abs:"{safe}" OR ti:"{hyphen_variant}" OR abs:"{hyphen_variant}")')
            else:
                phrase_clauses.append(f'(ti:"{safe}" OR abs:"{safe}")')

        # Token clauses (OR across a limited subset to increase recall)
        token_terms: List[str] = []
        # Prefer longer/more specific tokens first
        sorted_tokens = sorted(tokens, key=lambda t: (-len(t), t))[:5]
        for tok in sorted_tokens:
            # Add hyphen-space variant expansion
            variants = {tok}
            if '-' in tok:
                variants.add(tok.replace('-', ' '))
            # Build fielded clauses for each variant
            variant_clauses: List[str] = []
            for v in variants:
                if len(v) >= 4:
                    variant_clauses.append(f'(ti:{v} OR abs:{v})')
                else:
                    variant_clauses.append(f'(ti:{v} OR abs:{v} OR all:{v})')
            token_terms.append('(' + ' OR '.join(variant_clauses) + ')')

        if phrase_clauses:
            clauses.extend(phrase_clauses)
        if token_terms:
            # Join tokens with OR to avoid over-constraining queries
            clauses.append('(' + ' OR '.join(token_terms) + ')')

    # Add date filter if specified
    if from_year is not None:
        clauses.append(f"submittedDate:[{from_year}01010000+TO+300001010000]")

    # Join with AND between major groups (domain, phrases, token OR-group)
    return " AND ".join(clauses) if clauses else raw_query.strip()


def _relevance_score(title: str, summary: str, phrases: List[str], tokens: List[str]) -> float:
    """Enhanced heuristic scorer that better prioritizes relevant papers."""
    t = (title or "").lower()
    s = (summary or "").lower()
    score = 0.0
    
    # Heavily weight exact phrase matches in title
    for p in phrases:
        if p in t:
            score += 5.0  # Increased weight for title phrase matches
        elif p in s:
            score += 2.0  # Good weight for abstract phrase matches
    
    # Token scoring with word boundary checks and length consideration
    for tok in tokens:
        token_pattern = rf"\b{re.escape(tok)}\b"
        if re.search(token_pattern, t):
            # Weight longer, more specific tokens higher
            weight = 1.5 if len(tok) >= 4 else 1.0
            score += weight
        elif re.search(token_pattern, s):
            # Weight longer, more specific tokens higher in abstracts too
            weight = 0.8 if len(tok) >= 4 else 0.5
            score += weight
    
    # Bonus for multiple token matches (indicates topical coherence)
    title_token_matches = sum(1 for tok in tokens if re.search(rf"\b{re.escape(tok)}\b", t))
    if title_token_matches >= 2:
        score += title_token_matches * 0.5  # Coherence bonus
    
    # Small penalty for very long titles (often less focused)
    if len(title) > 100:
        score *= 0.95
    
    return score


def arxiv_search(query: str, from_year: Optional[int] = None, limit: int = 10) -> Dict[str, Any]:
    """Search arXiv via its Atom API and return minimal paper metadata.

    Returns a dict: {"papers": [{title, authors, year, venue, url}], "note": str?}
    Degrades gracefully to empty results with a note on failure/unavailability.
    """
    if not is_tool_available("arxiv_search"):
        return {"papers": [], "note": "httpx unavailable; could not query arXiv."}

    base_url = "https://export.arxiv.org/api/query"
    # Build a precise query and optimize sorting strategy
    full_query = _build_arxiv_query(query, from_year)
    
    # Choose sorting strategy based on query characteristics
    sort_by = "relevance"  # Default to relevance for better topical matching
    if from_year is not None and from_year >= 2022:
        # If looking for recent papers, balance relevance with recency
        sort_by = "submittedDate"
    
    params = {
        "search_query": full_query,
        "start": 0,
        # Fetch extra results for local re-ranking, capped at 30 for reasonable latency
        "max_results": max(1, min(int(max(limit * 2.5, limit + 10)), 30)),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    resp = _fetch_with_retry(base_url, params=params)
    if resp is None:
        return {"papers": [], "note": "arXiv request failed or timed out."}

    try:
        # Parse Atom XML minimally
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

        # If no results, try a more targeted fallback strategy
        if not parsed:
            phrases, tokens = _extract_phrases_and_tokens(query)
            # Try a relaxed query but still maintain some structure
            if phrases or tokens:
                relaxed_terms = []
                
                # Keep quoted phrases but search more broadly
                for phr in phrases:
                    safe = phr.replace('"', '')
                    relaxed_terms.append(f'(ti:"{safe}" OR abs:"{safe}" OR all:"{safe}")')
                
                # For tokens, be more selective - only use the most important ones
                important_tokens = [tok for tok in tokens if len(tok) >= 3][:3]  # Limit to top 3 longest tokens
                for tok in important_tokens:
                    relaxed_terms.append(f'(ti:{tok} OR abs:{tok})')
                
                if relaxed_terms:
                    relaxed_query = " AND ".join(relaxed_terms)
                    relaxed_params = dict(params)
                    relaxed_params["search_query"] = relaxed_query
                    relaxed_params["sortBy"] = "relevance"
                    relaxed_params["max_results"] = min(20, params["max_results"])  # Smaller result set for fallback
                    
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

        # Local re-ranking to boost topical relevance
        phrases, tokens = _extract_phrases_and_tokens(query)
        for item in parsed:
            item["_local_score"] = _relevance_score(item.get("title", ""), item.get("summary", ""), phrases, tokens)
        parsed.sort(key=lambda x: x.get("_local_score", 0.0), reverse=True)

        # Keep only items with non-trivial score when possible, but never return empty if API had results
        non_trivial = [p for p in parsed if p.get("_local_score", 0.0) > 0.0]
        chosen = non_trivial if len(non_trivial) >= max(1, min(int(limit), 10)) // 2 else parsed

        # Trim to requested limit and remove scoring key
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


# ----------------------
# Tool: openreview_fetch
# ----------------------

def openreview_fetch(query: str, limit: int = 5) -> Dict[str, Any]:
    """Best-effort OpenReview fetch.

    OpenReview APIs vary; we attempt a permissive search endpoint and degrade gracefully.
    Returns: {"threads": [{paper_title, venue?, year?, urls:{paper?, forum?}, excerpts?}], "note"?}
    """
    if not is_tool_available("openreview_fetch"):
        return {"threads": [], "note": "httpx unavailable; could not query OpenReview."}

    # Heuristic endpoint: search notes (fallback if unavailable)
    base_url = "https://api.openreview.net/notes"
    # Some deployments require additional params; keep it minimal
    params = {"term": query, "limit": max(1, min(int(limit), 10))}

    resp = _fetch_with_retry(base_url, params=params)
    if resp is None:
        # Fallback: lightweight web search for OpenReview forum links
        threads = _openreview_web_fallback(query=query, limit=limit)
        if threads:
            return {"threads": threads, "note": "OpenReview API unavailable; used web fallback."}
        return {"threads": [], "note": "OpenReview request failed or timed out."}

    try:
        data = resp.json()
        notes = data.get("notes") or data.get("results") or []
        threads: List[Dict[str, Any]] = []
        for note in notes[: limit]:
            content = note.get("content") or {}
            title = content.get("title") or note.get("title") or ""
            forum = note.get("forum") or note.get("id")
            threads.append({
                "paper_title": title,
                "venue": content.get("venue") or content.get("venueid"),
                "year": content.get("year"),
                "urls": {
                    "paper": f"https://openreview.net/forum?id={forum}" if forum else None,
                    "forum": f"https://openreview.net/forum?id={forum}" if forum else None,
                },
                "excerpts": [content.get("abstract")] if content.get("abstract") else [],
            })
        if threads:
            return {"threads": threads, "note": None}
        # Try web fallback if API returned nothing
        web_threads = _openreview_web_fallback(query=query, limit=limit)
        if web_threads:
            return {"threads": web_threads, "note": "OpenReview API returned no results; used web fallback."}
        return {"threads": [], "note": "No results or incompatible API response."}
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("OpenReview parse error: %s", exc)
        # Try web fallback on parse error
        web_threads = _openreview_web_fallback(query=query, limit=limit)
        if web_threads:
            return {"threads": web_threads, "note": "Failed to parse OpenReview API; used web fallback."}
        return {"threads": [], "note": "Failed to parse OpenReview response."}


def _openreview_web_fallback(query: str, limit: int) -> List[Dict[str, Any]]:
    """Very lightweight fallback: search for OpenReview forum links via DuckDuckGo HTML.

    Avoids heavy scraping and external deps. Best-effort extraction of titles and forum URLs.
    """
    if httpx is None:
        return []
    try:
        # Search for forum pages first; include conference keywords optionally
        q = f"site:openreview.net/forum {query}"
        url = "https://duckduckgo.com/html/"
        params = {"q": q}
        resp = _fetch_with_retry(url, params=params, timeout_s=DEFAULT_TIMEOUT_SECONDS)
        if resp is None or not resp.text:
            return []
        html_text = resp.text
        # DuckDuckGo wraps links via /l/?uddg=... Extract and unquote actual target URLs
        import re as _re
        from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs, unquote as _unquote

        threads: List[Dict[str, Any]] = []
        seen_forums: set[str] = set()

        # Extract openreview links from result anchors
        for m in _re.finditer(r'href=\"(/l/\?[^\"]+)\"[^>]*>(.*?)</a>', html_text, flags=_re.IGNORECASE | _re.DOTALL):
            href = m.group(1)
            text = m.group(2)
            # Resolve uddg param
            try:
                qs = _parse_qs(_urlparse(href).query)
                uddg = qs.get("uddg", [""])[0]
                target = _unquote(uddg)
            except Exception:
                target = ""
            if "openreview.net/forum" not in target:
                continue
            # Normalize forum id
            try:
                forum_id = _parse_qs(_urlparse(target).query).get("id", [""])[0]
            except Exception:
                forum_id = ""
            if not forum_id or forum_id in seen_forums:
                continue
            seen_forums.add(forum_id)
            # Clean title text
            clean_title = html.unescape(_re.sub(r"<[^>]+>", " ", text)).strip()
            threads.append({
                "paper_title": clean_title or "OpenReview paper",
                "venue": None,
                "year": None,
                "urls": {"paper": target, "forum": target},
                "excerpts": [],
            })
            if len(threads) >= max(1, int(limit)):
                break
        return threads
    except Exception:
        return []


# ----------------------
# Tool: venue_guidelines_get
# ----------------------

def venue_guidelines_get(venue: str, year: Optional[int] = None) -> Dict[str, Any]:
    """Return likely author guideline URLs for common venues.

    Avoid scraping for reliability. Provide URLs and minimal known fields when obvious.
    """
    normalized = venue.strip().upper()
    if year is None:
        # Default to near-future cycle; adjust as needed
        year = 2025

    guide_url: Optional[str] = None
    template_url: Optional[str] = None

    if normalized in {"ICLR"}:
        guide_url = f"https://iclr.cc/Conferences/{year}/AuthorGuide"
        template_url = None
    elif normalized in {"NEURIPS", "NIPS"}:
        guide_url = f"https://neurips.cc/Conferences/{year}/PaperInformation/AuthorGuidelines"
        template_url = None
    elif normalized in {"ACL"}:
        guide_url = "https://aclrollingreview.org/author_guidelines"  # rolling
    else:
        guide_url = None

    result: Dict[str, Any] = {
        "guidelines": {
            "page_limit": None,
            "template": None,
            "checklist_items": None,
            "ethics_policy": None,
            "artifact_policy": None,
            "anonymization": None,
            "deadlines": None,
            "urls": {"guide": guide_url, "template": template_url},
        }
    }
    return result


# ----------------------
# Tool: math_ground (heuristic)
# ----------------------

def math_ground(text_or_math: str, options: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """Heuristic math grounding.

    Returns a best-effort findings object without external dependencies.
    """
    text = text_or_math or ""
    findings: Dict[str, Any] = {
        "assumptions": [],
        "symbol_glossary": [],
        "dimensional_issues": [],
        "proof_skeleton": [],
        "references": [],
    }

    if "=>" in text or "implies" in text:
        findings["assumptions"].append("Ensure premises for implications are stated.")
    if "O(" in text or "Theta(" in text:
        findings["assumptions"].append("State complexity assumptions and input size definitions.")
    if any(tok in text for tok in ["d/dx", "∂", "partial"]):
        findings["symbol_glossary"].append("Define variables and constants used in derivatives.")
    if any(tok in text for tok in ["||", "norm", "L2", "L1"]):
        findings["symbol_glossary"].append("Clarify norm definitions and spaces.")

    # Tiny proof skeleton prompt
    findings["proof_skeleton"].extend([
        "State assumptions and definitions.",
        "Outline lemma(s) with clear dependencies.",
        "Provide main argument and bound(s).",
        "Conclude with conditions for equality or tightness.",
    ])

    return {"findings": findings}


# ----------------------
# Tool: methodology_validate (heuristic)
# ----------------------

def methodology_validate(plan: str, checklist: Optional[List[str]] = None) -> Dict[str, Any]:
    """Heuristic validation of an experiment plan.

    Flags common risks, suggests ablations, and notes reproducibility gaps.
    """
    text = plan.lower() if plan else ""

    risks: List[str] = []
    missing_controls: List[str] = []
    ablation_suggestions: List[str] = []
    reproducibility_gaps: List[str] = []
    sample_size_notes: Optional[str] = None

    if "leak" in text or "test set" in text and "train" in text:
        risks.append("Potential data leakage between train/test; ensure strict splits.")
    if "baseline" not in text:
        missing_controls.append("Add at least two strong baselines.")
    if "ablation" not in text:
        ablation_suggestions.append("Plan ablations for key components and hyperparameters.")
    if "seed" not in text:
        reproducibility_gaps.append("Specify seeds and report variance across ≥3 runs.")
    if "compute" in text or "gpu" in text:
        reproducibility_gaps.append("Document compute budget and runtime per experiment.")

    return {
        "report": {
            "risks": risks,
            "missing_controls": missing_controls,
            "ablation_suggestions": ablation_suggestions,
            "reproducibility_gaps": reproducibility_gaps,
            "sample_size_notes": sample_size_notes,
        }
    }


# ----------------------
# Gemini function declarations (JSON schema-like)
# ----------------------
GEMINI_FUNCTION_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "arxiv_search",
        "description": "Search arXiv for relevant papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "from_year": {"type": "number", "description": "Minimum publication year"},
                "limit": {"type": "number", "description": "Max results (≤25)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "openreview_fetch",
        "description": "Fetch related OpenReview threads (best-effort).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords and/or venue"},
                "limit": {"type": "number", "description": "Max threads (≤10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "venue_guidelines_get",
        "description": "Return likely author guideline URLs for a venue/year.",
        "parameters": {
            "type": "object",
            "properties": {
                "venue": {"type": "string", "description": "Venue name (e.g., ICLR)"},
                "year": {"type": "number", "description": "Target year"},
            },
            "required": ["venue"],
        },
    },
    {
        "name": "math_ground",
        "description": "Heuristic math grounding: assumptions, glossary, proof skeleton.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_or_math": {"type": "string", "description": "TeX or plain text"},
                "options": {
                    "type": "object",
                    "properties": {
                        "dimensional_check": {"type": "boolean"},
                        "assumptions_check": {"type": "boolean"},
                    },
                },
            },
            "required": ["text_or_math"],
        },
    },
    {
        "name": "methodology_validate",
        "description": "Heuristic validation of experiment plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "Design summary"},
                "checklist": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["plan"],
        },
    },
]


def get_gemini_tools_block() -> List[Dict[str, Any]]:
    """Produce the Gemini 'tools' block as used in the prompts main client."""
    return [{"function_declarations": GEMINI_FUNCTION_DECLARATIONS}]


def handle_mentor_function_call(function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch function calls to the appropriate mentor tool, with availability guards."""
    try:
        if function_name == "arxiv_search":
            return arxiv_search(
                query=str(function_args.get("query", "")),
                from_year=function_args.get("from_year"),
                limit=int(function_args.get("limit", 10)),
            )
        if function_name == "openreview_fetch":
            return openreview_fetch(
                query=str(function_args.get("query", "")),
                limit=int(function_args.get("limit", 5)),
            )
        if function_name == "venue_guidelines_get":
            return venue_guidelines_get(
                venue=str(function_args.get("venue", "")),
                year=function_args.get("year"),
            )
        if function_name == "math_ground":
            return math_ground(
                text_or_math=str(function_args.get("text_or_math", "")),
                options=function_args.get("options"),
            )
        if function_name == "methodology_validate":
            return methodology_validate(
                plan=str(function_args.get("plan", "")),
                checklist=function_args.get("checklist"),
            )
        return {"error": f"Unknown function: {function_name}"}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Tool call failed: %s args=%s", function_name, function_args)
        return {"error": f"Tool call failed for {function_name}: {exc}"}


if __name__ == "__main__":
    # Tiny smoke tests with graceful degradation
    print("is_available(arxiv_search)", is_tool_available("arxiv_search"))
    print("arxiv_search sample:", arxiv_search("diffusion models", from_year=2023, limit=3))
    print("openreview_fetch sample:", openreview_fetch("ICLR diffusion", limit=2))
    print("venue_guidelines_get sample:", venue_guidelines_get("ICLR", 2025))
    print("math_ground sample:", math_ground("d/dx f(x) = 0 implies stationary point; ||x||_2", {}))
    print("methodology_validate sample:", methodology_validate("We evaluate with baselines; report seeds and ablations.", []))
