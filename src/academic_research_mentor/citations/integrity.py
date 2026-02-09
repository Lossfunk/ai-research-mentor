from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx


_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"\b(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_BIB_ENTRY_SPLIT = re.compile(r"(?=@\w+\s*\{)", re.IGNORECASE)


def _extract_urls(text: str) -> List[str]:
    return sorted(set(m.group(0).rstrip(".,);]") for m in _URL_PATTERN.finditer(text or "")))


def _extract_dois(text: str) -> List[str]:
    return sorted(set(m.group(0).strip().rstrip(".,);]") for m in _DOI_PATTERN.finditer(text or "")))


def _extract_arxiv_ids(text: str) -> List[str]:
    return sorted(set(m.group(1).lower() for m in _ARXIV_PATTERN.finditer(text or "")))


def _parse_bib_entries(text: str) -> List[Dict[str, Any]]:
    if "@" not in (text or ""):
        return []
    chunks = [c.strip() for c in _BIB_ENTRY_SPLIT.split(text or "") if c.strip().startswith("@")]
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        lower = chunk.lower()
        missing = []
        for field in ("title", "author", "year"):
            if f"{field}=" not in lower and f"{field} =" not in lower:
                missing.append(field)
        if "journal=" not in lower and "journal =" not in lower and "booktitle=" not in lower and "booktitle =" not in lower:
            missing.append("journal|booktitle")
        key_match = re.search(r"@\w+\s*\{\s*([^,]+)", chunk)
        out.append(
            {
                "entry_key": key_match.group(1).strip() if key_match else "unknown",
                "missing_fields": missing,
                "ok": len(missing) == 0,
            }
        )
    return out


def _check_url(client: httpx.Client, url: str) -> Dict[str, Any]:
    try:
        resp = client.head(url, follow_redirects=True)
        status = int(resp.status_code)
        if status in (405, 403):
            resp = client.get(url, follow_redirects=True)
            status = int(resp.status_code)
        ok = status < 400
        return {"url": url, "ok": ok, "status_code": status}
    except Exception as exc:
        return {"url": url, "ok": False, "status_code": None, "error": str(exc)}


def _check_doi(client: httpx.Client, doi: str) -> Dict[str, Any]:
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        resp = client.get(url)
        status = int(resp.status_code)
        ok = status == 200
        return {"doi": doi, "ok": ok, "status_code": status}
    except Exception as exc:
        return {"doi": doi, "ok": False, "status_code": None, "error": str(exc)}


def _check_arxiv(client: httpx.Client, arxiv_id: str) -> Dict[str, Any]:
    url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
    try:
        resp = client.get(url)
        status = int(resp.status_code)
        body = (resp.text or "").lower()
        ok = status == 200 and "<entry>" in body
        return {"arxiv_id": arxiv_id, "ok": ok, "status_code": status}
    except Exception as exc:
        return {"arxiv_id": arxiv_id, "ok": False, "status_code": None, "error": str(exc)}


def audit_reference_text(
    text: str,
    *,
    extra_urls: Optional[List[str]] = None,
    check_urls: bool = True,
    verify_doi: bool = True,
    verify_arxiv: bool = False,
    timeout_seconds: float = 8.0,
) -> Dict[str, Any]:
    urls = _extract_urls(text) + [u.strip() for u in (extra_urls or []) if isinstance(u, str) and u.strip()]
    urls = sorted(set(urls))
    dois = _extract_dois(text)
    arxiv_ids = _extract_arxiv_ids(text)
    bib_entries = _parse_bib_entries(text)

    url_checks: List[Dict[str, Any]] = []
    doi_checks: List[Dict[str, Any]] = []
    arxiv_checks: List[Dict[str, Any]] = []
    if check_urls or verify_doi or verify_arxiv:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers={"User-Agent": "metis-citation-audit/1.0"}) as client:
            if check_urls:
                url_checks = [_check_url(client, u) for u in urls]
            if verify_doi:
                doi_checks = [_check_doi(client, d) for d in dois]
            if verify_arxiv:
                arxiv_checks = [_check_arxiv(client, a) for a in arxiv_ids]

    issues: List[str] = []
    suggestions: List[str] = []
    broken_urls = [c for c in url_checks if not c.get("ok")]
    bad_dois = [c for c in doi_checks if not c.get("ok")]
    bad_arxiv = [c for c in arxiv_checks if not c.get("ok")]
    weak_bib = [b for b in bib_entries if not b.get("ok")]

    if not urls and not dois and not arxiv_ids and not bib_entries:
        issues.append("No references detected in the provided text.")
        suggestions.append("Paste a references section or citations block to audit.")
    if broken_urls:
        issues.append(f"{len(broken_urls)} URL(s) are unreachable or returned errors.")
        suggestions.append("Replace dead links with publisher or DOI links.")
    if bad_dois:
        issues.append(f"{len(bad_dois)} DOI(s) were not found in Crossref.")
        suggestions.append("Double-check DOI typos and include canonical DOI URLs.")
    if bad_arxiv:
        issues.append(f"{len(bad_arxiv)} arXiv ID(s) were not found.")
        suggestions.append("Verify arXiv IDs and version suffixes (e.g., v2).")
    if weak_bib:
        issues.append(f"{len(weak_bib)} BibTeX entry(ies) are missing required fields.")
        suggestions.append("Ensure each entry has title, author, year, and journal/booktitle.")

    score = 100
    score -= min(len(broken_urls) * 8, 32)
    score -= min(len(bad_dois) * 10, 30)
    score -= min(len(bad_arxiv) * 10, 20)
    score -= min(len(weak_bib) * 6, 24)
    if not urls and not dois and not arxiv_ids and not bib_entries:
        score -= 30
    score = max(0, score)

    status = "good" if score >= 85 else "warning" if score >= 65 else "critical"
    return {
        "summary": {
            "score": score,
            "status": status,
            "totals": {
                "urls": len(urls),
                "dois": len(dois),
                "arxiv_ids": len(arxiv_ids),
                "bib_entries": len(bib_entries),
            },
        },
        "url_checks": url_checks,
        "doi_checks": doi_checks,
        "arxiv_checks": arxiv_checks,
        "bibtex_checks": bib_entries,
        "issues": issues,
        "suggestions": suggestions,
    }


def format_integrity_report(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})
    totals = summary.get("totals", {})
    score = summary.get("score", 0)
    status = str(summary.get("status", "unknown")).upper()
    issues = report.get("issues") or []
    suggestions = report.get("suggestions") or []

    lines = [
        f"Citation Integrity Report: {status} (score={score}/100)",
        f"- Detected: {totals.get('urls', 0)} URL(s), {totals.get('dois', 0)} DOI(s), {totals.get('arxiv_ids', 0)} arXiv ID(s), {totals.get('bib_entries', 0)} BibTeX entry(ies)",
    ]
    if issues:
        lines.append("- Issues:")
        lines.extend([f"  - {i}" for i in issues[:8]])
    else:
        lines.append("- Issues: none detected")
    if suggestions:
        lines.append("- Recommendations:")
        lines.extend([f"  - {s}" for s in suggestions[:6]])
    return "\n".join(lines)

