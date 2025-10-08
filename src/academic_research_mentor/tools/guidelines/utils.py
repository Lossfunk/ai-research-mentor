from __future__ import annotations

import re
from typing import Any, Dict


_GUIDELINES_PATTERNS = (
    r"\b(research\s+methodology|problem\s+selection|research\s+taste)\b",
    r"\b(academic\s+advice|phd\s+guidance|research\s+strategy)\b",
    r"\b(how\s+to\s+choose|develop\s+taste|research\s+skills)\b",
    r"\b(research\s+best\s+practices|methodology\s+advice)\b",
    r"\b(academic\s+career|research\s+planning|project\s+selection)\b",
    r"\b(hamming|effective\s+research|research\s+principles)\b",
    r"\bphd\b|\bcareer\s+guidance\b|\bmentoring\b|\bacademic\s+guidance\b",
    r"\bresearch\s+advice\b|\bgraduate\s+school\b|\bacademic\s+career\b",
)


def matches_guidelines_query(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _GUIDELINES_PATTERNS)


def enforce_domain_cap(result: Dict[str, Any], max_per_source: int) -> Dict[str, Any]:
    if max_per_source <= 0 or not isinstance(result, dict):
        return result
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return result

    filtered: list[Dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        domain_raw = str(item.get("domain") or "").strip()
        key = domain_raw.lower()
        if key and counts.get(key, 0) >= max_per_source:
            continue
        if key:
            counts[key] = counts.get(key, 0) + 1
        filtered.append(item)

    if len(filtered) == len(evidence):
        return result

    result["evidence"] = filtered
    result["total_evidence"] = len(filtered)

    normalized_sources: list[str] = []
    seen_sources: set[str] = set()
    for entry in filtered:
        domain_val = str(entry.get("domain") or "").strip()
        key = domain_val.lower()
        if key and key not in seen_sources:
            seen_sources.add(key)
            normalized_sources.append(domain_val)
    if normalized_sources:
        result["sources_covered"] = sorted(normalized_sources, key=str.lower)

    pagination = result.get("pagination")
    if isinstance(pagination, dict):
        pagination["has_more"] = False
        pagination["next_token"] = None
        result["pagination"] = pagination

    return result
