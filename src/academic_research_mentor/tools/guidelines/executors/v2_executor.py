from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..cache import GuidelinesCache
from ..evidence_collector import EvidenceCollector
from ..formatter import GuidelinesFormatter
from ..search_providers import BaseSearchProvider
from ..citation_handler import GuidelinesCitationHandler


class GuidelinesV2Executor:
    """Handle guidelines execution when FF_GUIDELINES_V2 is enabled."""

    def __init__(
        self,
        evidence_collector: EvidenceCollector,
        formatter: GuidelinesFormatter,
        citation_handler: GuidelinesCitationHandler,
        cache: Optional[GuidelinesCache],
        search_tool: Optional[BaseSearchProvider],
    ) -> None:
        self._evidence_collector = evidence_collector
        self._formatter = formatter
        self._citation_handler = citation_handler
        self._cache = cache
        self._search_tool = search_tool

    def run(
        self,
        topic: str,
        mode: str,
        max_per_source: int,
        response_format: str,
        page_size: int,
        next_token: Optional[str],
        cache_key: str,
    ) -> Dict[str, Any]:
        curated = self._evidence_collector.collect_curated_evidence(topic)
        evidence: List[Dict[str, Any]] = list(curated)
        sources_covered = sorted(
            {
                entry.get("domain", "")
                for entry in curated
                if entry.get("domain")
            }
        )

        if self._search_tool:
            searched, covered = self._evidence_collector.collect_structured_evidence(
                topic, mode, max_per_source
            )
            evidence.extend(searched)
            for domain in covered:
                if domain and domain not in sources_covered:
                    sources_covered.append(domain)

        if max_per_source > 0:
            capped: List[Dict[str, Any]] = []
            counts: Dict[str, int] = {}
            canonical: Dict[str, str] = {}
            for entry in evidence:
                domain_raw = str(entry.get("domain") or "").strip()
                key = domain_raw.lower()
                if key:
                    canonical.setdefault(key, domain_raw)
                    if counts.get(key, 0) >= max_per_source:
                        continue
                    counts[key] = counts.get(key, 0) + 1
                capped.append(entry)
            evidence = capped
            if canonical:
                sources_covered = sorted({canonical[k] for k, count in counts.items() if count > 0})

        if not evidence:
            result: Dict[str, Any] = {
                "topic": topic,
                "total_evidence": 0,
                "sources_covered": sources_covered,
                "evidence": [],
                "pagination": {"has_more": False, "next_token": None},
                "cached": False,
                "note": "No evidence found",
            }
            if self._cache:
                self._cache.set(cache_key, result)
            return result

        result = self._formatter.format_v2_response(
            topic, evidence, sources_covered, response_format, page_size, next_token
        )

        if max_per_source > 0:
            capped_evidence: List[Dict[str, Any]] = []
            final_counts: Dict[str, int] = {}
            for item in result.get("evidence", []):
                domain_raw = str(item.get("domain") or "").strip()
                key = domain_raw.lower()
                if key and final_counts.get(key, 0) >= max_per_source:
                    continue
                if key:
                    final_counts[key] = final_counts.get(key, 0) + 1
                capped_evidence.append(item)

            if capped_evidence != result.get("evidence", []):
                result["evidence"] = capped_evidence
                result["total_evidence"] = len(capped_evidence)
                pagination = result.get("pagination", {})
                pagination["has_more"] = False
                pagination["next_token"] = None
                result["pagination"] = pagination
                normalized_sources = []
                seen_sources = set()
                for entry in capped_evidence:
                    domain_val = str(entry.get("domain") or "").strip()
                    key = domain_val.lower()
                    if key and key not in seen_sources:
                        seen_sources.add(key)
                        normalized_sources.append(domain_val)
                if normalized_sources:
                    result["sources_covered"] = sorted(normalized_sources, key=str.lower)

        result = self._citation_handler.add_citation_metadata(result, evidence)

        if self._cache:
            self._cache.set(cache_key, result)
        return result
