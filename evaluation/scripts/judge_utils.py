from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from academic_research_mentor.citations import Citation, CitationValidator
from urllib.parse import urlparse

try:  # pragma: no cover
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore

try:  # pragma: no cover
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

try:  # pragma: no cover
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from academic_research_mentor.rich_formatter import print_error

from .judge_metrics import METRIC_SPECS, MetricSpec, metric_instruction
from .run_manual_stage import ANNOTATION_COLUMNS
from .config_loader import (
    citation_domains_digest,
    load_citation_domains,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return getattr(value, "name")
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return getattr(value, "value")
    return str(value)


def truncate_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    # Preserve trailing citations if present by keeping the tail block intact
    lower = text.lower()
    citations_marker = lower.rfind("citations")
    if citations_marker != -1:
        tail = text[citations_marker:]
        if len(tail) < limit // 2:
            available = max(limit - len(tail) - 20, 0)
            head = text[:available] + "\n...[TRUNCATED]\n"
            return head + tail
    return text[:limit] + "\n...[TRUNCATED]"


def load_tool_runs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _collect_sources_from_runs(runs: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract evidence summaries and source URLs from transparency runs.

    Returns (summaries, sources).
    """
    summaries: List[str] = []
    sources: List[str] = []
    for run in runs or []:
        events = run.get("events") or []
        for evt in events:
            payload = evt.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            event_type = evt.get("event_type")
            if event_type == "final_result":
                sm = payload.get("summary")
                if isinstance(sm, list):
                    for s in sm:
                        if isinstance(s, str) and s not in summaries:
                            summaries.append(s)
                sc = payload.get("sources")
                if isinstance(sc, list):
                    for url in sc:
                        if isinstance(url, str) and url not in sources:
                            sources.append(url)
            elif event_type == "tool_call_finished":
                result = payload.get("result") or {}
                if isinstance(result, dict):
                    findings = result.get("summary") or result.get("findings")
                    if isinstance(findings, list):
                        for f in findings:
                            if isinstance(f, str) and f not in summaries:
                                summaries.append(f)
                    result_sources = []
                    if "sources" in result and isinstance(result["sources"], list):
                        result_sources.extend(result["sources"])
                    for key in ("papers", "retrieved_guidelines", "citations"):
                        entries = result.get(key)
                        if isinstance(entries, list):
                            for entry in entries:
                                if isinstance(entry, dict):
                                    url = entry.get("url") or entry.get("link")
                                    if isinstance(url, str):
                                        result_sources.append(url)
                    for url in result_sources:
                        if isinstance(url, str) and url not in sources:
                            sources.append(url)
    return summaries, sources


def make_evidence_summary(runs: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    """Format a short evidence block for judge prompts from transparency runs."""
    summaries, sources = _collect_sources_from_runs(runs)
    lines: List[str] = []
    if summaries:
        lines.append("Top findings:")
        lines.extend(f"- {s}" for s in summaries[: max(1, min(5, limit))])
    if sources:
        lines.append("Sources:")
        lines.extend(f"- {u}" for u in sources[: max(1, min(5, limit))])
    return "\n".join(lines)


def heuristic_citation_presence(answer_text: str) -> float:
    """Binary: 1 if inline citations or a 'Citations' section present."""
    text = (answer_text or "").lower()
    if not text:
        return 0.0
    # Inline patterns: [P1], [G2], [1], etc.
    if re.search(r"\[(?:p\d+|g\d+|\d+)\]", text):
        return 1.0
    if "citations" in text and ("http://" in text or "https://" in text or "arxiv" in text or "doi" in text):
        return 1.0
    return 0.0


_citation_validator = CitationValidator()


_CITATION_LINE_PATTERN = re.compile(
    r"\[(?P<id>[A-Za-z0-9#]+)\]\s*(?P<title>[^\n]+?)\s*[\-–—]\s*(?P<url>https?://\S+)",
    re.IGNORECASE,
)


def _normalize_domain(netloc: str) -> str:
    value = (netloc or "").lower().strip()
    if value.startswith("www."):
        value = value[4:]
    return value


def _classify_domain(domain: str) -> str:
    config = load_citation_domains()
    for kind, domains in config.items():
        for candidate in domains:
            candidate = candidate.strip().lower()
            if not candidate:
                continue
            if domain == candidate or domain.endswith(f".{candidate}"):
                return kind
    return "other"


def extract_citations_from_answer(answer_text: str) -> List[Citation]:
    """Parse the final answer for a Citations section and return Citation objects."""
    citations: List[Citation] = []
    if not answer_text:
        return citations
    for match in _CITATION_LINE_PATTERN.finditer(answer_text):
        stable_id = match.group("id") or "source"
        title = match.group("title") or "Untitled"
        url = match.group("url")
        source = urlparse(url).netloc if url else "unknown"
        citations.append(
            Citation(
                id=stable_id.strip(),
                title=title.strip(),
                url=url.strip(),
                source=source.strip(),
            )
        )
    if citations:
        return citations

    # Fallback: look for any URLs in a citations section (lines after "Citations")
    section = answer_text
    lower = answer_text.lower()
    marker_index = lower.find("citations")
    if marker_index != -1:
        section = answer_text[marker_index:]

    url_pattern = re.compile(r"https?://\S+")
    seen_urls: set[str] = set()
    for url_match in url_pattern.finditer(section):
        url = url_match.group(0)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        # Grab the line containing the URL to attempt a title extraction
        start = section.rfind("\n", 0, url_match.start()) + 1
        end = section.find("\n", url_match.end())
        if end == -1:
            end = len(section)
        line = section[start:end].strip()
        title_part = line.replace(url, "").strip(" -–—")
        if not title_part:
            title_part = "Source"
        citations.append(
            Citation(
                id=f"url{len(citations)+1}",
                title=title_part.strip(),
                url=url.strip(),
                source=urlparse(url).netloc,
            )
        )
    return citations


def _serialize_citation(citation: Citation, kind: str, domain: str, malformed: bool) -> Dict[str, Any]:
    return {
        "id": citation.id,
        "title": citation.title,
        "url": citation.url,
        "domain": domain,
        "kind": kind,
        "malformed": malformed,
    }


def score_citation_validity_v2(citations: Sequence[Citation]) -> Dict[str, Any]:
    if not citations:
        return {
            "score": 0.0,
            "details": {
                "total_count": 0,
                "scholarly_count": 0,
                "guideline_count": 0,
                "portal_count": 0,
                "other_count": 0,
                "malformed_count": 0,
            },
            "citations": [],
        }

    config_digest = citation_domains_digest()
    typed: List[Dict[str, Any]] = []
    counts = {
        "scholarly": 0,
        "guideline": 0,
        "portal": 0,
        "other": 0,
    }
    malformed_count = 0

    for citation in citations:
        url = citation.url or ""
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            malformed_count += 1
            typed.append(_serialize_citation(citation, "malformed", "", True))
            continue
        domain = _normalize_domain(parsed.netloc)
        kind = _classify_domain(domain)
        if kind in counts:
            counts[kind] += 1
        else:
            counts["other"] += 1
        typed.append(_serialize_citation(citation, kind, domain, False))

    total_count = len(citations)
    valid_links = total_count - malformed_count
    has_scholarly = counts["scholarly"] > 0
    score = 0.0
    if has_scholarly:
        score = 1.0
    elif valid_links >= 2 and malformed_count == 0:
        score = 1.0

    return {
        "score": score,
        "details": {
            "total_count": total_count,
            "scholarly_count": counts["scholarly"],
            "guideline_count": counts["guideline"],
            "portal_count": counts["portal"],
            "other_count": counts["other"],
            "malformed_count": malformed_count,
            "domain_config_digest": config_digest,
        },
        "citations": typed,
    }


def score_citation_validity(answer_text: str) -> Dict[str, Any]:
    citations = extract_citations_from_answer(answer_text)
    result = score_citation_validity_v2(citations)
    legacy: Dict[str, Any] = {}
    if citations:
        try:
            legacy = dict(_citation_validator.validate_citations(citations))
        except Exception as exc:  # pragma: no cover - validator fallback
            legacy = {"error": str(exc)}
    if legacy:
        quality_score = legacy.pop("score", None)
        if quality_score is not None:
            result["legacy_quality_score"] = quality_score
        result["legacy_validator"] = legacy
    return result


def heuristic_fallback_robustness(runs: Sequence[Dict[str, Any]]) -> float:
    """Binary: 1 if any error occurred followed by a success (possibly with a different tool),
    or success under degraded/backoff conditions.
    """
    had_error = False
    success_after = False
    tool_names: List[str] = []
    for run in runs or []:
        name = run.get("tool_name")
        if name:
            tool_names.append(name)
        status = (run.get("status") or "").lower()
        if status == "failure":
            had_error = True
        if status == "success" and had_error:
            success_after = True
        meta = run.get("metadata") or {}
        if str(meta.get("tool_state", "")).lower() == "degraded" and status == "success":
            return 1.0
        if (meta.get("backoff_count") or 0) > 0 and status == "success":
            return 1.0
    if success_after and len(set(tool_names)) >= 2:
        return 1.0
    return 0.0


def heuristic_asks_questions(answer_text: str) -> float:
    """Binary: 1 if the agent asks at least one targeted question."""
    text = (answer_text or "").strip()
    if not text:
        return 0.0
    qm = text.count("?")
    if qm == 0:
        return 0.0
    import re
    if re.search(r"\b(what|how|which|could|would|can|are you|do you|have you|might you)\b", text.lower()):
        return 1.0
    return 0.0


def apply_evidence_integrity(metric_scores: Dict[str, Optional[float]], metric_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    validity_score = metric_scores.get("citation_validity")
    rag_score = metric_scores.get("rag_fidelity")

    if validity_score is None:
        return {
            "score": None,
            "details": {
                "reason": "no_citation_validity_metric",
            },
        }

    if validity_score == 0:
        for key in ("citation_relevance", "citation_quality"):
            if key in metric_results:
                metric_results[key]["score"] = 0.0
            metric_scores[key] = 0.0
        return {
            "score": 0.0,
            "details": {
                "reason": "invalid_citations",
            },
        }

    evidence_score = float(validity_score)
    if rag_score is not None:
        evidence_score = float(min(evidence_score, rag_score))

    return {
        "score": evidence_score,
        "details": {
            "validity": validity_score,
            "rag_fidelity": rag_score,
        },
    }

def aggregate_tool_routing(expected: Sequence[str], runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    expected_set = [t.strip() for t in (expected or []) if t]
    observed = {str(run.get("tool_name")) for run in runs if run.get("tool_name")}
    missing = [tool for tool in expected_set if tool not in observed]
    extra = [tool for tool in observed if expected_set and tool not in expected_set]
    score = 1.0 if not missing else 0.0
    return {
        "score": score,
        "details": {
            "expected": expected_set,
            "observed": sorted(observed),
            "missing": missing,
            "extra": extra,
        },
    }


def build_judge_clients(specs: Sequence[str]) -> List[Tuple[str, Any]]:
    clients: List[Tuple[str, Any]] = []
    for raw in specs:
        provider, _, model = raw.partition(":")
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Invalid judge spec: {raw}")
        if provider == "anthropic":
            if ChatAnthropic is None:
                raise RuntimeError("langchain-anthropic not available")
            client = ChatAnthropic(model=model, temperature=0.0, max_tokens=1536)
        elif provider in {"google", "gemini"}:
            if ChatGoogleGenerativeAI is None:
                raise RuntimeError("langchain-google-genai not available")
            client = ChatGoogleGenerativeAI(model=model, temperature=0.0, max_output_tokens=1536)
        elif provider in {"openai", "azure"}:
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai not available")
            client = ChatOpenAI(model=model, temperature=0.0, max_tokens=1536)
        elif provider == "openrouter":
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai not available")
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY must be set for openrouter judges")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            headers: Dict[str, str] = {}
            referer = os.environ.get("OPENROUTER_HTTP_REFERER")
            title = os.environ.get("OPENROUTER_TITLE")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            client_kwargs: Dict[str, Any] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": 0.0,
                "max_tokens": 1536,
            }
            if headers:
                client_kwargs["default_headers"] = headers
            client = ChatOpenAI(**client_kwargs)
        else:
            raise ValueError(f"Unsupported provider '{provider}' in {raw}")
        clients.append((raw, client))
    return clients


def call_judge(client: Any, spec: MetricSpec, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    system_prompt = (
        "You are an evaluation assistant scoring AI mentor responses according to a rubric. "
        "Be strict, cite rubric criteria, and output only JSON."
    )
    sections: List[str] = [
        f"Metric: {spec.key}",
        f"Rubric: {spec.description}",
        metric_instruction(spec),
        "",
        "### User Prompt",
        context["user_prompt"],
        "",
        "### Agent Response",
        context["agent_response"],
        "",
    ]

    evidence_block = (context.get("evidence") or "").strip()
    if evidence_block:
        sections.extend(["### Evidence Summary", evidence_block, ""])

    citations_block = (context.get("citations") or "").strip()
    if citations_block:
        sections.extend(["### Extracted Citations", citations_block, ""])

    sections.extend(
        [
            "### Metadata",
            json.dumps(context.get("metadata", {}), ensure_ascii=False, indent=2),
            "",
            "### Tool Runs (trimmed)",
            context["tool_runs"],
        ]
    )

    user_prompt = "\n".join(sections)

    result = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
    meta: Dict[str, Any] = {}
    try:
        resp_meta = getattr(result, "response_metadata", None)
        addl = getattr(result, "additional_kwargs", None)
        usage = getattr(result, "usage_metadata", None)
        finish_reason = None
        if isinstance(resp_meta, dict):
            finish_reason = resp_meta.get("finish_reason") or resp_meta.get("finish_reasons") or resp_meta.get("finish_reason_message")
            meta["response_metadata"] = resp_meta
        if not finish_reason and isinstance(addl, dict):
            finish_reason = addl.get("finish_reason") or addl.get("finish_details") or addl.get("finish_reason_message")
            meta["additional_kwargs"] = addl
        if usage is not None:
            meta["usage"] = usage
        if finish_reason:
            meta["finish_reason"] = finish_reason
    except Exception:  # pragma: no cover - metadata is best-effort
        pass
    return text, meta


def parse_score(raw: str) -> Optional[Dict[str, Any]]:
    """Parse judge output into a dict with at least a numeric 'score' when possible.

    Robust to code fences and partially truncated JSON by using a regex fallback
    to extract a numeric score when full JSON parsing fails.
    """
    candidate = (raw or "").strip()
    if not candidate:
        return None
    # Strip code fences if present
    if candidate.startswith("```"):
        try:
            candidate = candidate.split("\n", 1)[1]
            candidate = candidate.strip("`\n ")
        except Exception:
            pass
    # Try strict JSON first
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        # Fallback: extract minimal fields via regex (handles truncated JSON)
        import re
        out: Dict[str, Any] = {}
        # score: accept numeric or quoted numeric
        m = re.search(r'"score"\s*:\s*("(?P<qs>[0-9]+(?:\.[0-9]+)?)"|(?P<ns>[0-9]+(?:\.[0-9]+)?))', candidate)
        if m:
            sval = m.group("qs") or m.group("ns")
            try:
                out["score"] = float(sval)
            except Exception:
                pass
        # rationale (best-effort, short capture up to next quote)
        m = re.search(r'"rationale"\s*:\s*"(?P<rt>[^"\\]{0,800})', candidate)
        if m:
            out["rationale"] = m.group("rt")
        # confidence (simple token)
        m = re.search(r'"confidence"\s*:\s*"(?P<cf>high|medium|low)"', candidate, re.IGNORECASE)
        if m:
            out["confidence"] = m.group("cf")
        return out if out else None


def aggregate_scores(spec: MetricSpec, judge_results: Sequence[Dict[str, Any]]) -> Optional[float]:
    values: List[float] = []
    for record in judge_results:
        score = record.get("score")
        if isinstance(score, (int, float)):
            values.append(float(score))
    if not values:
        return None
    avg = sum(values) / len(values)
    if spec.kind == "binary":
        return 1.0 if avg >= 0.5 else 0.0
    return max(spec.min_score, min(spec.max_score, avg))


def load_annotation_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    return rows, headers


def ensure_headers(headers: list[str]) -> list[str]:
    if headers:
        return headers
    return list(ANNOTATION_COLUMNS)


def write_annotation_rows(path: Path, rows: list[dict[str, Any]], headers: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def format_score(spec: MetricSpec, value: Optional[float]) -> str:
    if value is None:
        return ""
    if spec.kind == "binary":
        return str(int(value))
    return f"{value:.2f}"


def resolve_metric_column(columns: Iterable[str], key: str) -> Optional[str]:
    column_set = set(columns)
    if key in column_set:
        return key
    alt = f"{key}_score"
    if alt in column_set:
        return alt
    if key.endswith("_score") and key[:-6] in column_set:
        return key[:-6]
    return None


def resolve_metric_spec(key: str) -> Optional[MetricSpec]:
    if key in METRIC_SPECS:
        return METRIC_SPECS[key]
    if key.endswith("_score"):
        base = key[:-6]
        return METRIC_SPECS.get(base)
    return METRIC_SPECS.get(key)


def upsert_annotation(
    path: Path,
    prompt_id: str,
    stage: str,
    annotator: str,
    system_id: Optional[str],
    metric_scores: Dict[str, Optional[float]],
    run_timestamp: str,
    response_path: str,
    tool_trace_path: str,
    force: bool,
) -> None:
    rows, headers = load_annotation_rows(path)
    headers = ensure_headers(headers)

    updated = False
    for row in rows:
        if row.get("prompt_id") != prompt_id:
            continue
        existing = row.get("annotator", "")
        if existing and existing != annotator and not force:
            continue
        row["annotator"] = annotator
        row["run_timestamp"] = run_timestamp
        if "system_id" in row:
            row["system_id"] = system_id or row.get("system_id") or "unknown"
        row["response_path"] = response_path
        row["tool_trace_path"] = tool_trace_path
        if "stage" in row:
            row["stage"] = stage
        for key, value in metric_scores.items():
            column = resolve_metric_column(row.keys(), key)
            if column is None:
                continue
            spec = resolve_metric_spec(key)
            if value is None:
                continue
            row[column] = format_score(spec, value) if spec else str(value)
        updated = True
        break

    if not updated:
        new_row = {h: "" for h in headers}
        new_row.update(
            {
                "prompt_id": prompt_id,
                "stage": stage,
                "annotator": annotator,
                "system_id": system_id or "unknown",
                "run_timestamp": run_timestamp,
                "response_path": response_path,
                "tool_trace_path": tool_trace_path,
            }
        )
        for key, value in metric_scores.items():
            column = resolve_metric_column(new_row.keys(), key)
            if column is None:
                continue
            spec = resolve_metric_spec(key)
            if value is None:
                continue
            new_row[column] = format_score(spec, value) if spec else str(value)
        rows.append(new_row)

    write_annotation_rows(path, rows, headers)


def build_context(
    meta: dict[str, Any],
    response: str,
    tool_runs: str,
    raw_runs: Optional[Sequence[Dict[str, Any]]] = None,
    full_response: Optional[str] = None,
) -> dict[str, Any]:
    evidence = make_evidence_summary(raw_runs or [])
    response_for_citations = full_response or response
    extracted = extract_citations_from_answer(response_for_citations)
    citation_lines: List[str] = []
    for citation in extracted:
        url = citation.url or ""
        domain = _normalize_domain(urlparse(url).netloc) if url else ""
        kind = _classify_domain(domain) if url else "other"
        parts = [f"[{citation.id}] {citation.title}".strip()]
        if url:
            parts[-1] += f" — {url}"
        if kind != "other" and url:
            parts[-1] += f" (kind: {kind})"
        citation_lines.append(parts[-1])

    return {
        "user_prompt": meta.get("prompt", ""),
        "agent_response": response,
        "metadata": dict(meta.get("metadata") or {}),
        "tool_runs": tool_runs,
        "evidence": evidence,
        "citations": "\n".join(citation_lines),
    }


def save_judge_payload(path: Path, payload: dict[str, Any]) -> None:
    safe_payload = _json_safe(payload)
    path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
