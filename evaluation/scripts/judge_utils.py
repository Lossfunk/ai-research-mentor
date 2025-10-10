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


def truncate_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
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
        # Events may contain a final_result payload with 'summary' and 'sources'
        events = run.get("events") or []
        for evt in events:
            payload = evt.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            if evt.get("event_type") == "final_result":
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
    return "\n".join(lines) if lines else "(no evidence summary available)"


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


def score_citation_validity(answer_text: str) -> Dict[str, Any]:
    """Use CitationValidator to score validity of cited sources.

    Returns a dict containing score (1 valid / 0 otherwise) and validator details.
    """
    citations = extract_citations_from_answer(answer_text)
    if not citations:
        return {
            "score": 0.0,
            "valid": False,
            "total_count": 0,
            "issues": ["no_citations_found"],
        }
    validation = _citation_validator.validate_citations(citations)
    validity_score = 1.0 if validation.get("valid") else 0.0
    # Do not overwrite the metric 'score' with the validator's numeric quality score.
    # Expose it separately as 'quality_score'.
    out = dict(validation)
    if "score" in out:
        out["quality_score"] = out.pop("score")
    out["score"] = validity_score
    return out


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
            client = ChatAnthropic(model=model, temperature=0.0, max_tokens=1024)
        elif provider in {"google", "gemini"}:
            if ChatGoogleGenerativeAI is None:
                raise RuntimeError("langchain-google-genai not available")
            client = ChatGoogleGenerativeAI(model=model, temperature=0.0, max_output_tokens=1024)
        elif provider in {"openai", "azure"}:
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai not available")
            client = ChatOpenAI(model=model, temperature=0.0, max_tokens=1024)
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
                "max_tokens": 1024,
            }
            if headers:
                client_kwargs["default_headers"] = headers
            client = ChatOpenAI(**client_kwargs)
        else:
            raise ValueError(f"Unsupported provider '{provider}' in {raw}")
        clients.append((raw, client))
    return clients


def call_judge(client: Any, spec: MetricSpec, context: Dict[str, Any]) -> str:
    system_prompt = (
        "You are an evaluation assistant scoring AI mentor responses according to a rubric. "
        "Be strict, cite rubric criteria, and output only JSON."
    )
    user_prompt = (
        f"Metric: {spec.key}\n"
        f"Rubric: {spec.description}\n"
        f"{metric_instruction(spec)}\n\n"
        "### User Prompt\n"
        f"{context['user_prompt']}\n\n"
        "### Agent Response\n"
        f"{context['agent_response']}\n\n"
        "### Evidence Summary\n"
        f"{context.get('evidence', '(none)')}\n\n"
        "### Metadata\n"
        f"{json.dumps(context.get('metadata', {}), ensure_ascii=False, indent=2)}\n\n"
        "### Tool Runs (trimmed)\n"
        f"{context['tool_runs']}"
    )

    result = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return getattr(result, "content", None) or getattr(result, "text", None) or str(result)


def parse_score(raw: str) -> Optional[Dict[str, Any]]:
    try:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1]
            candidate = candidate.strip("`\n ")
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


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


def build_context(meta: dict[str, Any], response: str, tool_runs: str, raw_runs: Optional[Sequence[Dict[str, Any]]] = None) -> dict[str, Any]:
    evidence = make_evidence_summary(raw_runs or [])
    return {
        "user_prompt": meta.get("prompt", ""),
        "agent_response": response,
        "metadata": dict(meta.get("metadata") or {}),
        "tool_runs": tool_runs,
        "evidence": evidence,
    }


def save_judge_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
