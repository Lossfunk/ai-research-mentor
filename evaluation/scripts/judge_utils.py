from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

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


def build_context(meta: dict[str, Any], response: str, tool_runs: str) -> dict[str, Any]:
    return {
        "user_prompt": meta.get("prompt", ""),
        "agent_response": response,
        "metadata": dict(meta.get("metadata") or {}),
        "tool_runs": tool_runs,
    }


def save_judge_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
