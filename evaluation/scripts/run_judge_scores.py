from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from academic_research_mentor.rich_formatter import print_error, print_info
from academic_research_mentor.cli.session import load_env_file

from .config_loader import (
    citation_domains_digest,
    compute_file_digest,
    load_metrics_config,
    metrics_config_digest,
)
from .judge_metrics import METRIC_SPECS, MetricSpec
from .judge_utils import (
    aggregate_scores,
    aggregate_tool_routing,
    apply_evidence_integrity,
    build_context,
    build_judge_clients,
    call_judge,
    heuristic_asks_questions,
    heuristic_citation_presence,
    heuristic_fallback_robustness,
    iso_timestamp,
    load_tool_runs,
    parse_score,
    save_judge_payload,
    score_citation_validity,
    truncate_text,
    upsert_annotation,
)
from .run_manual_stage import ensure_stage_directories, normalize_stage


ABSOLUTE_PROMPT_PATH = Path("evaluation/judges/single_turn_absolute_prompt.md")


def metric_prompt_digest(spec: MetricSpec) -> str:
    payload = json.dumps(
        {
            "key": spec.key,
            "description": spec.description,
            "kind": spec.kind,
            "min_score": spec.min_score,
            "max_score": spec.max_score,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_model_param(client: Any, attr_names: Sequence[str]) -> Optional[Any]:
    for name in attr_names:
        if not hasattr(client, name):
            continue
        value = getattr(client, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
    return None


def evaluate_metric(
    spec: MetricSpec,
    context: Dict[str, Any],
    judge_clients: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    judge_outputs: List[Dict[str, Any]] = []
    for name, client in judge_clients:
        try:
            raw = call_judge(client, spec, context)
            parsed = parse_score(raw) or {}
            entry: Dict[str, Any] = {
                "judge": name,
                "raw": raw,
                "rationale": parsed.get("rationale"),
                "confidence": parsed.get("confidence"),
            }
            score = parsed.get("score")
            if isinstance(score, str):
                try:
                    score = float(score.strip())
                except ValueError:
                    try:
                        score = int(score.strip())
                    except ValueError:
                        score = None
            if isinstance(score, (int, float)):
                entry["score"] = float(score)
            else:
                entry["error"] = "missing_score"
            judge_outputs.append(entry)
        except Exception as exc:  # noqa: BLE001
            judge_outputs.append(
                {
                    "judge": name,
                    "score": None,
                    "rationale": None,
                    "confidence": None,
                    "error": str(exc),
                }
            )

    aggregated = aggregate_scores(spec, judge_outputs)
    return {"score": aggregated, "judges": judge_outputs}


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return cleaned or "default"


def _derive_label(judge_specs: Sequence[str], provided: Optional[str]) -> str:
    if provided:
        return _sanitize_label(provided)
    combined = "__".join(spec.replace("/", "-") for spec in judge_specs)
    return _sanitize_label(combined or "default")


def run_judges(
    stage: str,
    prompt_ids: Optional[Sequence[str]],
    judge_specs: Sequence[str],
    annotator: str,
    force: bool,
    output_label: Optional[str],
    system_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if not judge_specs:
        raise ValueError("At least one --judge is required")

    judge_clients = build_judge_clients(judge_specs)
    judge_metadata = []
    for spec, client in judge_clients:
        judge_metadata.append(
            {
                "spec": spec,
                "temperature": _safe_model_param(client, ("temperature",)),
                "max_tokens": _safe_model_param(client, ("max_tokens", "max_output_tokens")),
            }
        )
    stage_letter, stage_folder = normalize_stage(stage)
    _, analysis_dir, _ = ensure_stage_directories(stage_folder)
    label = _derive_label(judge_specs, output_label)
    output_dir = analysis_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder_csv = output_dir / "annotation_placeholders.csv"

    metrics_config = load_metrics_config()
    metrics_version = metrics_config.get("version")
    metrics_digest = metrics_config_digest()
    absolute_prompt_digest = compute_file_digest(ABSOLUTE_PROMPT_PATH)
    domain_config_digest = citation_domains_digest()

    meta_files = sorted(analysis_dir.glob("*_meta.json"))
    prompt_filter = set(prompt_ids) if prompt_ids else None

    summaries: List[Dict[str, Any]] = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt_id = meta.get("prompt_id")
        if not prompt_id:
            continue
        if prompt_filter and prompt_id not in prompt_filter:
            continue
        meta_system_id = meta.get("system_id") or meta.get("system")
        if system_filter and meta_system_id != system_filter:
            continue

        response_path = Path(meta.get("response_path", ""))
        tool_path = Path(meta.get("tool_trace_path", ""))
        if not response_path.exists():
            print_error(f"Missing response file for {prompt_id}: {response_path}")
            continue

        full_response_text = response_path.read_text(encoding="utf-8")
        response_text = truncate_text(full_response_text)
        tool_runs = load_tool_runs(tool_path)
        tool_runs_str = truncate_text(json.dumps(tool_runs, ensure_ascii=False, indent=2))

        expected_checks = list(meta.get("expected_checks") or [])
        metadata = dict(meta.get("metadata") or {})
        context = build_context(
            meta,
            response_text,
            tool_runs_str,
            raw_runs=tool_runs,
            full_response=full_response_text,
        )

        metric_results: Dict[str, Dict[str, Any]] = {}
        metric_scores: Dict[str, Optional[float]] = {}
        metric_digests: Dict[str, Optional[str]] = {}

        if "tool_routing" in expected_checks:
            routing = aggregate_tool_routing(metadata.get("expected_tools", []), tool_runs)
            metric_results["tool_routing"] = routing
            metric_scores["tool_routing"] = routing.get("score")
            metric_digests["tool_routing"] = None

        if "citation_presence" in expected_checks:
            val = heuristic_citation_presence(full_response_text)
            metric_results["citation_presence"] = {"score": val}
            metric_scores["citation_presence"] = val
            metric_digests["citation_presence"] = None

        if "citation_validity" in expected_checks:
            validity = score_citation_validity(full_response_text)
            metric_results["citation_validity"] = validity
            metric_scores["citation_validity"] = validity.get("score")
            metric_digests["citation_validity"] = None

        if "fallback_robustness" in expected_checks:
            val = heuristic_fallback_robustness(tool_runs)
            metric_results["fallback_robustness"] = {"score": val}
            metric_scores["fallback_robustness"] = val
            metric_digests["fallback_robustness"] = None

        if "asks_questions" in expected_checks:
            val = heuristic_asks_questions(full_response_text)
            metric_results["asks_questions"] = {"score": val}
            metric_scores["asks_questions"] = val
            metric_digests["asks_questions"] = None

        for metric_key in expected_checks:
            if metric_key == "tool_routing":
                continue
            if metric_key in {"citation_presence", "citation_validity", "fallback_robustness", "asks_questions"}:
                # Already scored by heuristic above
                continue
            spec = METRIC_SPECS.get(metric_key)
            if not spec:
                print_error(f"Metric '{metric_key}' not defined; skipping for {prompt_id}")
                continue
            result = evaluate_metric(spec, context, judge_clients)
            metric_results[metric_key] = result
            metric_scores[metric_key] = result.get("score")
            metric_digests[metric_key] = metric_prompt_digest(spec)

        if metric_scores.get("citation_validity") is not None:
            evidence_metric = apply_evidence_integrity(metric_scores, metric_results)
            metric_results["evidence_integrity"] = evidence_metric
            metric_scores["evidence_integrity"] = evidence_metric.get("score")
            metric_digests["evidence_integrity"] = None

        timestamp = iso_timestamp()
        upsert_annotation(
            placeholder_csv,
            prompt_id,
            stage_letter,
            annotator,
            meta.get("system_id"),
            metric_scores,
            timestamp,
            str(response_path),
            str(tool_path),
            force=force,
        )

        payload = {
            "prompt_id": prompt_id,
            "stage": stage_letter,
            "generated_at": timestamp,
            "metrics": metric_results,
            "judge_models": [name for name, _ in judge_clients],
            "output_label": label,
            "metrics_version": metrics_version,
            "metrics_config_digest": metrics_digest,
            "judge_prompt_digest": absolute_prompt_digest,
             "citation_domains_digest": domain_config_digest,
            "metric_prompt_digests": {k: v for k, v in metric_digests.items() if v},
            "model_params": meta.get("model_params"),
            "model_spec": {
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "system_id": meta.get("system_id"),
                "system_alias": meta.get("system_alias"),
            },
            "expected_checks": expected_checks,
            "judge_parameters": judge_metadata,
        }
        save_judge_payload(output_dir / f"{prompt_id}_judges.json", payload)
        summaries.append(payload)
        print_info(f"Scored {prompt_id} [{label}]: {sorted(metric_results.keys())}")

    return {
        "stage": stage_letter,
        "processed": len(summaries),
        "judged_prompts": [data["prompt_id"] for data in summaries],
        "output_label": label,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judges over evaluation artifacts")
    parser.add_argument("--stage", default="stage_a", help="Stage to score (stage_a, stage_b, stage_c)")
    parser.add_argument(
        "--prompt-id",
        dest="prompt_ids",
        action="append",
        help="Limit scoring to a specific prompt id (repeatable)",
    )
    parser.add_argument(
        "--judge",
        dest="judges",
        action="append",
        required=True,
        help="Judge spec provider:model (repeat for multiple)",
    )
    parser.add_argument("--annotator", required=True, help="Name recorded in annotation CSV")
    parser.add_argument(
        "--label",
        help="Optional label for organizing outputs under evals-for-papers/results/analysis_reports/<stage>/<label>",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotator rows")
    parser.add_argument("--system", dest="system_filter", help="Optional system identifier to filter meta files")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    args = parse_args(argv)
    try:
        summary = run_judges(
            stage=args.stage,
            prompt_ids=args.prompt_ids,
            judge_specs=args.judges,
            annotator=args.annotator,
            force=args.force,
            output_label=args.label,
            system_filter=args.system_filter,
        )
    except Exception as exc:  # noqa: BLE001
        print_error(f"Judge run failed: {exc}")
        return 1

    print_info(
        f"Completed judge run for {summary['stage']} ({summary['output_label']}) — processed {summary['processed']} prompt(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
