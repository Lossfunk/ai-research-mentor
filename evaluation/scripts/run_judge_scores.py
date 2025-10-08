from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from academic_research_mentor.rich_formatter import print_error, print_info
from academic_research_mentor.cli.session import load_env_file

from .judge_metrics import METRIC_SPECS, MetricSpec
from .judge_utils import (
    aggregate_scores,
    aggregate_tool_routing,
    build_context,
    build_judge_clients,
    call_judge,
    iso_timestamp,
    load_tool_runs,
    parse_score,
    save_judge_payload,
    truncate_text,
    upsert_annotation,
)
from .run_manual_stage import ensure_stage_directories, normalize_stage


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
) -> Dict[str, Any]:
    if not judge_specs:
        raise ValueError("At least one --judge is required")

    judge_clients = build_judge_clients(judge_specs)
    stage_letter, stage_folder = normalize_stage(stage)
    _, analysis_dir, _ = ensure_stage_directories(stage_folder)
    label = _derive_label(judge_specs, output_label)
    output_dir = analysis_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder_csv = output_dir / "annotation_placeholders.csv"

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

        response_path = Path(meta.get("response_path", ""))
        tool_path = Path(meta.get("tool_trace_path", ""))
        if not response_path.exists():
            print_error(f"Missing response file for {prompt_id}: {response_path}")
            continue

        response_text = truncate_text(response_path.read_text(encoding="utf-8"))
        tool_runs = load_tool_runs(tool_path)
        tool_runs_str = truncate_text(json.dumps(tool_runs, ensure_ascii=False, indent=2))

        expected_checks = list(meta.get("expected_checks") or [])
        metadata = dict(meta.get("metadata") or {})
        context = build_context(meta, response_text, tool_runs_str)

        metric_results: Dict[str, Dict[str, Any]] = {}
        metric_scores: Dict[str, Optional[float]] = {}

        if "tool_routing" in expected_checks:
            routing = aggregate_tool_routing(metadata.get("expected_tools", []), tool_runs)
            metric_results["tool_routing"] = routing
            metric_scores["tool_routing"] = routing.get("score")

        for metric_key in expected_checks:
            if metric_key == "tool_routing":
                continue
            spec = METRIC_SPECS.get(metric_key)
            if not spec:
                print_error(f"Metric '{metric_key}' not defined; skipping for {prompt_id}")
                continue
            result = evaluate_metric(spec, context, judge_clients)
            metric_results[metric_key] = result
            metric_scores[metric_key] = result.get("score")

        timestamp = iso_timestamp()
        upsert_annotation(
            placeholder_csv,
            prompt_id,
            stage_letter,
            annotator,
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
        help="Optional label for organizing outputs under evaluation/results/analysis_reports/<stage>/<label>",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotator rows")
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
