#!/usr/bin/env python3
"""
Aggregate student-judge metrics across stages and systems.

Usage:
    uv run python evaluation/scripts/summarize_student_scores.py \
        --results-root evals-for-papers/results/analysis_reports \
        --output reports/evals/student_summary.json

The script scans for directories named `student_outcome_judge*` inside each
stage folder (stage_a … stage_f). For every system represented (mentor manual,
baselines, external models) it averages the scalar metrics found in the
`student_metrics` block of each `*_student_judges.json` file and reports the
prompt counts. The optional `--output` flag writes the aggregated data to disk
as pretty-printed JSON; otherwise results are printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


STAGES = [f"stage_{code}" for code in "abcdef"]
METRIC_KEYS = [
    "student_actionability",
    "student_clarity",
    "student_constraint_fit",
    "student_confidence_gain",
    "student_path_ready",
    "student_failure_modes",
    "student_outcome_score",
]


@dataclass
class Aggregate:
    prompts: int
    metrics: Dict[str, float]


def _load_student_metrics(file_path: Path) -> Optional[Dict[str, float]]:
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    student_metrics = data.get("student_metrics")
    if not isinstance(student_metrics, dict):
        return None

    metrics: Dict[str, float] = {}
    for key in METRIC_KEYS:
        value = student_metrics.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics if metrics else None


def _iter_system_dirs(stage_dir: Path) -> Iterable[Path]:
    for child in stage_dir.iterdir():
        if child.is_dir() and child.name.startswith("student_outcome_judge"):
            yield child


def _aggregate_system(system_dir: Path) -> Optional[Aggregate]:
    metric_lists: Dict[str, List[float]] = defaultdict(list)
    prompt_files = sorted(system_dir.glob("*_student_judges.json"))

    for file_path in prompt_files:
        metrics = _load_student_metrics(file_path)
        if not metrics:
            continue
        for key, value in metrics.items():
            metric_lists[key].append(value)

    if not metric_lists:
        return None

    aggregated = {key: mean(values) for key, values in metric_lists.items()}
    return Aggregate(prompts=len(prompt_files), metrics=aggregated)


def summarize(results_root: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for stage in STAGES:
        stage_dir = results_root / stage
        if not stage_dir.is_dir():
            continue

        systems: Dict[str, Dict[str, float]] = {}
        for system_dir in _iter_system_dirs(stage_dir):
            aggregate = _aggregate_system(system_dir)
            if not aggregate:
                continue
            systems[system_dir.name] = {
                "prompts": aggregate.prompts,
                **aggregate.metrics,
            }

        if systems:
            summary[stage] = systems

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize student judge metrics.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("evals-for-papers/results/analysis_reports"),
        help="Root directory containing per-stage analysis reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregated summary as JSON.",
    )
    args = parser.parse_args(argv)

    summary = summarize(args.results_root)
    if not summary:
        print("No student judge results found.", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"Wrote summary to {args.output}")
    else:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
