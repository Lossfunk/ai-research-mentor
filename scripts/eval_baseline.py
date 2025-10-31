#!/usr/bin/env python3
"""
Baseline evaluation runner for Academic Research Mentor.

Runs Stage A (or specified stages), writes artifacts (responses, tool traces, meta),
and optionally executes LLM judges to score metrics using the configured models.

Usage examples:
  # Run Stage A artifacts and judges with default label
  uv run python scripts/eval_baseline.py --stage stage_a \
      --judge openrouter:anthropic/claude-4-sonnet \
      --judge openrouter:google/gemini-2.5-flash \
      --annotator auto --label baseline --force

  # Only generate artifacts, skip judges
  uv run python scripts/eval_baseline.py --stage A --skip-judges --force

Outputs:
  - evals-for-papers/results/raw_logs/<stage>/*.txt, *_tools.json
  - evals-for-papers/results/analysis_reports/<stage>/*_meta.json, annotation CSVs
  - docs/tech-report/artifacts/<label>/latest.json (combined summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Import direct functions to avoid subprocesses
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.run_manual_stage import run_stage
from evaluation.scripts.run_judge_scores import run_judges


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline evals and optional judges")
    p.add_argument(
        "--stage",
        dest="stages",
        action="append",
        default=["stage_a"],
        help="Stage(s) to run (A|B|C or stage_a|stage_b|stage_c); repeat to run multiple",
    )
    p.add_argument(
        "--judge",
        dest="judges",
        action="append",
        help=(
            "Judge spec provider:model (repeat). Examples: "
            "openrouter:anthropic/claude-4-sonnet, openrouter:google/gemini-2.5-flash"
        ),
    )
    p.add_argument("--annotator", default="auto", help="Annotator name recorded in CSVs")
    p.add_argument("--label", default="baseline", help="Label for analysis outputs")
    p.add_argument("--force", action="store_true", help="Overwrite existing artifacts/rows where applicable")
    p.add_argument("--skip-judges", action="store_true", help="Generate artifacts but skip judge scoring")
    return p.parse_args(argv)


def _normalize_stage_name(raw: str) -> str:
    v = raw.strip().lower()
    if v in {"a", "stage_a"}:  # canonical folder name
        return "stage_a"
    if v in {"b", "stage_b"}:
        return "stage_b"
    if v in {"c", "stage_c"}:
        return "stage_c"
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    stages = [_normalize_stage_name(s) for s in (args.stages or ["stage_a"])]

    summaries: List[Dict[str, Any]] = []
    for stage in stages:
        # Generate artifacts for the stage
        stage_summary = run_stage(stage, prompt_ids=None, force=args.force)
        summaries.append(stage_summary)

        # Optionally run judges on produced artifacts
        if not args.skip_judges and args.judges:
            judge_summary = run_judges(
                stage=stage,
                prompt_ids=None,
                judge_specs=args.judges,
                annotator=args.annotator,
                force=args.force,
                output_label=args.label,
            )
            # Attach judge info for convenience
            stage_summary["judge_summary"] = judge_summary

    combined = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stages": [s.get("stage") for s in summaries],
        "total_prompts": sum(int(s.get("total_prompts", 0)) for s in summaries),
        "summaries": summaries,
        "label": args.label,
    }

    out_dir = Path("docs/tech-report/artifacts") / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.json"
    latest_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-baseline] wrote combined summary -> {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
