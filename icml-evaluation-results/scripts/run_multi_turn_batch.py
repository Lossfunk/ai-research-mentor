#!/usr/bin/env python3
"""Batch runner for multi-turn mentor evaluation using synthetic students."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.rich_formatter import print_error, print_info, print_success

from .multi_turn_orchestrator import (
    MultiTurnOrchestrator,
    export_transcripts,
    load_scenarios,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-turn mentor evaluations")
    parser.add_argument("--input", required=True, help="Path to JSONL scenario file")
    parser.add_argument("--mentors", nargs="+", required=True, help="Mentor system specs (e.g. openrouter:anthropic/claude-sonnet-4.5)")
    parser.add_argument("--student-model", default="google/gemini-2.5-flash-lite", help="OpenRouter model id for synthetic student")
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum turns to run per scenario")
    parser.add_argument("--output", required=True, help="Directory to write transcripts")
    parser.add_argument("--sample", type=int, help="Optional number of scenarios to sample from input")
    parser.add_argument("--dump-metadata", action="store_true", help="Print per-scenario metadata for debugging")
    parser.add_argument(
        "--baseline-mode",
        action="store_true",
        help="Run mentors in baseline mode (baseline prompt, guidelines off, limited tools)",
    )
    parser.add_argument(
        "--tool-whitelist",
        help="Comma-separated list of tools to allow (overrides default when provided)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    load_env_file()

    scenario_path = Path(args.input)
    if not scenario_path.exists():
        print_error(f"Scenario file not found: {scenario_path}")
        return 1

    scenarios = load_scenarios(scenario_path)
    if args.sample and args.sample > 0:
        scenarios = scenarios[: args.sample]

    if not scenarios:
        print_error("No scenarios loaded; aborting")
        return 1

    if args.dump_metadata:
        print_info(json.dumps(scenarios[0], ensure_ascii=False, indent=2))

    orchestrator = MultiTurnOrchestrator(
        mentor_specs=args.mentors,
        student_model=args.student_model,
        max_turns=args.max_turns,
        baseline_mode=args.baseline_mode,
        tool_whitelist=[tool.strip() for tool in args.tool_whitelist.split(",")] if args.tool_whitelist else None,
    )

    print_info(f"Running {len(scenarios)} scenario(s) across {len(args.mentors)} mentor(s) with max {args.max_turns} turn(s)")
    results = orchestrator.run_batch(scenarios)

    output_dir = Path(args.output)
    export_transcripts(results, output_dir)

    summary: Dict[str, Any] = {}
    for system_id, records in results.items():
        summary[system_id] = {
            "scenarios": len(records),
            "successes": sum(1 for record in records if record.success),
            "failures": sum(1 for record in records if not record.success),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_success(f"Multi-turn evaluation complete. Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
