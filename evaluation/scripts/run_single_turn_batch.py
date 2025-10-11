#!/usr/bin/env python3
"""
Batch runner for single-turn evaluation with automated analysis.

Usage:
  uv run python -m evaluation.scripts.run_single_turn_batch \
      --input evaluation/data/evals_single_turn.jsonl \
      --systems openai/gpt-4o claude/sonnet-4 \
      --judge claude/sonnet-4 \
      --output-label quick_test
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.rich_formatter import print_error, print_info, print_success

from .judge_utils import build_judge_clients
from .run_judge_scores import run_judges
from .run_manual_stage import ensure_stage_directories, normalize_stage
from .single_turn_orchestrator import SingleTurnOrchestrator


def load_test_data(data_path: Path) -> List[Dict]:
    """Load single-turn test prompts from JSONL file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")
    
    prompts = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def run_system_responses(
    prompts: List[Dict],
    systems: Sequence[str],
    stage: str,
    output_base: Path,
) -> Dict[str, List[str]]:
    """
    Generate responses from all target systems for each prompt.
    
    Returns mapping of system_id -> list of prompt_ids processed
    """
    orchestrator = SingleTurnOrchestrator(systems_to_test=systems)
    stage_letter, stage_folder = normalize_stage(stage)
    _, raw_dir, _ = ensure_stage_directories(stage_folder)
    
    system_results = {system: [] for system in systems}
    total_prompts = len(prompts)
    
    print_info(f"Generating responses for {total_prompts} prompts across {len(systems)} systems...")
    
    for i, prompt_data in enumerate(prompts, 1):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["prompt"]
        print_info(f"[{i}/{total_prompts}] Processing {prompt_id}...")
        
        system_responses = orchestrator.process_single_prompt(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            expected_tools=prompt_data.get("metadata", {}).get("expected_tools", []),
            stage=stage_letter
        )
        
        # Save responses to raw logs directory
        for system_id, response_data in system_responses.items():
            if response_data:
                # Save raw response
                response_path = raw_dir / f"{prompt_id}_{system_id.replace('/', '_')}.txt"
                response_path.write_text(response_data["response"], encoding="utf-8")
                
                # Save tool trace
                if response_data.get("tool_trace"):
                    tool_path = raw_dir / f"{prompt_id}_{system_id.replace('/', '_')}_tools.json"
                    tool_path.write_text(json.dumps(response_data["tool_trace"], indent=2), encoding="utf-8")
                
                # Create meta file
                meta_file = raw_dir / f"{prompt_id}_{system_id.replace('/', '_')}_meta.json"
                meta_data = {
                    "prompt_id": prompt_id,
                    "stage": stage_letter,
                    "prompt": prompt_text,
                    "system": system_id,
                    "expected_checks": prompt_data.get("expected_checks", []),
                    "metadata": prompt_data.get("metadata", {}),
                    "response_path": str(response_path),
                    "tool_trace_path": str(tool_path) if response_data.get("tool_trace") else None,
                    "run_timestamp": response_data.get("timestamp"),
                    "elapsed_seconds": response_data.get("elapsed", 0),
                    "tool_runs_count": len(response_data.get("tool_trace", [])),
                }
                meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                
                system_results[system_id].append(prompt_id)
                print_success(f"  ✓ {system_id}: Response saved")
            else:
                print_error(f"  ✗ {system_id}: Failed to generate response")
    
    return system_results


def generate_summary_report(
    system_results: Dict[str, List[str]], 
    total_prompts: int,
    stage: str,
    output_label: str,
    output_dir: Path
) -> None:
    """Generate summary report for the batch run."""
    summary = {
        "run_type": "single_turn_batch",
        "stage": stage,
        "output_label": output_label,
        "total_prompts": total_prompts,
        "systems": {
            system: {
                "processed_count": len(prompt_ids),
                "success_rate": len(prompt_ids) / total_prompts,
                "prompt_ids": prompt_ids
            }
            for system, prompt_ids in system_results.items()
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    summary_path = output_dir / f"{output_label}_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_success(f"Batch summary saved: {summary_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Batch runner for single-turn evaluation")
    parser.add_argument("--input", required=True, help="Path to single-turn JSONL test data")
    parser.add_argument("--systems", nargs="+", required=True, help="Systems to test (provider/model format)")
    parser.add_argument("--stage", default="stage_a", help="Stage to evaluate (stage_a, stage_b, stage_c)")
    parser.add_argument("--judge", required=True, help="Judge model for scoring")
    parser.add_argument("--output-label", default="batch_run", help="Label for organizing outputs")
    parser.add_argument("--skip-generation", action="store_true", help="Skip system response generation, only run scoring")
    args = parser.parse_args(argv)
    
    # Setup directories
    stage_letter, stage_folder = normalize_stage(args.stage)
    raw_dir, analysis_dir, _ = ensure_stage_directories(stage_folder)
    
    # Load test data
    prompts = load_test_data(Path(args.input))
    print_info(f"Loaded {len(prompts)} test prompts from {args.input}")
    
    # Phase 1: Generate system responses (unless skipped)
    if not args.skip_generation:
        system_results = run_system_responses(prompts, args.systems, args.stage, raw_dir)
        generate_summary_report(system_results, len(prompts), args.stage, args.output_label, analysis_dir)
    else:
        print_info("Skipping response generation (--skip-generation)")
    
    # Phase 2: Run judge scoring
    print_info(f"Running judge scoring with model: {args.judge}")
    
    # Find all meta files for the stage (including from multiple systems)
    all_meta_files = []
    for system in args.systems:
        system_safe = system.replace("/", "_")
        all_meta_files.extend(raw_dir.glob(f"*_{system_safe}_meta.json"))
    
    if not all_meta_files:
        print_error("No meta files found. Run with --skip-generation=false first.")
        return 1
    
    print_info(f"Found {len(all_meta_files)} meta files for scoring")
    
    # Score each system separately
    for system in args.systems:
        system_safe = system.replace("/", "_")
        system_meta_files = [f for f in all_meta_files if f.name.endswith(f"_{system_safe}_meta.json")]
        
        print_info(f"Scoring {system} ({len(system_meta_files)} prompts)...")
        
        # Create temporary directory for this system's judge outputs
        system_output_dir = analysis_dir / args.output_label / system_safe
        system_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move relevant meta files to temporary location for scoring
        temp_meta_dir = system_output_dir / "meta_temp"
        temp_meta_dir.mkdir(exist_ok=True)
        
        for meta_file in system_meta_files:
            import shutil
            shutil.copy2(meta_file, temp_meta_dir / meta_file.name)
        
        # Judge scoring (modifying run_judges to use our temp meta files)
        try:
            summary = run_judges(
                stage=args.stage,
                prompt_ids=None,  # Process all
                judge_specs=[args.judge],
                annotator=f"batch_{system_safe}",
                force=False,
                output_label=f"{args.output_label}/{system_safe}"
            )
            print_success(f"Completed scoring for {system}")
        except Exception as exc:
            print_error(f"Scoring failed for {system}: {exc}")
    
    print_success("Batch evaluation completed!")
    print_info(f"Results available in: {analysis_dir / args.output_label}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
