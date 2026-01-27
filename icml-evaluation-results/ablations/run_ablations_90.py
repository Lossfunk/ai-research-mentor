#!/usr/bin/env python3
"""
Run ablation experiments on all 90 single-turn prompts.

This ensures the ablation baseline matches the full single-turn evaluation (1.604).

Ablation conditions:
1. no_stage - Remove stage awareness from system prompt
2. no_guidelines - Replace mentor prompt with generic assistant

Usage:
    # Dry run (check setup, no API calls):
    uv run python icml-evaluation-results/ablations/run_ablations_90.py

    # Actually run ablations (costs ~$50-100 in API calls):
    RUN_ABLATIONS=1 uv run python icml-evaluation-results/ablations/run_ablations_90.py

    # Run only one condition:
    RUN_ABLATIONS=1 ABLATION_CONDITION=no_stage uv run python icml-evaluation-results/ablations/run_ablations_90.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from academic_research_mentor.cli.session import load_env_file
except ImportError:
    def load_env_file() -> None:
        pass

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
ABLATION_DIR = Path(__file__).parent
PROMPTS_FILE = ABLATION_DIR.parent / "prompts" / "evals_single_turn.jsonl"
OUTPUT_DIR = ABLATION_DIR / "ablations_90"

# Ablated prompt files
NO_STAGE_PROMPT = ABLATION_DIR / "stage_ablation" / "prompt_no_stage.md"
NO_GUIDELINES_PROMPT = ABLATION_DIR / "guidelines_ablation" / "prompt_no_guidelines.md"
FULL_MENTOR_PROMPT = PROJECT_ROOT / "src" / "academic_research_mentor" / "prompt.md"

# =============================================================================
# MODEL CONFIG
# =============================================================================
MODEL = "moonshotai/kimi-k2-thinking"
TEMPERATURE = 0.0
MAX_TOKENS = 4096

# Stage C directive (same as full MENTOR uses)
STAGE_C_DIRECTIVE = (
    "You are preparing a turnkey research execution plan with publication-grade rigor. "
    "Do not request additional context unless it is absolutely required to avoid catastrophic mistakes. "
    "Respond with the following sections in order:\n\n"
    "1. Problem framing and goals\n"
    "2. Experiments (each with hypothesis, setup, baselines, evaluation metrics, and expected outcomes)\n"
    "3. Timeline for the next 6 months with milestones\n"
    "4. Resources (compute, tools, datasets)\n"
    "5. Risks and mitigations table\n"
    "6. Stretch ideas or follow-up directions\n\n"
    "Use tools when they yield clearly relevant sources. If retrieved evidence is generic or off-topic, "
    "note the limitation and propose how to gather better references instead of forcing a citation. "
    "When invoking web search, craft domain-specific queries that combine method + task keywords and, "
    "when appropriate, filters such as 'arXiv', 'state-of-the-art', or key dataset names to surface precise academic sources. "
    "When citing, prefer [file:page] for attachments and [n] for web results. "
    "If no high-confidence references exist, state that explicitly and describe how you would acquire authoritative evidence. "
    "Finish with a single optional follow-up suggestion labelled 'Optional next step'."
)


def load_all_prompts() -> List[Dict[str, Any]]:
    """Load all 90 prompts from the evaluation JSONL file."""
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append({
                    "prompt_id": data["prompt_id"],
                    "prompt": data["prompt"],
                    "stage": data["metadata"]["stage"],
                    "expected_checks": data.get("expected_checks", []),
                    "metadata": data.get("metadata", {}),
                })
    return prompts


def load_system_prompt(condition: str) -> str:
    """Load the appropriate system prompt for the ablation condition."""
    if condition == "no_stage":
        prompt_path = NO_STAGE_PROMPT
    elif condition == "no_guidelines":
        prompt_path = NO_GUIDELINES_PROMPT
    elif condition == "full_mentor":
        prompt_path = FULL_MENTOR_PROMPT
    else:
        raise ValueError(f"Unknown condition: {condition}")

    with open(prompt_path) as f:
        return f.read()


def get_client() -> OpenAI:
    """Create OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def run_single_prompt(
    client: OpenAI,
    prompt_text: str,
    stage: str,
    system_prompt: str,
    condition: str,
) -> Dict[str, Any]:
    """Run a single prompt through the LLM."""

    # Apply stage C directive only for full_mentor condition
    final_prompt = prompt_text
    if stage.upper() == "C" and condition == "full_mentor":
        final_prompt = f"{STAGE_C_DIRECTIVE}\n\n{prompt_text}"

    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    elapsed = time.time() - start

    # Handle Kimi-K2-thinking response format (same as stage_ablation)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = None

    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        reasoning = choice.message.reasoning_content
    elif hasattr(choice.message, "model_extra") and choice.message.model_extra:
        extra = choice.message.model_extra
        for key in ("reasoning", "reasoning_content", "thinking"):
            if extra.get(key):
                reasoning = extra[key]
                break

    if reasoning:
        if not content:
            content = f"<thinking>\n{reasoning}\n</thinking>"
        elif not content.startswith("<thinking>"):
            content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"

    if not content and response.usage and response.usage.completion_tokens:
        import sys
        print(
            "\n    WARNING: Empty content but output_tokens > 0; model may use unsupported format.",
            file=sys.stderr,
        )

    return {
        "response": content,
        "elapsed_seconds": elapsed,
        "usage": {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
    }


def run_ablation_condition(
    client: OpenAI,
    condition: str,
    prompts: List[Dict[str, Any]],
    output_dir: Path,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single ablation condition on all prompts."""

    print(f"\n{'='*70}")
    print(f"ABLATION CONDITION: {condition}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load system prompt
    system_prompt = load_system_prompt(condition)

    # Create output directories
    condition_dir = output_dir / condition
    raw_logs_dir = condition_dir / "raw_logs"
    raw_logs_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = []
    total_input = 0
    total_output = 0
    errors = []

    # Check for existing results (for resume)
    existing = set()
    if resume_from:
        for f in raw_logs_dir.glob("*.txt"):
            existing.add(f.stem)
        print(f"  Resuming: found {len(existing)} existing responses\n")

    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["prompt"]
        stage = prompt_data["stage"]

        # Skip if already done (resume mode)
        if prompt_id in existing:
            print(f"  [{i+1:2d}/{len(prompts)}] {prompt_id} (skipped - exists)")
            continue

        print(f"  [{i+1:2d}/{len(prompts)}] {prompt_id} (stage {stage})...", end=" ", flush=True)

        try:
            result = run_single_prompt(
                client,
                prompt_text,
                stage,
                system_prompt,
                condition,
            )

            if not (result["response"] or "").strip():
                print("EMPTY (skipped)")
                errors.append({"prompt_id": prompt_id, "error": "empty_response"})
                continue

            # Save raw response
            response_path = raw_logs_dir / f"{prompt_id}.txt"
            response_path.write_text(result["response"], encoding="utf-8")

            # Save metadata (use absolute path so judges resolve correctly from any cwd)
            meta = {
                "prompt_id": prompt_id,
                "stage": stage,
                "condition": condition,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "elapsed_seconds": result["elapsed_seconds"],
                "usage": result["usage"],
                "response_path": str(response_path.resolve()),
                "timestamp": datetime.now().isoformat(),
                "expected_checks": prompt_data.get("expected_checks", []),
                "metadata": prompt_data.get("metadata", {}),
            }
            meta_path = raw_logs_dir / f"{prompt_id}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            results.append(meta)
            total_input += result["usage"]["input_tokens"]
            total_output += result["usage"]["output_tokens"]

            print(f"done ({result['elapsed_seconds']:.1f}s, {result['usage']['output_tokens']} tokens)")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"prompt_id": prompt_id, "error": str(e)})

    # Save summary
    summary = {
        "condition": condition,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "n_prompts": len(prompts),
        "n_success": len(results),
        "n_errors": len(errors),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = condition_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Summary: {len(results)} success, {len(errors)} errors")
    print(f"  Tokens: {total_input:,} input, {total_output:,} output")
    print(f"  Saved to: {condition_dir}")

    return summary


def main():
    import argparse

    load_env_file()

    parser = argparse.ArgumentParser(description="Run 90-prompt ablation experiments")
    parser.add_argument("--stages", nargs="+", help="Only run these stages (e.g., --stages D E)")
    parser.add_argument("--prompt-ids", nargs="+", help="Only run these prompt IDs (e.g., --prompt-ids stage_d_06 stage_e_01)")
    args = parser.parse_args()

    print("=" * 70)
    print("90-PROMPT ABLATION EXPERIMENT")
    print("=" * 70)

    # Load prompts
    prompts = load_all_prompts()
    print(f"\nLoaded {len(prompts)} prompts")

    # Filter prompts if requested
    if args.stages:
        stages_upper = [s.upper() for s in args.stages]
        prompts = [p for p in prompts if p["stage"].upper() in stages_upper]
        print(f"Filtered to stages {args.stages}: {len(prompts)} prompts")

    if args.prompt_ids:
        prompt_ids_set = set(args.prompt_ids)
        prompts = [p for p in prompts if p["prompt_id"] in prompt_ids_set]
        print(f"Filtered to prompt IDs: {len(prompts)} prompts")

    if not prompts:
        print("Error: No prompts to run after filtering")
        return

    # Count by stage
    stage_counts = {}
    for p in prompts:
        stage_counts[p["stage"]] = stage_counts.get(p["stage"], 0) + 1
    print(f"  By stage: {dict(sorted(stage_counts.items()))}")

    # Check if we should run
    run_ablations = os.environ.get("RUN_ABLATIONS", "0") == "1"
    condition_filter = os.environ.get("ABLATION_CONDITION", None)
    resume = os.environ.get("RESUME", "0") == "1"

    if not run_ablations:
        print("\n" + "=" * 70)
        print("DRY RUN - No API calls will be made")
        print("=" * 70)
        print("\nTo run the ablations:")
        print("  RUN_ABLATIONS=1 uv run python icml-evaluation-results/ablations/run_ablations_90.py")
        print("\nTo run only one condition:")
        print("  RUN_ABLATIONS=1 ABLATION_CONDITION=no_stage uv run python ...")
        print("  RUN_ABLATIONS=1 ABLATION_CONDITION=no_guidelines uv run python ...")
        print("\nTo resume interrupted run:")
        print("  RUN_ABLATIONS=1 RESUME=1 uv run python ...")
        print("\nTo run only specific stages:")
        print("  RUN_ABLATIONS=1 uv run python ... --stages D E")
        print("\nTo run only specific prompts:")
        print("  RUN_ABLATIONS=1 uv run python ... --prompt-ids stage_d_06 stage_e_01")
        print("\nEstimated cost: ~$50-100 for both conditions")
        print("\nFiles that will be created:")
        print(f"  {OUTPUT_DIR}/no_stage/raw_logs/*.txt")
        print(f"  {OUTPUT_DIR}/no_stage/raw_logs/*_meta.json")
        print(f"  {OUTPUT_DIR}/no_guidelines/raw_logs/*.txt")
        print(f"  {OUTPUT_DIR}/no_guidelines/raw_logs/*_meta.json")
        return

    # Create client
    try:
        client = get_client()
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which conditions to run
    conditions = ["no_stage", "no_guidelines"]
    if condition_filter:
        if condition_filter not in conditions:
            print(f"Error: Unknown condition '{condition_filter}'")
            print(f"Valid conditions: {conditions}")
            return
        conditions = [condition_filter]

    print(f"\nRunning conditions: {conditions}")
    print(f"Resume mode: {resume}")

    # Run ablations
    summaries = {}
    for condition in conditions:
        summary = run_ablation_condition(
            client,
            condition,
            prompts,
            OUTPUT_DIR,
            resume_from=condition if resume else None,
        )
        summaries[condition] = summary

    # Final summary
    print("\n" + "=" * 70)
    print("ABLATION RUN COMPLETE")
    print("=" * 70)

    for condition, summary in summaries.items():
        print(f"\n  {condition}:")
        print(f"    Success: {summary['n_success']}/{summary['n_prompts']}")
        print(f"    Tokens: {summary['total_input_tokens']:,} in, {summary['total_output_tokens']:,} out")
        if summary['errors']:
            print(f"    Errors: {len(summary['errors'])}")

    print(f"\n\nNext step: Run judges on ablation responses")
    print(f"  uv run python icml-evaluation-results/ablations/run_ablation_judges.py")


if __name__ == "__main__":
    main()
