#!/usr/bin/env python3
"""
Convert ablation responses into the format expected by run_judge_scores.py.

Creates:
- raw_logs/<condition>/<stage>/stage_X_YY_*.txt files
- analysis_reports/<condition>/<stage>/stage_X_YY_*_meta.json files
"""

import json
from pathlib import Path
from datetime import datetime

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "results"

# Expected checks for judge evaluation (same as MENTOR uses)
EXPECTED_CHECKS = [
    "clarification_quality",
    "actionability", 
    "persona_compliance",
    "stage_awareness",
    "tone_constructive",
    "holistic_score",  # Most important for comparison
]


def load_prompts() -> dict:
    """Load the original prompts."""
    with open(ABLATION_DIR / "selected_prompts.json") as f:
        prompts = json.load(f)
    return {p["prompt_id"]: p for p in prompts}


def load_ablation_responses() -> dict:
    """Load the ablation responses."""
    with open(RESULTS_DIR / "ablation_responses.json") as f:
        return json.load(f)


def create_judge_files(condition: str, responses: dict, prompts: dict):
    """Create raw_logs and analysis_reports files for a condition."""
    
    # Create output directories
    raw_logs_base = ABLATION_DIR / "judge_inputs" / condition / "raw_logs"
    analysis_base = ABLATION_DIR / "judge_inputs" / condition / "analysis_reports"
    
    created = 0
    skipped = 0
    
    for prompt_id, resp_data in responses.items():
        # Skip if no response content
        response_text = resp_data.get("response", "")
        if not response_text or resp_data.get("full_response_length", 0) == 0:
            skipped += 1
            continue
        
        # Get stage from prompt_id (e.g., "stage_a_01" -> "a")
        parts = prompt_id.split("_")
        stage_letter = parts[1].lower()
        prompt_num = parts[2]
        
        # Get original prompt data
        prompt_data = prompts.get(prompt_id, {})
        original_prompt = prompt_data.get("prompt", "")
        
        # Create stage directories
        stage_raw = raw_logs_base / f"stage_{stage_letter}"
        stage_analysis = analysis_base / f"stage_{stage_letter}"
        stage_raw.mkdir(parents=True, exist_ok=True)
        stage_analysis.mkdir(parents=True, exist_ok=True)
        
        # File naming: match the MENTOR format
        system_alias = "ablation_no_stage"
        base_name = f"{prompt_id}_{system_alias}"
        
        # Write raw response
        response_file = stage_raw / f"{base_name}.txt"
        with open(response_file, "w") as f:
            f.write(response_text)
        
        # Write meta file
        meta_file = stage_analysis / f"{base_name}_meta.json"
        meta = {
            "prompt_id": prompt_id,
            "stage": stage_letter.upper(),
            "prompt": original_prompt,
            "expected_checks": EXPECTED_CHECKS,
            "metadata": prompt_data.get("metadata", {}),
            "system_id": f"ablation:{condition}",
            "system_alias": system_alias,
            "provider": "ablation",
            "model": "moonshotai/kimi-k2-thinking",
            "model_params": {
                "temperature": 0.0,
                "max_output_tokens": None,
                "seed": None
            },
            "prompt_variant": "ablated",
            "baseline_mode": False,
            "generated_at": datetime.now().isoformat() + "Z",
            "elapsed_seconds": resp_data.get("elapsed_seconds", 0),
            "tool_runs_count": 0,
            "system": f"ablation:{condition}",
            "response_path": str(response_file.resolve()),  # Absolute path
            "tool_trace_path": None,
            "run_timestamp": datetime.now().isoformat() + "Z",
            "success": True,
            "error": None,
        }
        
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        
        created += 1
    
    return created, skipped


def main():
    print("=" * 70)
    print("PREPARING ABLATION RESPONSES FOR JUDGES")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    prompts = load_prompts()
    ablation_data = load_ablation_responses()
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Process each condition
    print("\n[2/3] Creating judge input files...")
    
    conditions_to_process = ["no_stage_all"]  # Focus on the complete condition
    
    for condition in conditions_to_process:
        responses = ablation_data.get(condition, {})
        if not responses:
            print(f"  {condition}: No responses found, skipping")
            continue
        
        created, skipped = create_judge_files(condition, responses, prompts)
        print(f"  {condition}: Created {created} file pairs, skipped {skipped} (empty)")
    
    # Print instructions
    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    
    judge_inputs = ABLATION_DIR / "judge_inputs"
    print(f"\nJudge input files created at: {judge_inputs}")
    print("\nDirectory structure:")
    print("  judge_inputs/")
    print("    no_stage_all/")
    print("      raw_logs/stage_a/*.txt")
    print("      analysis_reports/stage_a/*_meta.json")
    print("      ...")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nRun judges for each stage:")
    print("""
cd /Users/majortimberwolf/Projects/lossfunk/ai-research-mentor/academic-research-mentor

# Run judges for each stage (A-F)
for stage in a b c d e f; do
  uv run python -m evaluation-results.scripts.evaluation-scripts.run_judge_scores \\
    --stage stage_$stage \\
    --judge openrouter:qwen/qwen3-max \\
    --judge openrouter:deepseek/deepseek-v3.2-exp \\
    --judge openrouter:x-ai/grok-4-fast \\
    --annotator ablation_judge \\
    --label ablation_no_stage \\
    --results-root icml-evaluation-results/ablations/stage_ablation/judge_inputs/no_stage_all
done
""")


if __name__ == "__main__":
    main()
