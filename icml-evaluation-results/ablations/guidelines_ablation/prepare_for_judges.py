#!/usr/bin/env python3
"""
Convert guidelines ablation responses into the format expected by run_judge_scores.py.
"""

import json
from pathlib import Path
from datetime import datetime

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "results"
STAGE_ABLATION_DIR = ABLATION_DIR.parent / "stage_ablation"

# Expected checks for judge evaluation
EXPECTED_CHECKS = [
    "clarification_quality",
    "actionability", 
    "persona_compliance",
    "stage_awareness",
    "tone_constructive",
    "holistic_score",
]


def load_prompts() -> dict:
    """Load the original prompts from stage ablation."""
    with open(STAGE_ABLATION_DIR / "selected_prompts.json") as f:
        prompts = json.load(f)
    return {p["prompt_id"]: p for p in prompts}


def load_ablation_responses() -> dict:
    """Load the ablation responses."""
    with open(RESULTS_DIR / "ablation_responses.json") as f:
        return json.load(f)


def create_judge_files(condition: str, responses: dict, prompts: dict):
    """Create raw_logs and analysis_reports files for a condition."""
    
    raw_logs_base = ABLATION_DIR / "judge_inputs" / condition / "raw_logs"
    analysis_base = ABLATION_DIR / "judge_inputs" / condition / "analysis_reports"
    
    created = 0
    skipped = 0
    
    for prompt_id, resp_data in responses.items():
        response_text = resp_data.get("response", "")
        if not response_text or resp_data.get("full_response_length", 0) == 0:
            skipped += 1
            continue
        
        parts = prompt_id.split("_")
        stage_letter = parts[1].lower()
        
        prompt_data = prompts.get(prompt_id, {})
        original_prompt = prompt_data.get("prompt", "")
        
        stage_raw = raw_logs_base / f"stage_{stage_letter}"
        stage_analysis = analysis_base / f"stage_{stage_letter}"
        stage_raw.mkdir(parents=True, exist_ok=True)
        stage_analysis.mkdir(parents=True, exist_ok=True)
        
        system_alias = "ablation_no_guidelines"
        base_name = f"{prompt_id}_{system_alias}"
        
        response_file = stage_raw / f"{base_name}.txt"
        with open(response_file, "w") as f:
            f.write(response_text)
        
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
            "model_params": {"temperature": 0.0},
            "prompt_variant": "ablated_no_guidelines",
            "baseline_mode": False,
            "generated_at": datetime.now().isoformat() + "Z",
            "elapsed_seconds": resp_data.get("elapsed_seconds", 0),
            "tool_runs_count": 0,
            "system": f"ablation:{condition}",
            "response_path": str(response_file.resolve()),
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
    print("PREPARING GUIDELINES ABLATION FOR JUDGES")
    print("=" * 70)
    
    print("\n[1/3] Loading data...")
    prompts = load_prompts()
    ablation_data = load_ablation_responses()
    
    print(f"  Loaded {len(prompts)} prompts")
    
    print("\n[2/3] Creating judge input files...")
    
    responses = ablation_data.get("no_guidelines", {})
    if not responses:
        print("  no_guidelines: No responses found!")
        return
    
    created, skipped = create_judge_files("no_guidelines", responses, prompts)
    print(f"  no_guidelines: Created {created} file pairs, skipped {skipped} (empty)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nRun judges:")
    print("""
cd /Users/majortimberwolf/Projects/lossfunk/ai-research-mentor/academic-research-mentor

for stage in a b c d e f; do
  uv run python -m evaluation.scripts.run_judge_scores \\
    --stage stage_$stage \\
    --judge openrouter:qwen/qwen3-max \\
    --judge openrouter:deepseek/deepseek-v3.2-exp \\
    --judge openrouter:x-ai/grok-4-fast \\
    --annotator ablation_judge \\
    --label ablation_no_guidelines \\
    --results-root icml-evaluation-results/ablations/guidelines_ablation/judge_inputs/no_guidelines \\
    --system-subdir .
done
""")


if __name__ == "__main__":
    main()
