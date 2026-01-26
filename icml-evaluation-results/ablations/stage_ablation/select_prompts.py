#!/usr/bin/env python3
"""Select prompts for stage ablation study."""

import json
from pathlib import Path

def extract_prompts():
    """Extract prompts from existing meta files."""
    base_dir = Path(__file__).parent.parent.parent / "analysis_reports" / "mentor"
    
    # Select prompts 01, 05, 10 from each stage (3 per stage = 18 total)
    selected_ids = ["01", "05", "10"]
    
    prompts = []
    
    for stage in ["a", "b", "c", "d", "e", "f"]:
        stage_dir = base_dir / f"stage_{stage}"
        
        if not stage_dir.exists():
            print(f"Warning: {stage_dir} not found")
            continue
        
        for prompt_num in selected_ids:
            meta_file = stage_dir / f"stage_{stage}_{prompt_num}_openrouter_moonshotai_kimi-k2-thinking_meta.json"
            
            if not meta_file.exists():
                print(f"Warning: {meta_file} not found")
                continue
            
            with open(meta_file) as f:
                data = json.load(f)
            
            prompts.append({
                "prompt_id": data.get("prompt_id"),
                "stage": stage.upper(),
                "prompt": data.get("prompt"),
                "metadata": data.get("metadata", {}),
                "expected_checks": data.get("expected_checks", [])
            })
    
    return prompts


def main():
    prompts = extract_prompts()
    
    output_file = Path(__file__).parent / "selected_prompts.json"
    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Extracted {len(prompts)} prompts")
    print(f"Saved to {output_file}")
    
    # Print summary
    print("\nPrompts by stage:")
    for stage in ["A", "B", "C", "D", "E", "F"]:
        stage_prompts = [p for p in prompts if p["stage"] == stage]
        print(f"  Stage {stage}: {len(stage_prompts)} prompts")
        for p in stage_prompts:
            prompt_preview = p["prompt"][:60] + "..." if len(p["prompt"]) > 60 else p["prompt"]
            print(f"    - {p['prompt_id']}: {prompt_preview}")


if __name__ == "__main__":
    main()
