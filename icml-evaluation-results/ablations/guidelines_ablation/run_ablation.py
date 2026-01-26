#!/usr/bin/env python3
"""
Run guidelines ablation experiment.

Tests: Does the curated mentor prompt matter vs a generic research assistant?

Conditions:
1. full_mentor - Use existing results (MENTOR with full prompt)
2. no_guidelines - Generic research assistant prompt (no mentor guidance)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

from openai import OpenAI

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ABLATION_DIR = Path(__file__).parent
STAGE_ABLATION_DIR = ABLATION_DIR.parent / "stage_ablation"


def load_selected_prompts() -> List[Dict[str, Any]]:
    """Load the same prompts used in stage ablation."""
    prompts_file = STAGE_ABLATION_DIR / "selected_prompts.json"
    with open(prompts_file) as f:
        return json.load(f)


def load_full_mentor_results() -> Dict[str, Dict[str, Any]]:
    """Load existing Full MENTOR results for comparison."""
    baseline_file = STAGE_ABLATION_DIR / "results" / "full_mentor_baseline.json"
    with open(baseline_file) as f:
        return json.load(f)


def load_system_prompt(use_ablated: bool = False) -> str:
    """Load the system prompt."""
    if use_ablated:
        prompt_path = ABLATION_DIR / "prompt_no_guidelines.md"
    else:
        prompt_path = PROJECT_ROOT / "src" / "academic_research_mentor" / "prompt.md"
    
    with open(prompt_path) as f:
        return f.read()


def get_openrouter_client() -> OpenAI:
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
    system_prompt: str,
) -> Dict[str, Any]:
    """Run a single prompt through the LLM."""
    
    start = time.time()
    
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-thinking",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    
    elapsed = time.time() - start
    
    # Handle Kimi-K2-thinking response format
    choice = response.choices[0]
    content = choice.message.content or ""
    
    # Check for reasoning_content (Kimi-K2-thinking specific)
    if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
        reasoning = choice.message.reasoning_content
        if not content and reasoning:
            content = f"<thinking>\n{reasoning}\n</thinking>"
        elif reasoning and not content.startswith("<thinking>"):
            content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
    
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
    *,
    use_ablated_prompt: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run a single ablation condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"  use_ablated_prompt={use_ablated_prompt}")
    print(f"  prompts={len(prompts)}")
    print(f"{'='*60}\n")
    
    # Load appropriate system prompt
    system_prompt = load_system_prompt(use_ablated=use_ablated_prompt)
    
    results = {}
    total_input = 0
    total_output = 0
    
    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["prompt"]
        stage = prompt_data["stage"]
        
        print(f"  [{i+1}/{len(prompts)}] {prompt_id}...", end=" ", flush=True)
        
        try:
            result = run_single_prompt(
                client,
                prompt_text,
                system_prompt,
            )
            
            # Truncate response for storage
            response_preview = result["response"][:1000] + "..." if len(result["response"]) > 1000 else result["response"]
            
            results[prompt_id] = {
                "response": response_preview,
                "full_response_length": len(result["response"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "condition": condition,
                "stage": stage,
                "usage": result["usage"],
            }
            
            total_input += result["usage"]["input_tokens"]
            total_output += result["usage"]["output_tokens"]
            
            print(f"done ({result['elapsed_seconds']:.1f}s, {result['usage']['output_tokens']} tokens)")
            
        except Exception as e:
            print(f"error: {e}")
            results[prompt_id] = {"error": str(e), "condition": condition, "stage": stage}
    
    print(f"\n  Total tokens: {total_input:,} input, {total_output:,} output")
    
    return results


def main():
    print("=" * 70)
    print("GUIDELINES ABLATION EXPERIMENT")
    print("=" * 70)
    
    # Load prompts (same as stage ablation)
    prompts = load_selected_prompts()
    print(f"\nLoaded {len(prompts)} prompts for ablation")
    
    # Load existing Full MENTOR results
    print("\n[1/2] Loading Full MENTOR baseline results...")
    full_mentor_results = load_full_mentor_results()
    print(f"  Loaded {len(full_mentor_results)} existing results")
    
    # Print baseline summary
    holistic_scores = [v.get("holistic_score", 0) for v in full_mentor_results.values() if v.get("holistic_score")]
    
    print(f"\n  Full MENTOR baseline:")
    print(f"    holistic_score: {sum(holistic_scores)/len(holistic_scores):.3f} (n={len(holistic_scores)})")
    
    # Save baseline reference
    output_dir = ABLATION_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Check if we should run ablations
    run_ablations = os.environ.get("RUN_ABLATIONS", "0") == "1"
    
    if not run_ablations:
        print("\n" + "=" * 70)
        print("ABLATION SETUP COMPLETE")
        print("=" * 70)
        print("\nTo run the actual ablation experiments:")
        print("  RUN_ABLATIONS=1 uv run python run_ablation.py")
        print("\nThis will call the LLM API and incur costs (~$5-10 estimated)")
        print("\nFiles created:")
        print(f"  - prompt_no_guidelines.md")
        return
    
    # Create OpenRouter client
    try:
        client = get_openrouter_client()
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENROUTER_API_KEY in your environment or .env file")
        return
    
    # Run ablation condition
    print("\n[2/2] Running -Guidelines condition...")
    no_guidelines = run_ablation_condition(
        client,
        "no_guidelines",
        prompts,
        use_ablated_prompt=True,
    )
    
    # Save results
    all_results = {
        "metadata": {
            "model": "moonshotai/kimi-k2-thinking",
            "temperature": 0.0,
            "n_prompts": len(prompts),
            "conditions": ["full_mentor", "no_guidelines"],
            "ablation_type": "guidelines",
            "description": "Tests if curated mentor prompt matters vs generic research assistant"
        },
        "full_mentor": full_mentor_results,
        "no_guidelines": no_guidelines,
    }
    
    with open(output_dir / "ablation_responses.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Responses saved to: {output_dir / 'ablation_responses.json'}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION RESPONSES COLLECTED")
    print("=" * 70)
    
    errors = sum(1 for v in no_guidelines.values() if "error" in v)
    success = len(no_guidelines) - errors
    print(f"\n  no_guidelines:")
    print(f"    Success: {success}/{len(no_guidelines)}")
    if errors:
        print(f"    Errors: {errors}")
    
    print(f"\n\nNext step: Run judges on ablation responses")
    print(f"  python prepare_for_judges.py")


if __name__ == "__main__":
    main()
