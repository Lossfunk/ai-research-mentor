#!/usr/bin/env python3
"""
Run stage ablation experiment.

Conditions:
1. full_mentor - Use existing results (no re-run needed)
2. no_stage_prompt - Remove stage awareness from system prompt
3. no_stage_directives - Disable Stage C special instructions  
4. no_stage_all - Remove both
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


def load_selected_prompts() -> List[Dict[str, Any]]:
    """Load the selected prompts for ablation."""
    prompts_file = ABLATION_DIR / "selected_prompts.json"
    with open(prompts_file) as f:
        return json.load(f)


def load_full_mentor_results() -> Dict[str, Dict[str, Any]]:
    """Load existing Full MENTOR results for comparison."""
    results = {}
    base_dir = ABLATION_DIR.parent.parent / "analysis_reports" / "mentor"
    
    prompts = load_selected_prompts()
    
    for prompt_data in prompts:
        prompt_id = prompt_data["prompt_id"]
        stage = prompt_id.split("_")[1].lower()
        
        judges_file = base_dir / f"stage_{stage}" / f"stage_{stage}_mentor_icml" / f"{prompt_id}_judges.json"
        
        if judges_file.exists():
            with open(judges_file) as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            results[prompt_id] = {
                "holistic_score": metrics.get("holistic_score", {}).get("score"),
                "stage_awareness": metrics.get("stage_awareness", {}).get("score"),
                "actionability": metrics.get("actionability", {}).get("score"),
                "clarification_quality": metrics.get("clarification_quality", {}).get("score"),
            }
    
    return results


def load_system_prompt(use_ablated: bool = False) -> str:
    """Load the system prompt."""
    if use_ablated:
        prompt_path = ABLATION_DIR / "prompt_no_stage.md"
    else:
        prompt_path = PROJECT_ROOT / "src" / "academic_research_mentor" / "prompt.md"
    
    with open(prompt_path) as f:
        return f.read()


# Stage C directive (copied from stage_directives.py)
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
    stage: str,
    system_prompt: str,
    *,
    disable_stage_directives: bool = False,
) -> Dict[str, Any]:
    """Run a single prompt through the LLM."""
    
    # Apply stage directives if not disabled
    final_prompt = prompt_text
    if stage.upper() == "C" and not disable_stage_directives:
        final_prompt = f"{STAGE_C_DIRECTIVE}\n\n{prompt_text}"
    
    start = time.time()
    
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-thinking",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
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
        # If main content is empty but reasoning exists, use reasoning
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
    disable_stage_directives: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run a single ablation condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"  use_ablated_prompt={use_ablated_prompt}")
    print(f"  disable_stage_directives={disable_stage_directives}")
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
                stage,
                system_prompt,
                disable_stage_directives=disable_stage_directives,
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
    print("STAGE ABLATION EXPERIMENT")
    print("=" * 70)
    
    # Load prompts
    prompts = load_selected_prompts()
    print(f"\nLoaded {len(prompts)} prompts for ablation")
    
    # Load existing Full MENTOR results
    print("\n[1/4] Loading Full MENTOR baseline results...")
    full_mentor_results = load_full_mentor_results()
    print(f"  Loaded {len(full_mentor_results)} existing results")
    
    # Print baseline summary
    holistic_scores = [v.get("holistic_score", 0) for v in full_mentor_results.values() if v.get("holistic_score")]
    stage_awareness_scores = [v.get("stage_awareness", 0) for v in full_mentor_results.values() if v.get("stage_awareness")]
    
    print(f"\n  Full MENTOR baseline:")
    print(f"    holistic_score: {sum(holistic_scores)/len(holistic_scores):.3f} (n={len(holistic_scores)})")
    print(f"    stage_awareness: {sum(stage_awareness_scores)/len(stage_awareness_scores):.3f} (n={len(stage_awareness_scores)})")
    
    # Save baseline
    output_dir = ABLATION_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "full_mentor_baseline.json", "w") as f:
        json.dump(full_mentor_results, f, indent=2)
    
    print(f"\n  Saved baseline to: {output_dir / 'full_mentor_baseline.json'}")
    
    # Check if we should run ablations
    run_ablations = os.environ.get("RUN_ABLATIONS", "0") == "1"
    
    if not run_ablations:
        print("\n" + "=" * 70)
        print("ABLATION SETUP COMPLETE")
        print("=" * 70)
        print("\nTo run the actual ablation experiments:")
        print("  RUN_ABLATIONS=1 uv run python run_ablation.py")
        print("\nThis will call the LLM API and incur costs (~$10-20 estimated)")
        print("\nFiles created:")
        print(f"  - {output_dir / 'full_mentor_baseline.json'}")
        print(f"  - selected_prompts.json")
        print(f"  - prompt_no_stage.md")
        return
    
    # Create OpenRouter client
    try:
        client = get_openrouter_client()
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENROUTER_API_KEY in your environment or .env file")
        return
    
    # Run ablation conditions
    print("\n[2/4] Running -StagePrompt condition...")
    no_stage_prompt = run_ablation_condition(
        client,
        "no_stage_prompt",
        prompts,
        use_ablated_prompt=True,
        disable_stage_directives=False,
    )
    
    print("\n[3/4] Running -StageDirectives condition...")
    no_stage_directives = run_ablation_condition(
        client,
        "no_stage_directives", 
        prompts,
        use_ablated_prompt=False,
        disable_stage_directives=True,
    )
    
    print("\n[4/4] Running -AllStage condition...")
    no_stage_all = run_ablation_condition(
        client,
        "no_stage_all",
        prompts,
        use_ablated_prompt=True,
        disable_stage_directives=True,
    )
    
    # Save results
    all_results = {
        "metadata": {
            "model": "moonshotai/kimi-k2-thinking",
            "temperature": 0.0,
            "n_prompts": len(prompts),
            "conditions": ["full_mentor", "no_stage_prompt", "no_stage_directives", "no_stage_all"],
        },
        "full_mentor": full_mentor_results,
        "no_stage_prompt": no_stage_prompt,
        "no_stage_directives": no_stage_directives,
        "no_stage_all": no_stage_all,
    }
    
    with open(output_dir / "ablation_responses.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Responses saved to: {output_dir / 'ablation_responses.json'}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION RESPONSES COLLECTED")
    print("=" * 70)
    
    for condition in ["no_stage_prompt", "no_stage_directives", "no_stage_all"]:
        data = all_results[condition]
        errors = sum(1 for v in data.values() if "error" in v)
        success = len(data) - errors
        print(f"\n  {condition}:")
        print(f"    Success: {success}/{len(data)}")
        if errors:
            print(f"    Errors: {errors}")
    
    print(f"\n\nNext step: Run judges on ablation responses to get scores")
    print(f"  python run_judges.py  (to be created)")


if __name__ == "__main__":
    main()
