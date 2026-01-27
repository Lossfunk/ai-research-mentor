#!/usr/bin/env python3
"""
Run judges on 90-prompt ablation responses.

This scores the ablation responses using the same 3-judge ensemble
used for the main evaluation.

Usage:
    # Score all ablation conditions:
    uv run python icml-evaluation-results/ablations/run_ablation_judges.py

    # Score only one condition:
    uv run python icml-evaluation-results/ablations/run_ablation_judges.py --condition no_stage

    # Score specific stages:
    uv run python icml-evaluation-results/ablations/run_ablation_judges.py --stage a --stage b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

try:
    from academic_research_mentor.cli.session import load_env_file
except ImportError:
    def load_env_file() -> None:
        pass  # no .env when run outside project

# =============================================================================
# PATHS
# =============================================================================
ABLATION_DIR = Path(__file__).parent
OUTPUT_DIR = ABLATION_DIR / "ablations_90"
JUDGES_DIR = ABLATION_DIR.parent / "judges"
HOLISTIC_PROMPT = JUDGES_DIR / "single_turn_holistic_prompt.md"

# =============================================================================
# JUDGE CONFIG (align with main eval 3-judge ensemble)
# =============================================================================
JUDGE_MODELS = [
    "qwen/qwen3-max",
    "deepseek/deepseek-v3.2-exp",
    "x-ai/grok-4-fast",
]


def load_judge_prompt() -> str:
    """Load the holistic judge prompt template."""
    if not HOLISTIC_PROMPT.exists():
        raise FileNotFoundError(f"Judge prompt not found: {HOLISTIC_PROMPT}")
    return HOLISTIC_PROMPT.read_text()


def build_judge_prompt(
    template: str,
    user_prompt: str,
    response: str,
    stage: str,
    persona_card: str = "Student seeking research guidance.",
    task_card: str = "Single-turn research query.",
) -> str:
    """Substitute template placeholders. Matches single_turn_holistic_prompt.md."""
    stage_labels = {
        "A": "A: Orientation",
        "B": "B: Novelty/Hypothesis",
        "C": "C: Research Planning",
        "D": "D: Methodology",
        "E": "E: Implementation",
        "F": "F: Writing/Submission",
    }
    stage_label = stage_labels.get(stage.upper(), stage)
    return template.replace("{user_query}", user_prompt).replace(
        "{system_response}", response
    ).replace("{stage}", stage_label).replace(
        "{persona_card}", persona_card
    ).replace("{task_card}", task_card)


def get_judge_client(model: str) -> OpenAI:
    """Create OpenRouter client for a judge model."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _persona_task_from_meta(metadata: Dict[str, Any]) -> tuple:
    """Build persona_card and task_card from prompt metadata."""
    m = metadata or {}
    parts = [m.get("persona", "student"), m.get("domain", ""), m.get("background", "")]
    persona = " ".join(str(p).strip() for p in parts if p).strip() or "Student seeking research guidance."
    task = str(m.get("focus", "") or m.get("constraint", "") or "Single-turn research query.").strip() or "Single-turn research query."
    return (persona, task)


def call_judge(
    client: OpenAI,
    model: str,
    template: str,
    user_prompt: str,
    response: str,
    stage: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call a single judge model using the holistic template."""

    persona_card, task_card = _persona_task_from_meta(metadata)
    user_message = build_judge_prompt(
        template, user_prompt, response, stage, persona_card, task_card
    )
    system_message = (
        "You are a critical evaluator. Follow the instructions exactly. "
        "Return ONLY valid JSON with no markdown, no extra text."
    )

    try:
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        raw = result.choices[0].message.content or ""

        # Parse JSON from response (template uses 'score'; we treat as holistic_score)
        scores = parse_judge_response(raw)

        return {
            "model": model,
            "raw": raw,
            "scores": scores,
            "usage": {
                "input": result.usage.prompt_tokens if result.usage else 0,
                "output": result.usage.completion_tokens if result.usage else 0,
            }
        }

    except Exception as e:
        return {
            "model": model,
            "error": str(e),
            "scores": None,
        }


def parse_judge_response(raw: str) -> Optional[Dict[str, float]]:
    """Parse scores from judge response. Template uses 'score'; we map to holistic_score."""
    import re

    def _normalize(parsed: dict) -> dict:
        out = {}
        if "holistic_score" in parsed:
            out["holistic_score"] = float(parsed["holistic_score"])
        elif "score" in parsed:
            out["holistic_score"] = float(parsed["score"])
        if "stage_awareness" in parsed:
            out["stage_awareness"] = float(parsed["stage_awareness"])
        return out if out else None

    # Try to find JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                n = _normalize(parsed)
                if n:
                    return n
        except json.JSONDecodeError:
            pass

    # Try to parse entire response as JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            n = _normalize(parsed)
            if n:
                return n
    except json.JSONDecodeError:
        pass

    # Regex fallback
    scores = {}
    m = re.search(r'holistic_score["\s:]+(\d+\.?\d*)', raw)
    if not m:
        m = re.search(r'"score"["\s:]+(\d+\.?\d*)', raw)
    if m:
        scores["holistic_score"] = float(m.group(1))
    m = re.search(r'stage_awareness["\s:]+(\d+\.?\d*)', raw)
    if m:
        scores["stage_awareness"] = float(m.group(1))

    return scores if scores else None


def aggregate_judge_scores(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scores from multiple judges."""
    holistic_scores = []
    stage_scores = []

    for result in judge_results:
        if result.get("scores"):
            if "holistic_score" in result["scores"]:
                holistic_scores.append(result["scores"]["holistic_score"])
            if "stage_awareness" in result["scores"]:
                stage_scores.append(result["scores"]["stage_awareness"])

    return {
        "holistic_score": sum(holistic_scores) / len(holistic_scores) if holistic_scores else None,
        "stage_awareness": sum(stage_scores) / len(stage_scores) if stage_scores else None,
        "n_judges": len(judge_results),
        "n_valid": len(holistic_scores),
    }


def score_condition(
    condition: str,
    stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score all responses for an ablation condition."""

    condition_dir = OUTPUT_DIR / condition
    raw_logs_dir = condition_dir / "raw_logs"
    scores_dir = condition_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    if not raw_logs_dir.exists():
        print(f"  Error: No responses found at {raw_logs_dir}")
        return {"error": "no_responses"}

    # Load judge template
    template = load_judge_prompt()

    # Create judge clients
    clients = [(model, get_judge_client(model)) for model in JUDGE_MODELS]

    # Find all meta files
    meta_files = sorted(raw_logs_dir.glob("*_meta.json"))

    if stages:
        stages_upper = [s.upper() for s in stages]
        meta_files = [f for f in meta_files if json.loads(f.read_text()).get("stage", "").upper() in stages_upper]

    # Check for existing scores (resume mode) - only skip if they have valid scores
    existing_valid_scores = set()
    existing_invalid_scores = []
    for score_file in scores_dir.glob("*_scores.json"):
        prompt_id = score_file.stem.replace("_scores", "")
        try:
            existing_data = json.loads(score_file.read_text())
            agg = existing_data.get("aggregated", {})
            holistic_score = agg.get("holistic_score")
            # Only skip if we have a valid (non-null) holistic score
            if holistic_score is not None:
                existing_valid_scores.add(prompt_id)
            else:
                existing_invalid_scores.append(prompt_id)
        except Exception:
            # If we can't parse it, treat as invalid and re-judge
            existing_invalid_scores.append(prompt_id)

    if existing_valid_scores:
        print(f"  Found {len(existing_valid_scores)} valid scores (will skip)")
    if existing_invalid_scores:
        print(f"  Found {len(existing_invalid_scores)} invalid/error scores (will re-judge)")

    print(f"\n  Scoring {len(meta_files)} responses with {len(JUDGE_MODELS)} judges...")

    results = []
    skipped = 0

    for i, meta_path in enumerate(meta_files):
        meta = json.loads(meta_path.read_text())
        prompt_id = meta["prompt_id"]
        stage = meta["stage"]
        metadata = meta.get("metadata") or {}

        # Skip if already has valid score
        if prompt_id in existing_valid_scores:
            skipped += 1
            # Load existing score for summary
            score_path = scores_dir / f"{prompt_id}_scores.json"
            if score_path.exists():
                try:
                    existing_data = json.loads(score_path.read_text())
                    agg = existing_data.get("aggregated", {})
                    results.append({
                        "prompt_id": prompt_id,
                        "stage": stage,
                        "holistic_score": agg.get("holistic_score"),
                        "stage_awareness": agg.get("stage_awareness"),
                    })
                except Exception:
                    pass
            continue

        # If it's an invalid score file, we'll overwrite it below

        # Load response
        response_path = Path(meta["response_path"])
        if not response_path.exists():
            print(f"  [{i+1}/{len(meta_files)}] {prompt_id} - MISSING RESPONSE: {response_path}")
            continue

        response = response_path.read_text().strip()
        if not response:
            print(f"  [{i+1}/{len(meta_files)}] {prompt_id} - EMPTY RESPONSE (skipped)")
            continue

        # Load original prompt (from prompts file)
        user_prompt = get_original_prompt(prompt_id)
        if not user_prompt:
            print(f"  [{i+1}/{len(meta_files)}] {prompt_id} - MISSING PROMPT")
            continue

        print(f"  [{i+1}/{len(meta_files)}] {prompt_id} (stage {stage})...", end=" ", flush=True)

        # Call all judges
        judge_results = []
        for model, client in clients:
            result = call_judge(
                client, model, template, user_prompt, response, stage, metadata
            )
            judge_results.append(result)

        # Aggregate scores
        aggregated = aggregate_judge_scores(judge_results)

        # Save detailed results
        score_data = {
            "prompt_id": prompt_id,
            "stage": stage,
            "condition": condition,
            "aggregated": aggregated,
            "judges": judge_results,
            "timestamp": datetime.now().isoformat(),
        }

        score_path = scores_dir / f"{prompt_id}_scores.json"
        score_path.write_text(json.dumps(score_data, indent=2))

        results.append({
            "prompt_id": prompt_id,
            "stage": stage,
            "holistic_score": aggregated["holistic_score"],
            "stage_awareness": aggregated.get("stage_awareness"),
        })

        hs = aggregated["holistic_score"]
        sa = aggregated.get("stage_awareness")
        sa_str = f"{sa:.2f}" if sa is not None else "—"
        print(f"holistic={hs:.2f}, stage_aware={sa_str}" if hs is not None else "ERROR")

    # Compute summary statistics
    valid_results = [r for r in results if r["holistic_score"] is not None]
    valid_with_sa = [r for r in valid_results if r.get("stage_awareness") is not None]

    mean_holistic = (
        sum(r["holistic_score"] for r in valid_results) / len(valid_results)
        if valid_results else None
    )
    mean_stage_awareness = (
        sum(r["stage_awareness"] for r in valid_with_sa) / len(valid_with_sa)
        if valid_with_sa else None
    )

    summary = {
        "condition": condition,
        "n_total": len(meta_files),
        "n_scored": len(valid_results),
        "n_skipped": skipped,
        "mean_holistic_score": mean_holistic,
        "mean_stage_awareness": mean_stage_awareness,
        "by_stage": {},
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    # Compute by-stage statistics
    for st in ["A", "B", "C", "D", "E", "F"]:
        stage_results = [r for r in valid_results if r["stage"].upper() == st]
        if stage_results:
            st_sa = [r for r in stage_results if r.get("stage_awareness") is not None]
            summary["by_stage"][st] = {
                "n": len(stage_results),
                "mean_holistic": sum(r["holistic_score"] for r in stage_results) / len(stage_results),
                "mean_stage_awareness": (
                    sum(r["stage_awareness"] for r in st_sa) / len(st_sa) if st_sa else None
                ),
            }

    # Save summary
    summary_path = condition_dir / "scores_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    h_str = f"{mean_holistic:.3f}" if mean_holistic is not None else "—"
    sa_summary = f"{mean_stage_awareness:.3f}" if mean_stage_awareness is not None else "—"
    print(f"\n  Summary: holistic={h_str}, stage_awareness={sa_summary}")
    if skipped > 0:
        print(f"  Skipped {skipped} already-scored prompts")

    return summary


def get_original_prompt(prompt_id: str) -> Optional[str]:
    """Get the original prompt text from the prompts file."""
    prompts_file = ABLATION_DIR.parent / "prompts" / "evals_single_turn.jsonl"

    with open(prompts_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data["prompt_id"] == prompt_id:
                    return data["prompt"]
    return None


def compute_ablation_comparison() -> Dict[str, Any]:
    """Compute ablation comparison against full MENTOR baseline."""

    baseline_path = ABLATION_DIR.parent / "results" / "single_turn_holistic_results.json"

    if not baseline_path.exists():
        print(f"  Warning: Baseline not found at {baseline_path}")
        result = {
            "experiment": "90-Prompt Ablation Study",
            "baseline": None,
            "ablations": {},
            "error": "baseline_not_found",
            "timestamp": datetime.now().isoformat(),
        }
        comparison_path = OUTPUT_DIR / "ablation_comparison_90.json"
        comparison_path.write_text(json.dumps(result, indent=2))
        return result

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_score = baseline["stage_averages"]["mentor"]["overall"]["avg"]

    comparisons = {}
    for condition in ["no_stage", "no_guidelines"]:
        summary_path = OUTPUT_DIR / condition / "scores_summary.json"
        if not summary_path.exists():
            print(f"  No scores for {condition} (run judges first: {summary_path})")
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        ablated_score = summary.get("mean_holistic_score")
        if ablated_score is None:
            print(f"  {condition}: no mean_holistic_score in summary")
            continue
        delta = ablated_score - baseline_score
        delta_pct = (delta / baseline_score) * 100

        comparisons[condition] = {
            "baseline_mean": baseline_score,
            "ablated_mean": ablated_score,
            "delta": delta,
            "delta_pct": delta_pct,
            "n_scored": summary.get("n_scored"),
        }

    result = {
        "experiment": "90-Prompt Ablation Study",
        "baseline": {
            "source": "single_turn_holistic_results.json",
            "system": "Full MENTOR",
            "n_prompts": 90,
            "holistic_score": baseline_score,
        },
        "ablations": comparisons,
        "timestamp": datetime.now().isoformat(),
    }

    comparison_path = OUTPUT_DIR / "ablation_comparison_90.json"
    comparison_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    load_env_file()

    parser = argparse.ArgumentParser(description="Run judges on ablation responses")
    parser.add_argument("--condition", choices=["no_stage", "no_guidelines"], help="Score only this condition")
    parser.add_argument("--stage", action="append", help="Score only these stages (repeatable)")
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION JUDGE SCORING")
    print("=" * 70)

    conditions = [args.condition] if args.condition else ["no_stage", "no_guidelines"]

    summaries = {}
    for condition in conditions:
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition}")
        print(f"{'='*70}")

        summary = score_condition(condition, stages=args.stage)
        summaries[condition] = summary

    # Compute comparison against baseline
    print("\n" + "=" * 70)
    print("COMPUTING ABLATION COMPARISON")
    print("=" * 70)

    comparison = compute_ablation_comparison()

    if comparison.get("ablations"):
        baseline_score = comparison["baseline"]["holistic_score"] if comparison.get("baseline") else "?"
        print(f"\nResults vs Full MENTOR (baseline {baseline_score}):")
        for condition, data in comparison["ablations"].items():
            print(f"  {condition}: {data['ablated_mean']:.3f} ({data['delta_pct']:+.1f}%)")
    else:
        print("\n  No ablation scores found. Ensure you have run judges on at least one condition.")
        print("  Check that scores_summary.json exists under no_stage/ and/or no_guidelines/.")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
