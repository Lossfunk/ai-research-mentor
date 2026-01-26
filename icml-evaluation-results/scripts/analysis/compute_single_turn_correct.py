#!/usr/bin/env python3
"""
Correctly compute single-turn pairwise wins using holistic_score from judges.json files.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_holistic_scores(analysis_dir: str) -> dict:
    """Load holistic_score from judges.json files."""
    analysis_path = Path(analysis_dir)
    scores = {}
    
    system_dirs = {
        "mentor": "mentor",
        "gpt5": "gpt-5-baseline", 
        "claude": "sonnet-4.5-baseline",
        "gemini": "gemini-baseline"
    }
    
    for system_key, dir_name in system_dirs.items():
        scores[system_key] = {}
        system_path = analysis_path / dir_name
        
        if not system_path.exists():
            print(f"  Warning: {system_path} not found")
            continue
            
        # Search for judges.json files in all stage directories
        for stage_dir in system_path.glob("stage_*"):
            if not stage_dir.is_dir():
                continue
                
            # Find judges.json files in subdirectories
            for judges_file in stage_dir.glob("**/*_judges.json"):
                try:
                    with open(judges_file, 'r') as f:
                        data = json.load(f)
                    
                    prompt_id = data.get('prompt_id', '')
                    holistic = data.get('metrics', {}).get('holistic_score', {})
                    score = holistic.get('score', None)
                    
                    if prompt_id and score is not None:
                        scores[system_key][prompt_id] = score
                except Exception as e:
                    print(f"  Error reading {judges_file}: {e}")
    
    return scores


def compute_stage_averages(scores: dict) -> dict:
    """Compute average holistic score per stage per system."""
    results = {}
    
    for system, prompt_scores in scores.items():
        results[system] = {}
        stage_scores = defaultdict(list)
        
        for prompt_id, score in prompt_scores.items():
            # Extract stage from prompt_id (e.g., "stage_a_01" -> "A")
            parts = prompt_id.split("_")
            if len(parts) >= 2:
                stage = parts[1].upper()
                stage_scores[stage].append(score)
        
        for stage, stage_list in stage_scores.items():
            results[system][stage] = {
                "avg": round(sum(stage_list) / len(stage_list), 3) if stage_list else 0,
                "count": len(stage_list),
                "scores": stage_list
            }
        
        # Overall average
        all_scores = list(prompt_scores.values())
        results[system]["overall"] = {
            "avg": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            "count": len(all_scores)
        }
    
    return results


def determine_winner(mentor_score: float, baseline_score: float, threshold: float = 0.05) -> str:
    """Determine winner with tie threshold."""
    diff = mentor_score - baseline_score
    if abs(diff) < threshold:
        return "tie"
    elif diff > 0:
        return "mentor"
    else:
        return "baseline"


def compute_pairwise_by_stage(scores: dict) -> dict:
    """Compute pairwise wins by comparing same prompts."""
    results = {}
    mentor_scores = scores.get('mentor', {})
    
    for baseline_key in ['gpt5', 'claude', 'gemini']:
        baseline_scores = scores.get(baseline_key, {})
        matchup = f"mentor_vs_{baseline_key}"
        results[matchup] = {
            "by_stage": defaultdict(lambda: {"mentor": 0, "baseline": 0, "tie": 0}),
            "overall": {"mentor": 0, "baseline": 0, "tie": 0},
            "details": []
        }
        
        common_prompts = set(mentor_scores.keys()) & set(baseline_scores.keys())
        
        for prompt_id in sorted(common_prompts):
            m_score = mentor_scores[prompt_id]
            b_score = baseline_scores[prompt_id]
            
            # Extract stage
            parts = prompt_id.split("_")
            stage = parts[1].upper() if len(parts) >= 2 else "?"
            
            winner = determine_winner(m_score, b_score)
            
            results[matchup]["by_stage"][stage][winner if winner != "baseline" else "baseline"] += 1
            results[matchup]["overall"][winner if winner != "baseline" else "baseline"] += 1
            
            results[matchup]["details"].append({
                "prompt_id": prompt_id,
                "stage": stage,
                "mentor": m_score,
                "baseline": b_score,
                "winner": winner
            })
    
    return results


def main():
    base_dir = Path(__file__).parent
    analysis_dir = base_dir / "analysis_reports"
    
    print("=" * 70)
    print("SINGLE-TURN HOLISTIC SCORE ANALYSIS (CORRECTED)")
    print("=" * 70)
    
    # Load scores
    print("\n[1/3] Loading holistic_score from judges.json files...")
    scores = load_holistic_scores(str(analysis_dir))
    
    for system, prompt_scores in scores.items():
        print(f"  {system}: {len(prompt_scores)} prompts loaded")
    
    # Compute stage averages
    print("\n[2/3] Computing stage averages...")
    stage_avgs = compute_stage_averages(scores)
    
    print("\n" + "=" * 70)
    print("STAGE AVERAGES BY SYSTEM")
    print("=" * 70)
    
    stages = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Header
    print(f"\n{'Stage':<8}", end="")
    for system in ['mentor', 'gpt5', 'claude', 'gemini']:
        print(f"{system.upper():<12}", end="")
    print("Winner")
    print("-" * 60)
    
    for stage in stages:
        print(f"{stage:<8}", end="")
        stage_scores = {}
        for system in ['mentor', 'gpt5', 'claude', 'gemini']:
            avg = stage_avgs.get(system, {}).get(stage, {}).get('avg', 0)
            stage_scores[system] = avg
            print(f"{avg:<12.3f}", end="")
        
        # Find winner
        max_score = max(stage_scores.values())
        winners = [s for s, v in stage_scores.items() if abs(v - max_score) < 0.02]
        if len(winners) > 1:
            print(f"~Tie ({'/'.join(winners)})")
        else:
            print(f"**{winners[0].upper()}**")
    
    print("-" * 60)
    print(f"{'Overall':<8}", end="")
    overall_scores = {}
    for system in ['mentor', 'gpt5', 'claude', 'gemini']:
        avg = stage_avgs.get(system, {}).get('overall', {}).get('avg', 0)
        overall_scores[system] = avg
        print(f"{avg:<12.3f}", end="")
    
    # Overall winner
    max_score = max(overall_scores.values())
    winners = [s for s, v in overall_scores.items() if abs(v - max_score) < 0.02]
    if len(winners) > 1:
        print(f"~Tie ({'/'.join(winners)})")
    else:
        print(f"**{winners[0].upper()}**")
    
    # Compute pairwise
    print("\n[3/3] Computing pairwise wins...")
    pairwise = compute_pairwise_by_stage(scores)
    
    print("\n" + "=" * 70)
    print("PAIRWISE WIN RATES (MENTOR vs BASELINE)")
    print("=" * 70)
    
    for matchup, data in pairwise.items():
        total = data["overall"]["mentor"] + data["overall"]["baseline"] + data["overall"]["tie"]
        decisive = data["overall"]["mentor"] + data["overall"]["baseline"]
        win_rate = data["overall"]["mentor"] / decisive if decisive > 0 else 0
        
        print(f"\n{matchup}:")
        print(f"  Overall: {data['overall']['mentor']} wins, {data['overall']['baseline']} losses, {data['overall']['tie']} ties")
        print(f"  Win rate: {win_rate*100:.1f}% ({data['overall']['mentor']}/{decisive})")
        
        print(f"\n  By stage:")
        for stage in stages:
            stage_data = data["by_stage"].get(stage, {"mentor": 0, "baseline": 0, "tie": 0})
            stage_decisive = stage_data["mentor"] + stage_data["baseline"]
            stage_rate = stage_data["mentor"] / stage_decisive if stage_decisive > 0 else 0
            print(f"    {stage}: {stage_data['mentor']}-{stage_data['baseline']}-{stage_data['tie']} ({stage_rate*100:.0f}%)")
    
    # Final rankings
    print("\n" + "=" * 70)
    print("FINAL RANKINGS")
    print("=" * 70)
    
    sorted_systems = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (system, score) in enumerate(sorted_systems, 1):
        delta = score - overall_scores['mentor'] if system != 'mentor' else 0
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}" if delta < 0 else "-"
        print(f"  {rank}. **{system.upper()}**: {score:.3f} (Δ vs MENTOR: {delta_str})")
    
    # Save results
    output = {
        "stage_averages": stage_avgs,
        "pairwise": {k: {"overall": v["overall"], "by_stage": dict(v["by_stage"])} for k, v in pairwise.items()},
        "rankings": sorted_systems
    }
    
    output_file = base_dir / "single_turn_holistic_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
