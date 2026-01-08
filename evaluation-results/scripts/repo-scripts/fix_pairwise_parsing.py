#!/usr/bin/env python3
"""
Re-parse pairwise judge outputs with improved JSON extraction.
Fixes the bug where JSON blocks embedded in prose are missed.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
from collections import Counter


def extract_json_block(raw: str) -> Optional[str]:
    """Extract JSON from code fence or find JSON object in text."""
    # Try code fence first
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try finding standalone JSON object
    match = re.search(r'\{[^{}]*"winner"[^{}]*"aspect_votes"[^{}]*\}', raw, re.DOTALL)
    if match:
        # Expand to full object (handle nested braces)
        start = match.start()
        brace_count = 0
        for i, char in enumerate(raw[start:], start=start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return raw[start:i+1]
    
    return None


def parse_pairwise_output_fixed(raw: str) -> Optional[Dict[str, Any]]:
    """Improved parser that handles JSON embedded in prose."""
    candidate = extract_json_block(raw)
    if not candidate:
        return None
    
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        print(f"  [WARN] JSON parse failed: {e}")
        return None
    
    if not isinstance(parsed, dict):
        return None
    
    winner = parsed.get("winner")
    if winner not in {"A", "B", "Tie"}:
        print(f"  [WARN] Invalid winner: {winner}, defaulting to Tie")
        parsed["winner"] = "Tie"
    
    aspect_votes = parsed.get("aspect_votes")
    if not isinstance(aspect_votes, dict):
        parsed["aspect_votes"] = {}
    
    return parsed


def reprocess_file(filepath: Path) -> bool:
    """Re-parse a single comparison file. Returns True if changed."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    
    changed = False
    for judge_output in data.get("judge_outputs", []):
        raw = judge_output.get("raw", "")
        old_parsed = judge_output.get("parsed", {})
        old_winner = old_parsed.get("winner", "Tie")
        
        new_parsed = parse_pairwise_output_fixed(raw)
        if new_parsed and new_parsed != old_parsed:
            new_winner = new_parsed.get("winner", "Tie")
            if new_winner != old_winner:
                print(f"  {filepath.stem}: {old_winner} → {new_winner}")
                changed = True
            judge_output["parsed"] = new_parsed
    
    # Recompute majority vote
    vote_counts = Counter()
    for output in data.get("judge_outputs", []):
        winner = output.get("parsed", {}).get("winner", "Tie")
        if winner in {"A", "B", "Tie"}:
            vote_counts[winner] += 1
    
    if vote_counts:
        majority_vote, majority_count = vote_counts.most_common(1)[0]
        if majority_count == vote_counts.get("Tie", 0) and majority_vote != "Tie":
            majority_vote = "Tie"
    else:
        majority_vote = "Tie"
    
    order_map = data.get("order", {})
    if majority_vote == "A":
        winning_system = order_map.get("A")
    elif majority_vote == "B":
        winning_system = order_map.get("B")
    else:
        winning_system = None
    
    old_winner_sys = data.get("winner_system_id")
    if data.get("winner") != majority_vote or old_winner_sys != winning_system:
        print(f"  Overall: {data.get('winner')} → {majority_vote}")
        changed = True
    
    data["winner"] = majority_vote
    data["winner_system_id"] = winning_system
    
    if changed:
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return changed


def rebuild_summaries(pairwise_dir: Path) -> None:
    """Rebuild summary.json files from individual comparisons."""
    for matchup_dir in pairwise_dir.iterdir():
        if not matchup_dir.is_dir() or matchup_dir.name == "pairwise":
            continue
        
        comparison_files = sorted(matchup_dir.glob("stage_*.json"))
        if not comparison_files:
            continue
        
        # Extract systems from first file
        first = json.loads(comparison_files[0].read_text(encoding="utf-8"))
        order_map = first.get("order", {})
        systems = [order_map.get("A"), order_map.get("B")]
        if not all(systems):
            continue
        
        wins = Counter()
        ties = 0
        judges_set = set()
        prompt_digest = None
        
        for cf in comparison_files:
            data = json.loads(cf.read_text(encoding="utf-8"))
            winner_sys = data.get("winner_system_id")
            if winner_sys:
                wins[winner_sys] += 1
            else:
                ties += 1
            
            for jo in data.get("judge_outputs", []):
                judges_set.add(jo.get("judge"))
            
            if prompt_digest is None:
                prompt_digest = data.get("pairwise_prompt_digest")
        
        summary = {
            "systems": systems,
            "total_comparisons": len(comparison_files),
            "wins": dict(wins),
            "ties": ties,
            "judges": sorted(judges_set),
            "pairwise_prompt_digest": prompt_digest,
            "pairwise_aspects": {
                "inquiry_quality": 1.0,
                "persona_adaptation": 1.0,
                "methodology_critique": 1.0,
                "plan_completeness": 1.0,
                "literature_quality": 1.0,
                "actionability_risks": 1.0,
                "guideline_adherence": 1.0,
            },
        }
        
        summary_path = matchup_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[REBUILT] {matchup_dir.name}")
    
    # Rebuild overall_summary.json
    all_wins = Counter()
    for matchup_dir in pairwise_dir.iterdir():
        if not matchup_dir.is_dir() or matchup_dir.name == "pairwise":
            continue
        summary_path = matchup_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for sys, count in summary.get("wins", {}).items():
            all_wins[sys] += count
    
    # Get metadata from any summary
    sample_summary_path = next((d / "summary.json" for d in pairwise_dir.iterdir() if (d / "summary.json").exists()), None)
    if sample_summary_path:
        sample = json.loads(sample_summary_path.read_text(encoding="utf-8"))
        overall = {
            "wins": dict(all_wins),
            "judges": sample.get("judges", []),
            "pairwise_prompt_digest": sample.get("pairwise_prompt_digest"),
            "pairwise_aspects": sample.get("pairwise_aspects", {}),
            "seed": 1,  # Hardcoded since all current runs are seed1
        }
        overall_path = pairwise_dir / "overall_summary.json"
        overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[REBUILT] overall_summary.json")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    analysis_root = repo_root / "evals-for-papers" / "results" / "analysis_reports"
    
    if not analysis_root.exists():
        print(f"[ERROR] Analysis dir not found: {analysis_root}")
        return 1
    
    stages = sorted(analysis_root.glob("stage_*"))
    total_changed = 0
    
    for stage_dir in stages:
        print(f"\n[STAGE] {stage_dir.name}")
        pairwise_dir = stage_dir / "judge_seed1" / "pairwise"
        if not pairwise_dir.exists():
            continue
        
        for matchup_dir in sorted(pairwise_dir.iterdir()):
            if not matchup_dir.is_dir():
                continue
            
            comparison_files = sorted(matchup_dir.glob("stage_*.json"))
            if not comparison_files:
                continue
            
            print(f"  [MATCHUP] {matchup_dir.name}")
            for cf in comparison_files:
                if reprocess_file(cf):
                    total_changed += 1
        
        # Rebuild summaries for this stage
        rebuild_summaries(pairwise_dir)
    
    print(f"\n[DONE] Fixed {total_changed} comparisons across {len(stages)} stages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

