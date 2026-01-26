#!/usr/bin/env python3
"""
Compute stage awareness metrics for stage detector validation.

The stage_awareness metric (0-2 scale) measures whether responses correctly
identify and adapt to the research stage.
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_stage_awareness_scores(analysis_dir: str) -> dict:
    """Load stage_awareness scores from judges.json files."""
    analysis_path = Path(analysis_dir)
    scores = defaultdict(lambda: defaultdict(list))  # system -> stage -> [scores]
    
    system_dirs = {
        "mentor": "mentor",
        "gpt5": "gpt-5-baseline", 
        "claude": "sonnet-4.5-baseline",
        "gemini": "gemini-baseline"
    }
    
    for system_key, dir_name in system_dirs.items():
        system_path = analysis_path / dir_name
        
        if not system_path.exists():
            continue
            
        for judges_file in system_path.glob("**/*_judges.json"):
            try:
                with open(judges_file, 'r') as f:
                    data = json.load(f)
                
                prompt_id = data.get('prompt_id', '')
                stage_awareness = data.get('metrics', {}).get('stage_awareness', {})
                score = stage_awareness.get('score', None)
                
                # Extract stage from prompt_id
                parts = prompt_id.split("_")
                stage = parts[1].upper() if len(parts) >= 2 else "?"
                
                if score is not None:
                    scores[system_key][stage].append({
                        "prompt_id": prompt_id,
                        "score": score,
                        "max_possible": 2.0
                    })
                    
            except Exception as e:
                pass
    
    return scores


def compute_stage_awareness_summary(scores: dict) -> dict:
    """Compute summary statistics for stage awareness."""
    summary = {}
    
    for system, stage_scores in scores.items():
        summary[system] = {
            "by_stage": {},
            "overall": {}
        }
        
        all_scores = []
        
        for stage in ['A', 'B', 'C', 'D', 'E', 'F']:
            stage_list = [s["score"] for s in stage_scores.get(stage, [])]
            all_scores.extend(stage_list)
            
            if stage_list:
                # Count perfect scores (2.0), good scores (1.5+), low scores (<1.0)
                perfect = sum(1 for s in stage_list if s >= 1.95)
                good = sum(1 for s in stage_list if s >= 1.5)
                low = sum(1 for s in stage_list if s < 1.0)
                
                summary[system]["by_stage"][stage] = {
                    "n": len(stage_list),
                    "mean": round(statistics.mean(stage_list), 3),
                    "std": round(statistics.stdev(stage_list), 3) if len(stage_list) > 1 else 0,
                    "min": round(min(stage_list), 2),
                    "max": round(max(stage_list), 2),
                    "perfect_rate": round(perfect / len(stage_list), 3),
                    "good_rate": round(good / len(stage_list), 3),
                    "low_rate": round(low / len(stage_list), 3),
                }
        
        if all_scores:
            perfect_all = sum(1 for s in all_scores if s >= 1.95)
            good_all = sum(1 for s in all_scores if s >= 1.5)
            low_all = sum(1 for s in all_scores if s < 1.0)
            
            summary[system]["overall"] = {
                "n": len(all_scores),
                "mean": round(statistics.mean(all_scores), 3),
                "std": round(statistics.stdev(all_scores), 3) if len(all_scores) > 1 else 0,
                "min": round(min(all_scores), 2),
                "max": round(max(all_scores), 2),
                "perfect_rate": round(perfect_all / len(all_scores), 3),
                "good_rate": round(good_all / len(all_scores), 3),
                "low_rate": round(low_all / len(all_scores), 3),
            }
    
    return summary


def find_misclassifications(scores: dict) -> list:
    """Find cases where stage awareness is low (<1.5) - potential misclassifications."""
    issues = []
    
    for system, stage_scores in scores.items():
        for stage, prompt_scores in stage_scores.items():
            for item in prompt_scores:
                if item["score"] < 1.5:
                    issues.append({
                        "system": system,
                        "stage": stage,
                        "prompt_id": item["prompt_id"],
                        "score": item["score"],
                        "severity": "low" if item["score"] >= 1.0 else "critical"
                    })
    
    return sorted(issues, key=lambda x: x["score"])


def generate_report(summary: dict, issues: list, output_path: str):
    """Generate markdown report."""
    
    lines = [
        "# Stage Awareness Validation Report",
        "",
        f"**Generated:** 2026-01-24",
        "",
        "The `stage_awareness` metric (0-2 scale) measures whether responses correctly",
        "identify and adapt to the research stage (A-F).",
        "",
        "---",
        "",
        "## 1. Overall Stage Awareness by System",
        "",
        "| System | n | Mean | Std | Perfect (≥1.95) | Good (≥1.5) | Low (<1.0) |",
        "|--------|---|------|-----|-----------------|-------------|------------|",
    ]
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = summary.get(system, {}).get('overall', {})
        lines.append(
            f"| {system.upper()} | {data.get('n', 0)} | "
            f"{data.get('mean', 0):.2f} | ±{data.get('std', 0):.2f} | "
            f"{data.get('perfect_rate', 0)*100:.0f}% | "
            f"{data.get('good_rate', 0)*100:.0f}% | "
            f"{data.get('low_rate', 0)*100:.0f}% |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. Stage Awareness by Stage",
        "",
    ])
    
    # Stage definitions
    stage_names = {
        'A': 'Pre-idea (Orientation)',
        'B': 'Idea (Feasibility)',
        'C': 'Research Plan',
        'D': 'First Draft (Methodology)',
        'E': 'Second Draft (Discussion)',
        'F': 'Final (Venue/Release)'
    }
    
    for stage in ['A', 'B', 'C', 'D', 'E', 'F']:
        lines.append(f"### Stage {stage}: {stage_names[stage]}")
        lines.append("")
        lines.append("| System | n | Mean | Good Rate | Low Rate |")
        lines.append("|--------|---|------|-----------|----------|")
        
        for system in ['mentor', 'gemini', 'gpt5', 'claude']:
            data = summary.get(system, {}).get('by_stage', {}).get(stage, {})
            lines.append(
                f"| {system.upper()} | {data.get('n', 0)} | "
                f"{data.get('mean', 0):.2f} | "
                f"{data.get('good_rate', 0)*100:.0f}% | "
                f"{data.get('low_rate', 0)*100:.0f}% |"
            )
        lines.append("")
    
    # Potential misclassifications
    lines.extend([
        "---",
        "",
        "## 3. Potential Stage Misclassifications",
        "",
        f"Found **{len(issues)}** responses with stage_awareness < 1.5:",
        "",
    ])
    
    if issues:
        critical = [i for i in issues if i["severity"] == "critical"]
        low = [i for i in issues if i["severity"] == "low"]
        
        lines.append(f"- **Critical (< 1.0):** {len(critical)} cases")
        lines.append(f"- **Low (1.0-1.5):** {len(low)} cases")
        lines.append("")
        
        # Show worst cases
        lines.append("### Worst Cases (score < 1.0)")
        lines.append("")
        if critical:
            lines.append("| System | Stage | Prompt | Score |")
            lines.append("|--------|-------|--------|-------|")
            for issue in critical[:10]:  # Top 10
                lines.append(
                    f"| {issue['system']} | {issue['stage']} | "
                    f"{issue['prompt_id']} | {issue['score']:.2f} |"
                )
        else:
            lines.append("*No critical cases found.*")
    else:
        lines.append("*No issues found.*")
    
    lines.extend([
        "",
        "---",
        "",
        "## 4. Key Findings",
        "",
    ])
    
    # Compare systems
    mentor_mean = summary.get('mentor', {}).get('overall', {}).get('mean', 0)
    mentor_issues = len([i for i in issues if i['system'] == 'mentor'])
    
    lines.extend([
        f"- **MENTOR** achieves {mentor_mean:.2f}/2.0 mean stage awareness",
        f"- MENTOR has **{mentor_issues}** responses with low stage awareness",
    ])
    
    # Which stages are hardest?
    mentor_by_stage = summary.get('mentor', {}).get('by_stage', {})
    stage_means = [(s, mentor_by_stage.get(s, {}).get('mean', 0)) for s in ['A', 'B', 'C', 'D', 'E', 'F']]
    stage_means_sorted = sorted(stage_means, key=lambda x: x[1])
    
    lines.append(f"- Hardest stage for MENTOR: **{stage_means_sorted[0][0]}** ({stage_means_sorted[0][1]:.2f})")
    lines.append(f"- Easiest stage for MENTOR: **{stage_means_sorted[-1][0]}** ({stage_means_sorted[-1][1]:.2f})")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    base_dir = Path(__file__).parent
    analysis_dir = base_dir / "analysis_reports"
    
    print("=" * 70)
    print("STAGE AWARENESS VALIDATION")
    print("=" * 70)
    
    # Load scores
    print("\n[1/3] Loading stage_awareness scores...")
    scores = load_stage_awareness_scores(str(analysis_dir))
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        total = sum(len(v) for v in scores[system].values())
        print(f"  {system.upper()}: {total} prompts")
    
    # Compute summary
    print("\n[2/3] Computing summary statistics...")
    summary = compute_stage_awareness_summary(scores)
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = summary[system]["overall"]
        print(f"  {system.upper()}: {data['mean']:.2f}/2.0 mean, {data['low_rate']*100:.0f}% low")
    
    # Find issues
    print("\n[3/3] Finding potential misclassifications...")
    issues = find_misclassifications(scores)
    print(f"  Found {len(issues)} responses with score < 1.5")
    
    # Save results
    output_json = base_dir / "stage_awareness_results.json"
    with open(output_json, 'w') as f:
        json.dump({"summary": summary, "issues": issues}, f, indent=2)
    print(f"\n✓ JSON saved to: {output_json}")
    
    # Generate report
    output_md = base_dir / "stage_awareness_report.md"
    generate_report(summary, issues, str(output_md))
    print(f"✓ Report saved to: {output_md}")
    
    # Print stage-by-stage for MENTOR
    print("\n" + "=" * 70)
    print("MENTOR STAGE AWARENESS BY STAGE")
    print("=" * 70)
    print(f"\n{'Stage':<8} {'Mean':<8} {'Good%':<8} {'Low%':<8}")
    print("-" * 32)
    for stage in ['A', 'B', 'C', 'D', 'E', 'F']:
        data = summary['mentor']['by_stage'].get(stage, {})
        print(f"{stage:<8} {data.get('mean', 0):<8.2f} {data.get('good_rate', 0)*100:<8.0f} {data.get('low_rate', 0)*100:<8.0f}")


if __name__ == "__main__":
    main()
