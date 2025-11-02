#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

base_path = Path("evals-for-papers/results/analysis_reports")
stages = ["stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "stage_f"]

results = {}

# For stages A-E with lofo_summary.json files
for stage in stages[:5]:
    results[stage] = {}
    stage_path = base_path / stage
    
    for judge_dir in stage_path.iterdir():
        if judge_dir.is_dir() and "student_outcome_judge" in judge_dir.name:
            lofo_file = judge_dir / "lofo_summary.json"
            if lofo_file.exists():
                with open(lofo_file) as f:
                    data = json.load(f)
                    judge_name = judge_dir.name
                    results[stage][judge_name] = data.get("systems", {})

# For stage F (different structure - aggregate from individual files)
stage_f_path = base_path / "stage_f" / "student_outcome_judge"
if stage_f_path.exists():
    if "stage_f" not in results:
        results["stage_f"] = {}
    all_metrics = defaultdict(lambda: defaultdict(list))
    
    for json_file in sorted(stage_f_path.glob("stage_f_*.json")):
        with open(json_file) as f:
            data = json.load(f)
            student_metrics = data.get("student_metrics", {})
            for key, value in student_metrics.items():
                all_metrics[key]["values"].append(value)
    
    # Compute means for stage F
    stage_f_agg = {}
    for metric_name, metric_data in all_metrics.items():
        values = metric_data["values"]
        if values:
            stage_f_agg[metric_name] = {
                "mean": sum(values) / len(values),
                "count": len(values)
            }
    
    results["stage_f"]["aggregated"] = stage_f_agg

# Print comprehensive summary
print("=" * 80)
print("STUDENT JUDGE RESULTS SUMMARY (All Stages A-F)")
print("=" * 80)

metrics_to_track = [
    "student_actionability",
    "student_clarity", 
    "student_constraint_fit",
    "student_confidence_gain",
    "student_path_ready",
    "student_failure_modes",
]

for stage in stages:
    print(f"\n{stage.upper()} RESULTS")
    print("-" * 80)
    
    if stage in results and results[stage]:
        if stage == "stage_f":
            agg = results.get("stage_f", {}).get("aggregated", {})
            if agg:
                print(f"\n  Stage F aggregated (from individual evaluations):")
                for metric in metrics_to_track:
                    if metric in agg:
                        m_data = agg[metric]
                        mean_val = m_data.get("mean", "N/A")
                        count = m_data.get("count", "N/A")
                        if isinstance(mean_val, (int, float)):
                            print(f"    {metric}: {mean_val:.3f} (n={count})")
        else:
            for judge_type, systems in results[stage].items():
                if systems:  # Check if systems exist
                    for system_name, system_data in systems.items():
                        print(f"\n  {judge_type} → {system_name}:")
                        metrics = system_data.get("metrics", {})
                        
                        for metric in metrics_to_track:
                            if metric in metrics:
                                m = metrics[metric]
                                all_data = m.get("all", {})
                                mean_val = all_data.get("mean", "N/A")
                                ci_low = all_data.get("ci_low", "N/A")
                                ci_high = all_data.get("ci_high", "N/A")
                                count = all_data.get("count", "N/A")
                                
                                if isinstance(mean_val, (int, float)):
                                    print(f"    {metric}: {mean_val:.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f}, n={count})")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)

# Cross-stage comparison for composite scores
print("\nCross-Stage Trends (Mentor Manual Judge - Composite Scores):")
mentor_composite_by_stage = {}
judge_compositions = ["mentor_manual", "openrouter_openai_gpt_5", "openrouter_anthropic_claude_sonnet_4_5"]

for stage in stages[:5]:
    if stage not in results:
        continue
    
    stage_data = results.get(stage, {})
    for judge_type, systems in stage_data.items():
        if systems:
            for system_name, system_data in systems.items():
                if system_name == "mentor_manual":
                    metrics = system_data.get("metrics", {})
                    # Try to find composite score
                    if "student_outcome_score" in metrics:
                        composite = metrics["student_outcome_score"].get("all", {}).get("mean")
                        if composite:
                            if judge_type not in mentor_composite_by_stage:
                                mentor_composite_by_stage[judge_type] = {}
                            mentor_composite_by_stage[judge_type][stage] = composite

# Print composite trends
for judge_type in mentor_composite_by_stage:
    trends = mentor_composite_by_stage[judge_type]
    if len(trends) > 2:
        print(f"\n  {judge_type}:")
        for stage in sorted(trends.keys()):
            print(f"    {stage.upper()}: {trends[stage]:.3f}")
        
        stages_ordered = sorted(trends.keys())
        if len(stages_ordered) > 1:
            first_val = trends[stages_ordered[0]]
            last_val = trends[stages_ordered[-1]]
            trend_pct = ((last_val - first_val) / first_val * 100)
            print(f"    Trend: {trend_pct:+.1f}% from {stages_ordered[0].upper()} to {stages_ordered[-1].upper()}")

# Judge model comparison on stage A metrics
print("\nJudge Model Comparison (Stage A - Primary Metrics):")
if "stage_a" in results:
    comparisons = {}
    for judge_type, systems in results["stage_a"].items():
        for system_name, system_data in systems.items():
            metrics = system_data.get("metrics", {})
            if "student_actionability" in metrics:
                action = metrics["student_actionability"].get("all", {}).get("mean")
                clarity = metrics["student_clarity"].get("all", {}).get("mean")
                composite = metrics.get("student_outcome_score", {}).get("all", {}).get("mean")
                if action and clarity:
                    key = f"{judge_type} → {system_name}"
                    comparisons[key] = {"actionability": action, "clarity": clarity, "composite": composite}
    
    for judge_key in sorted(comparisons.keys()):
        scores = comparisons[judge_key]
        print(f"  {judge_key}:")
        print(f"    Actionability: {scores['actionability']:.3f}, Clarity: {scores['clarity']:.3f}, Composite: {scores.get('composite', 'N/A')}")
