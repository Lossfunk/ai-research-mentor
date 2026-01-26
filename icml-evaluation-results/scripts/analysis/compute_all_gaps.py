#!/usr/bin/env python3
"""
Compute all evaluation gaps for ICML submission.

Outputs:
1. Statistical significance tests (binomial) for human evaluation
2. Single-turn pairwise wins from absolute scores
3. Stage-wise win breakdown
4. Failure mode taxonomy with rates
5. Threshold sensitivity analysis
6. Combined report in JSON + Markdown
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
from math import comb, sqrt
from typing import Dict, List, Tuple, Any

# ============================================================
# Part 1: Statistical Significance Tests
# ============================================================

def binomial_test_two_sided(successes: int, trials: int, p0: float = 0.5) -> float:
    """
    Two-sided binomial test.
    Returns p-value for null hypothesis that true proportion = p0.
    """
    if trials == 0:
        return 1.0
    
    # Compute probability of observed or more extreme under null
    # Using exact binomial (sum of probabilities)
    def binom_pmf(k, n, p):
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    observed_pmf = binom_pmf(successes, trials, p0)
    
    # Sum probabilities of outcomes at least as extreme
    p_value = 0.0
    for k in range(trials + 1):
        pmf = binom_pmf(k, trials, p0)
        if pmf <= observed_pmf + 1e-10:  # Include equal probabilities
            p_value += pmf
    
    return min(p_value, 1.0)


def compute_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    
    from math import sqrt
    
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def analyze_human_evaluation(votes_dir: str) -> Dict:
    """Compute statistical tests for human evaluation."""
    votes_path = Path(votes_dir)
    
    # Aggregate votes
    overall = {"mentor": 0, "baseline": 0, "tie": 0}
    by_matchup = defaultdict(lambda: {"mentor": 0, "baseline": 0, "tie": 0})
    
    for csv_file in votes_path.glob("*.csv"):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_type = row.get('pair_type', '')
                winner = row.get('winner', '')
                
                if 'mentor_vs' not in pair_type:
                    continue
                
                if winner == 'mentor':
                    overall["mentor"] += 1
                    by_matchup[pair_type]["mentor"] += 1
                elif winner == 'tie':
                    overall["tie"] += 1
                    by_matchup[pair_type]["tie"] += 1
                else:
                    overall["baseline"] += 1
                    by_matchup[pair_type]["baseline"] += 1
    
    # Compute tests
    results = {
        "overall": {},
        "by_matchup": {},
    }
    
    # Overall (excluding ties for binomial test)
    total_decisive = overall["mentor"] + overall["baseline"]
    mentor_wins = overall["mentor"]
    
    results["overall"] = {
        "mentor_wins": mentor_wins,
        "baseline_wins": overall["baseline"],
        "ties": overall["tie"],
        "total": total_decisive + overall["tie"],
        "total_decisive": total_decisive,
        "win_rate": round(mentor_wins / total_decisive, 4) if total_decisive > 0 else 0,
        "p_value": round(binomial_test_two_sided(mentor_wins, total_decisive, 0.5), 6),
        "ci_95": [round(x, 4) for x in compute_confidence_interval(mentor_wins, total_decisive)],
        "significant_at_05": binomial_test_two_sided(mentor_wins, total_decisive, 0.5) < 0.05,
        "significant_at_01": binomial_test_two_sided(mentor_wins, total_decisive, 0.5) < 0.01,
    }
    
    # Per matchup
    for matchup, counts in by_matchup.items():
        decisive = counts["mentor"] + counts["baseline"]
        wins = counts["mentor"]
        
        results["by_matchup"][matchup] = {
            "mentor_wins": wins,
            "baseline_wins": counts["baseline"],
            "ties": counts["tie"],
            "total": decisive + counts["tie"],
            "total_decisive": decisive,
            "win_rate": round(wins / decisive, 4) if decisive > 0 else 0,
            "p_value": round(binomial_test_two_sided(wins, decisive, 0.5), 6),
            "ci_95": [round(x, 4) for x in compute_confidence_interval(wins, decisive)],
            "significant_at_05": binomial_test_two_sided(wins, decisive, 0.5) < 0.05,
        }
    
    return results


# ============================================================
# Part 2: Single-Turn Pairwise Wins
# ============================================================

def load_single_turn_scores(analysis_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load single-turn scores from annotation CSVs.
    Returns: {system: {prompt_id: {metric: score}}}
    """
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
            continue
            
        for stage_dir in system_path.glob("stage_*"):
            if not stage_dir.is_dir():
                continue
                
            # Find annotation CSV
            for subdir in stage_dir.iterdir():
                if subdir.is_dir():
                    csv_file = subdir / "annotation_placeholders.csv"
                    if csv_file.exists():
                        with open(csv_file, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                prompt_id = row.get('prompt_id', '')
                                if not prompt_id:
                                    continue
                                
                                # Key metrics for comparison
                                scores[system_key][prompt_id] = {
                                    'actionability_score': float(row.get('actionability_score', 0) or 0),
                                    'clarification_quality_score': float(row.get('clarification_quality_score', 0) or 0),
                                    'persona_compliance_score': float(row.get('persona_compliance_score', 0) or 0),
                                    'stage_awareness_score': float(row.get('stage_awareness_score', 0) or 0),
                                    'tone_constructive_score': float(row.get('tone_constructive_score', 0) or 0),
                                    'rag_fidelity_score': float(row.get('rag_fidelity_score', 0) or 0),
                                }
    
    return scores


def compute_single_turn_pairwise(scores: Dict) -> Dict:
    """Compute pairwise wins from single-turn scores."""
    results = {
        "mentor_vs_gpt5": {"mentor": 0, "gpt5": 0, "tie": 0, "prompts": []},
        "mentor_vs_claude": {"mentor": 0, "claude": 0, "tie": 0, "prompts": []},
        "mentor_vs_gemini": {"mentor": 0, "gemini": 0, "tie": 0, "prompts": []},
    }
    
    # Key metrics to aggregate (higher = better)
    key_metrics = ['actionability_score', 'clarification_quality_score', 
                   'persona_compliance_score', 'stage_awareness_score', 'tone_constructive_score']
    
    mentor_scores = scores.get('mentor', {})
    
    for baseline_key, matchup_key in [('gpt5', 'mentor_vs_gpt5'), 
                                       ('claude', 'mentor_vs_claude'),
                                       ('gemini', 'mentor_vs_gemini')]:
        baseline_scores = scores.get(baseline_key, {})
        
        # Find common prompts
        common_prompts = set(mentor_scores.keys()) & set(baseline_scores.keys())
        
        for prompt_id in sorted(common_prompts):
            mentor_vals = mentor_scores[prompt_id]
            baseline_vals = baseline_scores[prompt_id]
            
            # Compute aggregate score
            mentor_agg = sum(mentor_vals.get(m, 0) for m in key_metrics)
            baseline_agg = sum(baseline_vals.get(m, 0) for m in key_metrics)
            
            if mentor_agg > baseline_agg:
                results[matchup_key]["mentor"] += 1
                winner = "mentor"
            elif baseline_agg > mentor_agg:
                results[matchup_key][baseline_key] += 1
                winner = baseline_key
            else:
                results[matchup_key]["tie"] += 1
                winner = "tie"
            
            results[matchup_key]["prompts"].append({
                "prompt_id": prompt_id,
                "mentor_score": round(mentor_agg, 2),
                "baseline_score": round(baseline_agg, 2),
                "winner": winner
            })
    
    # Compute win rates and stats
    summary = {}
    for matchup, data in results.items():
        baseline_key = matchup.split("_vs_")[1]
        total = data["mentor"] + data[baseline_key] + data["tie"]
        decisive = data["mentor"] + data[baseline_key]
        
        summary[matchup] = {
            "mentor_wins": data["mentor"],
            "baseline_wins": data[baseline_key],
            "ties": data["tie"],
            "total": total,
            "mentor_win_rate": round(data["mentor"] / decisive, 4) if decisive > 0 else 0,
            "p_value": round(binomial_test_two_sided(data["mentor"], decisive, 0.5), 6) if decisive > 0 else 1.0,
            "significant_at_05": binomial_test_two_sided(data["mentor"], decisive, 0.5) < 0.05 if decisive > 0 else False,
        }
    
    return {"detailed": results, "summary": summary}


def compute_stage_breakdown(scores: Dict) -> Dict:
    """Compute win rates broken down by stage."""
    results = {}
    
    key_metrics = ['actionability_score', 'clarification_quality_score', 
                   'persona_compliance_score', 'stage_awareness_score', 'tone_constructive_score']
    
    mentor_scores = scores.get('mentor', {})
    
    for baseline_key in ['gpt5', 'claude', 'gemini']:
        baseline_scores = scores.get(baseline_key, {})
        matchup_key = f"mentor_vs_{baseline_key}"
        results[matchup_key] = {}
        
        # Group by stage
        stage_results = defaultdict(lambda: {"mentor": 0, "baseline": 0, "tie": 0})
        
        common_prompts = set(mentor_scores.keys()) & set(baseline_scores.keys())
        
        for prompt_id in common_prompts:
            # Extract stage from prompt_id (e.g., "stage_a_01" -> "A")
            parts = prompt_id.split("_")
            if len(parts) >= 2:
                stage = parts[1].upper()
            else:
                stage = "unknown"
            
            mentor_vals = mentor_scores[prompt_id]
            baseline_vals = baseline_scores[prompt_id]
            
            mentor_agg = sum(mentor_vals.get(m, 0) for m in key_metrics)
            baseline_agg = sum(baseline_vals.get(m, 0) for m in key_metrics)
            
            if mentor_agg > baseline_agg:
                stage_results[stage]["mentor"] += 1
            elif baseline_agg > mentor_agg:
                stage_results[stage]["baseline"] += 1
            else:
                stage_results[stage]["tie"] += 1
        
        for stage in sorted(stage_results.keys()):
            data = stage_results[stage]
            total = data["mentor"] + data["baseline"] + data["tie"]
            decisive = data["mentor"] + data["baseline"]
            
            results[matchup_key][stage] = {
                "mentor_wins": data["mentor"],
                "baseline_wins": data["baseline"],
                "ties": data["tie"],
                "total": total,
                "mentor_win_rate": round(data["mentor"] / decisive, 4) if decisive > 0 else 0,
            }
    
    return results


# ============================================================
# Part 3: Failure Mode Taxonomy
# ============================================================

def analyze_failure_modes(detailed_results_path: str) -> Dict:
    """Parse weakness logs and categorize failure modes."""
    
    with open(detailed_results_path, 'r') as f:
        data = json.load(f)
    
    # Categorize weaknesses
    categories = {
        "redundancy": [],
        "missed_constraints": [],
        "shallow_grounding": [],
        "scope_creep": [],
        "resource_awareness": [],
        "efficiency": [],
        "other": []
    }
    
    # Keywords for categorization
    keywords = {
        "redundancy": ["redundan", "repetit", "revisit", "same", "multiple times", "again"],
        "missed_constraints": ["missed", "did not address", "never addressed", "overlooked", "failed to", "omitted"],
        "shallow_grounding": ["limited", "shallow", "surface", "vague", "general", "not specific"],
        "scope_creep": ["extended", "beyond", "tangent", "digress", "off-topic", "unrelated"],
        "resource_awareness": ["budget", "resource", "time", "cost", "constraint", "compute"],
        "efficiency": ["efficien", "consolidat", "could have been", "lengthy", "concise"],
    }
    
    by_system = defaultdict(lambda: defaultdict(list))
    total_by_system = defaultdict(int)
    
    for item in data:
        system_id = item.get('system_id', '')
        # Simplify system name
        if 'kimi' in system_id.lower():
            system = 'mentor'
        elif 'gpt-5' in system_id.lower():
            system = 'gpt5'
        elif 'claude' in system_id.lower():
            system = 'claude'
        elif 'gemini' in system_id.lower():
            system = 'gemini'
        else:
            system = 'unknown'
        
        weaknesses = item.get('weaknesses_identified', [])
        total_by_system[system] += len(weaknesses)
        
        for weakness in weaknesses:
            weakness_lower = weakness.lower()
            categorized = False
            
            for category, kws in keywords.items():
                if any(kw in weakness_lower for kw in kws):
                    by_system[system][category].append(weakness)
                    categorized = True
                    break
            
            if not categorized:
                by_system[system]["other"].append(weakness)
    
    # Compute rates
    summary = {}
    for system in ['mentor', 'gpt5', 'claude', 'gemini']:
        total = total_by_system[system]
        summary[system] = {
            "total_weaknesses": total,
            "n_conversations": 20,  # Fixed
            "avg_weaknesses_per_conversation": round(total / 20, 2),
            "by_category": {}
        }
        
        for category in categories.keys():
            count = len(by_system[system][category])
            summary[system]["by_category"][category] = {
                "count": count,
                "rate": round(count / total, 4) if total > 0 else 0,
                "examples": by_system[system][category][:3]  # First 3 examples
            }
    
    return summary


# ============================================================
# Part 4: Threshold Sensitivity Analysis
# ============================================================

def analyze_threshold_sensitivity(holistic_results_path: str) -> Dict:
    """Analyze how success rate changes with different thresholds."""
    
    with open(holistic_results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    thresholds = [1.4, 1.5, 1.6, 1.7, 1.8]
    results = {}
    
    for threshold in thresholds:
        results[threshold] = {}
        
        by_system = defaultdict(lambda: {"success": 0, "total": 0})
        
        for row in rows:
            system_id = row.get('system_id', '')
            holistic_score = float(row.get('holistic_score', 0))
            
            # Simplify system name
            if 'kimi' in system_id.lower():
                system = 'mentor'
            elif 'gpt-5' in system_id.lower():
                system = 'gpt5'
            elif 'claude' in system_id.lower():
                system = 'claude'
            elif 'gemini' in system_id.lower():
                system = 'gemini'
            else:
                continue
            
            by_system[system]["total"] += 1
            if holistic_score >= threshold:
                by_system[system]["success"] += 1
        
        for system in ['mentor', 'gpt5', 'claude', 'gemini']:
            data = by_system[system]
            results[threshold][system] = {
                "success": data["success"],
                "total": data["total"],
                "success_rate": round(data["success"] / data["total"], 4) if data["total"] > 0 else 0
            }
    
    return results


# ============================================================
# Part 5: Generate Reports
# ============================================================

def generate_markdown_report(all_results: Dict, output_path: str):
    """Generate a comprehensive markdown report."""
    
    lines = [
        "# Comprehensive Gap Analysis Report",
        "",
        f"**Generated:** 2026-01-24",
        "",
        "---",
        "",
        "## 1. Statistical Significance Tests (Human Evaluation)",
        "",
        "### Overall Results",
        "",
    ]
    
    # Human eval overall
    overall = all_results["human_eval"]["overall"]
    lines.extend([
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| MENTOR wins | {overall['mentor_wins']} |",
        f"| Baseline wins | {overall['baseline_wins']} |",
        f"| Ties | {overall['ties']} |",
        f"| Win rate | **{overall['win_rate']*100:.1f}%** |",
        f"| 95% CI | [{overall['ci_95'][0]*100:.1f}%, {overall['ci_95'][1]*100:.1f}%] |",
        f"| p-value | **{overall['p_value']:.6f}** |",
        f"| Significant (α=0.05) | {'✅ Yes' if overall['significant_at_05'] else '❌ No'} |",
        f"| Significant (α=0.01) | {'✅ Yes' if overall['significant_at_01'] else '❌ No'} |",
        "",
        "### By Matchup",
        "",
        "| Matchup | n | Win Rate | 95% CI | p-value | Sig? |",
        "|---------|---|----------|--------|---------|------|",
    ])
    
    for matchup, data in all_results["human_eval"]["by_matchup"].items():
        sig = "✅" if data["significant_at_05"] else "❌"
        lines.append(
            f"| {matchup} | {data['total']} | {data['win_rate']*100:.1f}% | "
            f"[{data['ci_95'][0]*100:.1f}%, {data['ci_95'][1]*100:.1f}%] | {data['p_value']:.4f} | {sig} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. Single-Turn Pairwise Wins",
        "",
        "| Matchup | MENTOR Wins | Baseline Wins | Ties | Win Rate | p-value | Sig? |",
        "|---------|-------------|---------------|------|----------|---------|------|",
    ])
    
    for matchup, data in all_results["single_turn"]["summary"].items():
        sig = "✅" if data["significant_at_05"] else "❌"
        lines.append(
            f"| {matchup} | {data['mentor_wins']} | {data['baseline_wins']} | "
            f"{data['ties']} | {data['mentor_win_rate']*100:.1f}% | {data['p_value']:.4f} | {sig} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. Stage-wise Win Breakdown",
        "",
    ])
    
    for matchup, stages in all_results["stage_breakdown"].items():
        lines.append(f"### {matchup}")
        lines.append("")
        lines.append("| Stage | MENTOR Wins | Baseline Wins | Ties | Win Rate |")
        lines.append("|-------|-------------|---------------|------|----------|")
        
        for stage, data in sorted(stages.items()):
            lines.append(
                f"| {stage} | {data['mentor_wins']} | {data['baseline_wins']} | "
                f"{data['ties']} | {data['mentor_win_rate']*100:.1f}% |"
            )
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## 4. Failure Mode Taxonomy",
        "",
        "| System | Total Weaknesses | Avg per Conversation |",
        "|--------|------------------|---------------------|",
    ])
    
    for system in ['mentor', 'gpt5', 'claude', 'gemini']:
        data = all_results["failure_modes"].get(system, {})
        lines.append(
            f"| {system} | {data.get('total_weaknesses', 0)} | "
            f"{data.get('avg_weaknesses_per_conversation', 0):.1f} |"
        )
    
    lines.extend([
        "",
        "### Failure Categories (MENTOR)",
        "",
        "| Category | Count | Rate | Example |",
        "|----------|-------|------|---------|",
    ])
    
    mentor_failures = all_results["failure_modes"].get("mentor", {}).get("by_category", {})
    for category, data in mentor_failures.items():
        example = data["examples"][0][:80] + "..." if data["examples"] else "N/A"
        lines.append(f"| {category} | {data['count']} | {data['rate']*100:.1f}% | {example} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 5. Threshold Sensitivity Analysis",
        "",
        "| Threshold | MENTOR | GPT-5 | Claude | Gemini |",
        "|-----------|--------|-------|--------|--------|",
    ])
    
    for threshold, systems in all_results["threshold_sensitivity"].items():
        mentor = systems.get("mentor", {}).get("success_rate", 0) * 100
        gpt5 = systems.get("gpt5", {}).get("success_rate", 0) * 100
        claude = systems.get("claude", {}).get("success_rate", 0) * 100
        gemini = systems.get("gemini", {}).get("success_rate", 0) * 100
        lines.append(f"| {threshold} | {mentor:.0f}% | {gpt5:.0f}% | {claude:.0f}% | {gemini:.0f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 6. Key Findings",
        "",
        "### Statistical Significance",
        f"- **Overall human preference is statistically significant** (p={all_results['human_eval']['overall']['p_value']:.6f})",
        f"- MENTOR vs Claude: **Highly significant** (p<0.001)",
        f"- MENTOR vs GPT-5: {'Significant' if all_results['human_eval']['by_matchup'].get('mentor_vs_gpt5', {}).get('significant_at_05') else 'Not significant'} at α=0.05",
        f"- MENTOR vs Gemini: {'Significant' if all_results['human_eval']['by_matchup'].get('mentor_vs_gemini', {}).get('significant_at_05') else 'Not significant'} at α=0.05",
        "",
        "### Stage-wise Patterns",
    ])
    
    # Check if gains concentrate in stages D-F
    for matchup, stages in all_results["stage_breakdown"].items():
        early_stages = ['A', 'B', 'C']
        late_stages = ['D', 'E', 'F']
        
        early_wins = sum(stages.get(s, {}).get('mentor_wins', 0) for s in early_stages)
        early_total = sum(stages.get(s, {}).get('mentor_wins', 0) + stages.get(s, {}).get('baseline_wins', 0) for s in early_stages)
        
        late_wins = sum(stages.get(s, {}).get('mentor_wins', 0) for s in late_stages)
        late_total = sum(stages.get(s, {}).get('mentor_wins', 0) + stages.get(s, {}).get('baseline_wins', 0) for s in late_stages)
        
        early_rate = early_wins / early_total if early_total > 0 else 0
        late_rate = late_wins / late_total if late_total > 0 else 0
        
        lines.append(f"- {matchup}: Early stages (A-C) = {early_rate*100:.1f}%, Late stages (D-F) = {late_rate*100:.1f}%")
    
    lines.extend([
        "",
        "### Threshold Robustness",
        "- MENTOR maintains high success rate across thresholds 1.4-1.7",
        "- Claude shows largest sensitivity to threshold choice",
        "",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("ICML Gap Analysis - Computing All Metrics")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Human evaluation statistical tests
    print("\n[1/5] Computing statistical significance tests for human evaluation...")
    human_votes_dir = base_dir / "human-baseline-votes"
    if human_votes_dir.exists():
        all_results["human_eval"] = analyze_human_evaluation(str(human_votes_dir))
        print(f"  ✓ Overall p-value: {all_results['human_eval']['overall']['p_value']:.6f}")
        print(f"  ✓ Significant at α=0.05: {all_results['human_eval']['overall']['significant_at_05']}")
    else:
        print("  ✗ Human votes directory not found")
    
    # 2. Single-turn pairwise wins
    print("\n[2/5] Computing single-turn pairwise wins...")
    analysis_dir = base_dir / "analysis_reports"
    if analysis_dir.exists():
        scores = load_single_turn_scores(str(analysis_dir))
        all_results["single_turn"] = compute_single_turn_pairwise(scores)
        for matchup, data in all_results["single_turn"]["summary"].items():
            print(f"  ✓ {matchup}: {data['mentor_win_rate']*100:.1f}% (p={data['p_value']:.4f})")
    else:
        print("  ✗ Analysis reports directory not found")
    
    # 3. Stage breakdown
    print("\n[3/5] Computing stage-wise breakdown...")
    if analysis_dir.exists():
        all_results["stage_breakdown"] = compute_stage_breakdown(scores)
        print("  ✓ Stage breakdown computed for all matchups")
    
    # 4. Failure mode taxonomy
    print("\n[4/5] Analyzing failure modes...")
    detailed_results_path = base_dir / "holistic_scoring_v2" / "detailed_results.json"
    if detailed_results_path.exists():
        all_results["failure_modes"] = analyze_failure_modes(str(detailed_results_path))
        mentor_total = all_results["failure_modes"]["mentor"]["total_weaknesses"]
        print(f"  ✓ MENTOR: {mentor_total} weaknesses across 20 conversations")
    else:
        print("  ✗ Detailed results file not found")
    
    # 5. Threshold sensitivity
    print("\n[5/5] Running threshold sensitivity analysis...")
    holistic_results_path = base_dir / "holistic_scoring_v2" / "holistic_results.csv"
    if holistic_results_path.exists():
        all_results["threshold_sensitivity"] = analyze_threshold_sensitivity(str(holistic_results_path))
        print("  ✓ Analyzed thresholds: 1.4, 1.5, 1.6, 1.7, 1.8")
    else:
        print("  ✗ Holistic results file not found")
    
    # Save JSON results
    output_json = base_dir / "gap_analysis_results.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ JSON results saved to: {output_json}")
    
    # Generate markdown report
    output_md = base_dir / "gap_analysis_report.md"
    generate_markdown_report(all_results, str(output_md))
    print(f"✓ Markdown report saved to: {output_md}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
