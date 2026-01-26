#!/usr/bin/env python3
"""
Compute cost and latency metrics for ICML evaluation.

Outputs:
- Multi-turn: wall-clock time, turns, time per turn
- Single-turn: token usage from judges
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_multi_turn(holistic_csv_path: str) -> dict:
    """Analyze multi-turn timing metrics."""
    results = defaultdict(lambda: {
        "total_time": 0,
        "total_turns": 0,
        "times": [],
        "turns": [],
        "time_per_turn": []
    })
    
    with open(holistic_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            system_id = row.get('system_id', '')
            elapsed = float(row.get('elapsed_seconds', 0))
            turns = int(row.get('total_turns', 0))
            
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
            
            results[system]["total_time"] += elapsed
            results[system]["total_turns"] += turns
            results[system]["times"].append(elapsed)
            results[system]["turns"].append(turns)
            if turns > 0:
                results[system]["time_per_turn"].append(elapsed / turns)
    
    # Compute statistics
    summary = {}
    for system, data in results.items():
        n = len(data["times"])
        summary[system] = {
            "n_conversations": n,
            "total_time_seconds": round(data["total_time"], 1),
            "total_time_minutes": round(data["total_time"] / 60, 1),
            "total_turns": data["total_turns"],
            "avg_time_seconds": round(statistics.mean(data["times"]), 1) if data["times"] else 0,
            "std_time_seconds": round(statistics.stdev(data["times"]), 1) if len(data["times"]) > 1 else 0,
            "avg_turns": round(statistics.mean(data["turns"]), 1) if data["turns"] else 0,
            "avg_time_per_turn": round(statistics.mean(data["time_per_turn"]), 2) if data["time_per_turn"] else 0,
            "min_time": round(min(data["times"]), 1) if data["times"] else 0,
            "max_time": round(max(data["times"]), 1) if data["times"] else 0,
        }
    
    return summary


def analyze_single_turn_tokens(analysis_dir: str) -> dict:
    """Analyze token usage from single-turn judges files."""
    analysis_path = Path(analysis_dir)
    
    results = defaultdict(lambda: {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "n_prompts": 0,
        "input_tokens_list": [],
        "output_tokens_list": [],
    })
    
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
            
        # Find all judges.json files
        for judges_file in system_path.glob("**/*_judges.json"):
            try:
                with open(judges_file, 'r') as f:
                    data = json.load(f)
                
                # Get usage from each metric's judges
                metrics = data.get('metrics', {})
                
                prompt_input = 0
                prompt_output = 0
                
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'judges' in metric_data:
                        for judge in metric_data['judges']:
                            usage = judge.get('usage', {})
                            prompt_input += usage.get('input_tokens', 0)
                            prompt_output += usage.get('output_tokens', 0)
                
                if prompt_input > 0:
                    results[system_key]["total_input_tokens"] += prompt_input
                    results[system_key]["total_output_tokens"] += prompt_output
                    results[system_key]["total_tokens"] += prompt_input + prompt_output
                    results[system_key]["n_prompts"] += 1
                    results[system_key]["input_tokens_list"].append(prompt_input)
                    results[system_key]["output_tokens_list"].append(prompt_output)
                    
            except Exception as e:
                pass
    
    # Compute statistics
    summary = {}
    for system, data in results.items():
        n = data["n_prompts"]
        summary[system] = {
            "n_prompts": n,
            "total_input_tokens": data["total_input_tokens"],
            "total_output_tokens": data["total_output_tokens"],
            "total_tokens": data["total_tokens"],
            "avg_input_tokens_per_prompt": round(data["total_input_tokens"] / n, 0) if n > 0 else 0,
            "avg_output_tokens_per_prompt": round(data["total_output_tokens"] / n, 0) if n > 0 else 0,
            "avg_total_tokens_per_prompt": round(data["total_tokens"] / n, 0) if n > 0 else 0,
        }
    
    return summary


def generate_report(multi_turn: dict, single_turn: dict, output_path: str):
    """Generate markdown report."""
    
    lines = [
        "# Cost & Latency Analysis",
        "",
        f"**Generated:** 2026-01-24",
        "",
        "---",
        "",
        "## 1. Multi-Turn Conversations (Wall-Clock Time)",
        "",
        "| System | Conversations | Total Time | Avg Time | Avg Turns | Time/Turn |",
        "|--------|---------------|------------|----------|-----------|-----------|",
    ]
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = multi_turn.get(system, {})
        lines.append(
            f"| {system.upper()} | {data.get('n_conversations', 0)} | "
            f"{data.get('total_time_minutes', 0):.1f} min | "
            f"{data.get('avg_time_seconds', 0):.0f}s | "
            f"{data.get('avg_turns', 0):.0f} | "
            f"{data.get('avg_time_per_turn', 0):.1f}s |"
        )
    
    lines.extend([
        "",
        "### Detailed Statistics",
        "",
        "| System | Min Time | Max Time | Std Dev |",
        "|--------|----------|----------|---------|",
    ])
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = multi_turn.get(system, {})
        lines.append(
            f"| {system.upper()} | {data.get('min_time', 0):.0f}s | "
            f"{data.get('max_time', 0):.0f}s | "
            f"±{data.get('std_time_seconds', 0):.0f}s |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. Single-Turn Token Usage (Judge Evaluation)",
        "",
        "| System | Prompts | Total Tokens | Avg Input | Avg Output | Avg Total |",
        "|--------|---------|--------------|-----------|------------|-----------|",
    ])
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = single_turn.get(system, {})
        lines.append(
            f"| {system.upper()} | {data.get('n_prompts', 0)} | "
            f"{data.get('total_tokens', 0):,} | "
            f"{data.get('avg_input_tokens_per_prompt', 0):,.0f} | "
            f"{data.get('avg_output_tokens_per_prompt', 0):,.0f} | "
            f"{data.get('avg_total_tokens_per_prompt', 0):,.0f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. Key Findings",
        "",
    ])
    
    # Compute relative efficiency
    mentor_time = multi_turn.get('mentor', {}).get('avg_time_seconds', 1)
    
    lines.append("### Multi-Turn Efficiency")
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = multi_turn.get(system, {})
        avg_time = data.get('avg_time_seconds', 0)
        ratio = avg_time / mentor_time if mentor_time > 0 else 0
        lines.append(f"- **{system.upper()}**: {avg_time:.0f}s avg ({ratio:.1f}x vs MENTOR)")
    
    lines.extend([
        "",
        "### Observations",
        f"- MENTOR completes conversations in **{multi_turn.get('mentor', {}).get('avg_time_seconds', 0):.0f}s** average",
        f"- Claude is **{multi_turn.get('claude', {}).get('avg_time_seconds', 0) / mentor_time:.1f}x slower** than MENTOR",
        f"- MENTOR averages **{multi_turn.get('mentor', {}).get('avg_turns', 0):.0f} turns** per conversation",
        f"- Time per turn: MENTOR ({multi_turn.get('mentor', {}).get('avg_time_per_turn', 0):.1f}s) vs Claude ({multi_turn.get('claude', {}).get('avg_time_per_turn', 0):.1f}s)",
        "",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("COST & LATENCY ANALYSIS")
    print("=" * 70)
    
    # Multi-turn analysis
    print("\n[1/2] Analyzing multi-turn timing...")
    holistic_csv = base_dir / "holistic_scoring_v2" / "holistic_results.csv"
    multi_turn = analyze_multi_turn(str(holistic_csv))
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = multi_turn.get(system, {})
        print(f"  {system.upper()}: {data.get('avg_time_seconds', 0):.0f}s avg, {data.get('avg_turns', 0):.0f} turns avg")
    
    # Single-turn token analysis
    print("\n[2/2] Analyzing single-turn token usage...")
    analysis_dir = base_dir / "analysis_reports"
    single_turn = analyze_single_turn_tokens(str(analysis_dir))
    
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = single_turn.get(system, {})
        print(f"  {system.upper()}: {data.get('avg_total_tokens_per_prompt', 0):,.0f} tokens/prompt avg")
    
    # Save results
    results = {
        "multi_turn": multi_turn,
        "single_turn": single_turn
    }
    
    output_json = base_dir / "cost_latency_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ JSON saved to: {output_json}")
    
    # Generate report
    output_md = base_dir / "cost_latency_report.md"
    generate_report(multi_turn, single_turn, str(output_md))
    print(f"✓ Report saved to: {output_md}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("MULTI-TURN SUMMARY")
    print("=" * 70)
    print(f"\n{'System':<10} {'Avg Time':<12} {'Avg Turns':<12} {'Time/Turn':<12} {'Total':<12}")
    print("-" * 58)
    for system in ['mentor', 'gemini', 'gpt5', 'claude']:
        data = multi_turn.get(system, {})
        print(f"{system.upper():<10} {data.get('avg_time_seconds', 0):>8.0f}s   {data.get('avg_turns', 0):>8.0f}     {data.get('avg_time_per_turn', 0):>8.1f}s    {data.get('total_time_minutes', 0):>8.1f}m")


if __name__ == "__main__":
    main()
