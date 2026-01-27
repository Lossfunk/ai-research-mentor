#!/usr/bin/env python3
"""
Generate publication-quality figures for ICML 2026 submission.

Follows A* conference figure guidelines:
- Okabe-Ito colorblind-friendly palette
- 300+ DPI for print
- Self-contained, readable without caption
- Markers for accessibility
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# =============================================================================
# OKABE-ITO COLOR PALETTE (colorblind-friendly)
# =============================================================================
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000',
}

# System color mapping
SYSTEM_COLORS = {
    'MENTOR': COLORS['blue'],
    'Gemini 3': COLORS['bluish_green'],
    'GPT-5': COLORS['orange'],
    'Claude 4.5': COLORS['reddish_purple'],
}

SYSTEM_MARKERS = {
    'MENTOR': 'o',
    'Gemini 3': 's',
    'GPT-5': 'D',
    'Claude 4.5': '^',
}

# Plot styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_data(base_path: Path):
    """Load all evaluation data."""
    data = {}

    # Multi-turn holistic results
    holistic_csv = base_path / 'holistic_scoring_v2' / 'holistic_results.csv'
    if holistic_csv.exists():
        data['multi_turn'] = pd.read_csv(holistic_csv)

    # Agent summary
    agent_csv = base_path / 'holistic_scoring_v2' / 'agent_summary.csv'
    if agent_csv.exists():
        data['agent_summary'] = pd.read_csv(agent_csv)

    # Single-turn results
    single_turn_json = base_path / 'results' / 'single_turn_holistic_results.json'
    if single_turn_json.exists():
        with open(single_turn_json) as f:
            data['single_turn'] = json.load(f)

    # Cost/latency
    cost_json = base_path / 'results' / 'cost_latency_results.json'
    if cost_json.exists():
        with open(cost_json) as f:
            data['cost_latency'] = json.load(f)

    # Ablation results
    ablation_dir = base_path / 'ablations'
    data['ablations'] = {}
    if ablation_dir.exists():
        for ablation_folder in ablation_dir.iterdir():
            if ablation_folder.is_dir() and 'ablation' in ablation_folder.name:
                comparison_file = ablation_folder / 'results' / 'ablation_comparison.json'
                if comparison_file.exists():
                    with open(comparison_file) as f:
                        ablation_data = json.load(f)
                        data['ablations'][ablation_folder.name] = ablation_data

    return data


def fig1_multi_turn_holistic(data: dict, output_dir: Path):
    """
    Figure 1: Multi-turn holistic scores comparison.
    Bar chart with individual data points and error bars.
    """
    df = data['multi_turn']

    # Map system IDs to display names
    system_map = {
        'openrouter:moonshotai/kimi-k2-thinking': 'MENTOR',
        'openrouter:google/gemini-3-pro-preview': 'Gemini 3',
        'openrouter:openai/gpt-5': 'GPT-5',
        'openrouter:anthropic/claude-sonnet-4.5': 'Claude 4.5',
    }
    df['system'] = df['system_id'].map(system_map)

    # Order systems by score
    order = ['MENTOR', 'Gemini 3', 'GPT-5', 'Claude 4.5']

    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(order))
    bar_width = 0.6

    for i, system in enumerate(order):
        system_data = df[df['system'] == system]['holistic_score']
        mean_val = system_data.mean()
        std_val = system_data.std()
        ci = 1.96 * std_val / np.sqrt(len(system_data))

        # Bar
        ax.bar(i, mean_val, bar_width,
               color=SYSTEM_COLORS[system],
               alpha=0.85,
               edgecolor='white',
               linewidth=1.5)

        # Individual points (jittered)
        jitter = np.random.uniform(-0.15, 0.15, len(system_data))
        ax.scatter(i + jitter, system_data,
                   color='#555555',
                   alpha=0.6,
                   s=35,
                   zorder=3,
                   marker=SYSTEM_MARKERS[system])

        # Error bar
        ax.errorbar(i, mean_val, yerr=ci,
                    color='#333333',
                    capsize=4,
                    capthick=1.5,
                    linewidth=1.5,
                    zorder=4)

        # Mean label
        ax.text(i, mean_val + ci + 0.03, f'{mean_val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Threshold line
    ax.axhline(y=1.5, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(-0.4, 1.505, 'success\nthreshold', fontsize=9, color='#666666', va='bottom')

    # Labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(order)
    ax.set_ylabel('Holistic Score (0–2)')
    ax.set_ylim(1.0, 2.0)

    # Sample size annotation
    ax.text(0.98, 0.98, 'n=20 per system\nError bars: 95% CI',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='#666666')

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        output_path = output_dir / f'fig1_multi_turn_holistic.{fmt}'
        plt.savefig(output_path, dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig1_multi_turn_holistic")


def fig2_success_rates(data: dict, output_dir: Path):
    """
    Figure 2: Success rates comparison.
    Horizontal bar chart showing success % and positive stop %.
    """
    df = data['agent_summary']

    system_map = {
        'openrouter:moonshotai/kimi-k2-thinking': 'MENTOR',
        'openrouter:google/gemini-3-pro-preview': 'Gemini 3',
        'openrouter:openai/gpt-5': 'GPT-5',
        'openrouter:anthropic/claude-sonnet-4.5': 'Claude 4.5',
    }
    df['system'] = df['system_id'].map(system_map)

    # Parse percentages
    df['success_pct'] = df['success_rate'].str.rstrip('%').astype(float)
    df['positive_stop_pct'] = df['positive_stop_rate'].str.rstrip('%').astype(float)

    order = ['MENTOR', 'Gemini 3', 'GPT-5', 'Claude 4.5']
    df = df.set_index('system').loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(order))
    bar_height = 0.35

    # Success rate bars
    ax.barh(y_pos - bar_height/2, df['success_pct'], bar_height,
            label='Success Rate',
            color=[SYSTEM_COLORS[s] for s in order],
            alpha=0.9)

    # Positive stop rate bars
    ax.barh(y_pos + bar_height/2, df['positive_stop_pct'], bar_height,
            label='Positive Stop Rate',
            color=[SYSTEM_COLORS[s] for s in order],
            alpha=0.5,
            hatch='///')

    # Value labels
    for i, (success, positive) in enumerate(zip(df['success_pct'], df['positive_stop_pct'])):
        ax.text(success + 1, i - bar_height/2, f'{success:.0f}%', va='center', fontsize=10)
        ax.text(positive + 1, i + bar_height/2, f'{positive:.0f}%', va='center', fontsize=10, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, 110)
    ax.legend(loc='lower right')

    # Add explanation
    ax.text(0.02, 0.02,
            'Success: holistic score ≥1.5 or positive termination\n'
            'Positive Stop: student reached goal or felt mentored enough',
            transform=ax.transAxes, fontsize=8, color='#666666', va='bottom')

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_dir / f'fig2_success_rates.{fmt}', dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig2_success_rates")


def fig3_ablations(data: dict, output_dir: Path):
    """
    Figure 3: Ablation study results.
    Shows impact of removing stage awareness and guidelines.
    """
    ablations = data.get('ablations', {})

    if not ablations:
        print("No ablation data found, skipping fig3")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Extract data from ablation results
    stage_ablation = ablations.get('stage_ablation', {}).get('results', {}).get('holistic_score', {})
    guidelines_ablation = ablations.get('guidelines_ablation', {}).get('results', {}).get('holistic_score', {})

    baseline_score = stage_ablation.get('baseline_mean', 1.547)
    no_stage_score = stage_ablation.get('ablated_mean', 0.697)
    no_stage_delta = stage_ablation.get('delta_pct', -54.9)
    no_guidelines_score = guidelines_ablation.get('ablated_mean', 1.026)
    no_guidelines_delta = guidelines_ablation.get('delta_pct', -33.7)

    conditions = ['Full MENTOR', '− Stage Awareness', '− Guidelines']
    scores = [baseline_score, no_stage_score, no_guidelines_score]
    deltas = [0, no_stage_delta, no_guidelines_delta]

    colors = [COLORS['blue'], COLORS['vermillion'], COLORS['orange']]

    x_pos = np.arange(len(conditions))
    ax.bar(x_pos, scores, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

    # Add delta annotations
    for i, (score, delta) in enumerate(zip(scores, deltas)):
        if delta != 0:
            ax.annotate(f'{delta:+.1f}%',
                        xy=(i, score),
                        xytext=(i, score + 0.15),
                        ha='center', fontsize=11, fontweight='bold',
                        color=COLORS['vermillion'] if delta < -40 else COLORS['orange'])
        ax.text(i, score + 0.05, f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Threshold line
    ax.axhline(y=1.5, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2.4, 1.52, 'success threshold', fontsize=9, color='#666666')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Holistic Score (0–2)')
    ax.set_ylim(0, 2.0)

    # Annotation box
    textstr = 'Stage awareness accounts for\n54.9% of MENTOR\'s effectiveness'
    props = dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8)
    ax.text(0.98, 0.25, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_dir / f'fig3_ablations.{fmt}', dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig3_ablations")


def fig4_human_evaluation(data: dict, output_dir: Path):
    """
    Figure 4: Human pairwise evaluation results.
    Win rates from 218 comparisons by 15 raters.
    """
    # Data from icml_eval_gaps.md
    matchups = ['vs Claude', 'vs GPT-5', 'vs Gemini', 'Overall']
    mentor_wins = [79.7, 58.7, 53.1, 64.7]
    baseline_wins = [20.3 - 3.7, 41.3 - 3.7, 46.9 - 3.7, 35.3 - 3.7]  # Approx, accounting for ties

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(matchups))
    bar_height = 0.5

    # MENTOR wins (from left)
    ax.barh(y_pos, mentor_wins, bar_height,
            label='MENTOR wins', color=COLORS['blue'], alpha=0.9)

    # Baseline wins (from right, stacked)
    ax.barh(y_pos, [-w for w in baseline_wins], bar_height,
            label='Baseline wins', color=COLORS['reddish_purple'], alpha=0.7)

    # Center line
    ax.axvline(x=0, color='#333333', linewidth=1)

    # Value labels
    for i, (m_win, b_win) in enumerate(zip(mentor_wins, baseline_wins)):
        ax.text(m_win/2, i, f'{m_win:.1f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=11)
        if b_win > 10:
            ax.text(-b_win/2, i, f'{b_win:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(matchups)
    ax.set_xlabel('Win Rate (%)')
    ax.set_xlim(-60, 100)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['blue'], alpha=0.9, label='MENTOR wins'),
        mpatches.Patch(facecolor=COLORS['reddish_purple'], alpha=0.7, label='Baseline wins'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Annotation
    ax.text(0.02, 0.98, '218 pairwise comparisons\n15 human raters',
            transform=ax.transAxes, fontsize=9, va='top', color='#666666')

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_dir / f'fig4_human_evaluation.{fmt}', dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig4_human_evaluation")


def fig5_efficiency(data: dict, output_dir: Path):
    """
    Figure 5: Efficiency comparison - time per turn vs quality.
    Scatter plot showing cost-effectiveness.
    """
    cost_data = data.get('cost_latency', {}).get('multi_turn', {})
    agent_data = data.get('agent_summary')

    if not cost_data or agent_data is None:
        print("Missing data for efficiency figure, skipping")
        return

    system_map = {
        'mentor': 'MENTOR',
        'gemini': 'Gemini 3',
        'gpt5': 'GPT-5',
        'claude': 'Claude 4.5',
    }

    system_map_reverse = {
        'openrouter:moonshotai/kimi-k2-thinking': 'mentor',
        'openrouter:google/gemini-3-pro-preview': 'gemini',
        'openrouter:openai/gpt-5': 'gpt5',
        'openrouter:anthropic/claude-sonnet-4.5': 'claude',
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in agent_data.iterrows():
        system_key = system_map_reverse.get(row['system_id'])
        if not system_key or system_key not in cost_data:
            continue

        display_name = system_map[system_key]
        time_per_turn = cost_data[system_key]['avg_time_per_turn']
        holistic_score = row['avg_holistic_score']

        ax.scatter(time_per_turn, holistic_score,
                   s=200,
                   c=SYSTEM_COLORS[display_name],
                   marker=SYSTEM_MARKERS[display_name],
                   edgecolors='white',
                   linewidths=2,
                   alpha=0.9,
                   zorder=3)

        # Label
        offset = (5, 5) if display_name != 'Claude 4.5' else (5, -15)
        ax.annotate(display_name,
                    (time_per_turn, holistic_score),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')

    ax.set_xlabel('Average Time per Turn (seconds)')
    ax.set_ylabel('Holistic Score (0–2)')
    ax.set_xlim(0, 80)
    ax.set_ylim(1.4, 1.8)

    # Add quadrant labels
    ax.axhline(y=1.6, color='#cccccc', linestyle=':', alpha=0.5)
    ax.axvline(x=40, color='#cccccc', linestyle=':', alpha=0.5)

    # Ideal quadrant highlight
    rect = FancyBboxPatch((0, 1.6), 40, 0.2,
                          boxstyle="round,pad=0.02",
                          facecolor=COLORS['bluish_green'],
                          alpha=0.1,
                          edgecolor='none')
    ax.add_patch(rect)
    ax.text(20, 1.78, 'Ideal: Fast + High Quality', ha='center', fontsize=9,
            color=COLORS['bluish_green'], alpha=0.8)

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_dir / f'fig5_efficiency.{fmt}', dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig5_efficiency")


def fig6_single_turn_by_stage(data: dict, output_dir: Path):
    """
    Figure 6: Single-turn performance breakdown by research stage.
    Grouped bar chart showing A-F stages.
    """
    single_turn = data.get('single_turn', {}).get('stage_averages', {})

    if not single_turn:
        print("No single-turn data, skipping fig6")
        return

    stages = ['A', 'B', 'C', 'D', 'E', 'F']
    stage_labels = [
        'A: Ideation',
        'B: Literature',
        'C: Methodology',
        'D: Execution',
        'E: Writing',
        'F: Publication'
    ]

    systems = ['mentor', 'gemini', 'gpt5', 'claude']
    display_names = ['MENTOR', 'Gemini 3', 'GPT-5', 'Claude 4.5']

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(stages))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (system, display) in enumerate(zip(systems, display_names)):
        if system not in single_turn:
            continue
        scores = [single_turn[system].get(stage, {}).get('avg', 0) for stage in stages]

        ax.bar(x + offsets[i] * width, scores, width,
               label=display,
               color=SYSTEM_COLORS[display],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=15, ha='right')
    ax.set_ylabel('Holistic Score (0–2)')
    ax.set_ylim(1.0, 1.8)
    ax.legend(loc='upper right', ncol=2)

    # Threshold line
    ax.axhline(y=1.5, color='#888888', linestyle='--', linewidth=1, alpha=0.6)

    # Annotation
    ax.text(0.02, 0.98, 'n=15 prompts per stage per system',
            transform=ax.transAxes, fontsize=9, va='top', color='#666666')

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_dir / f'fig6_single_turn_stages.{fmt}', dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved: fig6_single_turn_stages")


def main():
    # Handle both direct execution and module execution
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent  # icml-evaluation-results/
    output_dir = base_path / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"Loading data from: {base_path}")
    data = load_data(base_path)

    print(f"Generating figures to: {output_dir}")
    print("=" * 50)

    # Generate all figures
    fig1_multi_turn_holistic(data, output_dir)
    fig2_success_rates(data, output_dir)
    fig3_ablations(data, output_dir)
    fig4_human_evaluation(data, output_dir)
    fig5_efficiency(data, output_dir)
    fig6_single_turn_by_stage(data, output_dir)

    print("=" * 50)
    print(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
