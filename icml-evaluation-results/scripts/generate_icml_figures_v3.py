#!/usr/bin/env python3
"""
Generate publication-quality figures for ICML 2026 submission (v3).

ICML Format Specifications:
- Dual-column layout
- Single column width: ~3.25 inches (8.25 cm)
- Full width (figure*): ~6.875 inches (17.5 cm)

Figure Layout:
- Fig 1 (figure*): Multi-turn quality + Human evaluation (2 panels)
- Fig 2 (figure): Ablation study (single column)
- Fig 3 (figure*): Stage-wise analysis (2 panels)

Removed: Efficiency figure (confounded by API provider routing)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ICML DIMENSIONS
# =============================================================================
SINGLE_COL_WIDTH = 3.25  # inches
FULL_WIDTH = 6.875       # inches
DPI = 300

# =============================================================================
# OKABE-ITO COLOR PALETTE
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
    'gray': '#888888',
    'light_gray': '#CCCCCC',
}

SYSTEM_COLORS = {
    'MENTOR': COLORS['blue'],
    'Gemini 3 Pro': COLORS['bluish_green'],
    'GPT-5': COLORS['orange'],
    'Claude Sonnet 4.5': COLORS['reddish_purple'],
}

SYSTEM_SHORT = {
    'MENTOR': 'MENTOR',
    'Gemini 3 Pro': 'Gemini 3',
    'GPT-5': 'GPT-5',
    'Claude Sonnet 4.5': 'Claude 4.5',
}

SYSTEM_MARKERS = {
    'MENTOR': 'o',
    'Gemini 3 Pro': 's',
    'GPT-5': 'D',
    'Claude Sonnet 4.5': '^',
}

SYSTEM_ORDER = ['MENTOR', 'Gemini 3 Pro', 'GPT-5', 'Claude Sonnet 4.5']

# Plot styling (ICML compliant - larger fonts for readability)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.0,
})


def load_data(base_path: Path) -> Dict[str, Any]:
    """Load all evaluation data."""
    data = {}
    
    holistic_csv = base_path / 'holistic_scoring_v2' / 'holistic_results.csv'
    if holistic_csv.exists():
        data['multi_turn'] = pd.read_csv(holistic_csv)
    
    single_turn_json = base_path / 'results' / 'single_turn_holistic_results.json'
    if single_turn_json.exists():
        with open(single_turn_json) as f:
            data['single_turn'] = json.load(f)
    
    ablation_90_json = base_path / 'ablations' / 'ablations_90' / 'ablation_comparison_90.json'
    if ablation_90_json.exists():
        with open(ablation_90_json) as f:
            data['ablations_90'] = json.load(f)
    
    iaa_json = base_path / 'inter_annotator_agreement' / 'iaa_report.json'
    if iaa_json.exists():
        with open(iaa_json) as f:
            data['iaa'] = json.load(f)
    
    return data


def map_system_id(system_id: str) -> str:
    mapping = {
        'openrouter:moonshotai/kimi-k2-thinking': 'MENTOR',
        'openrouter:google/gemini-3-pro-preview': 'Gemini 3 Pro',
        'openrouter:openai/gpt-5': 'GPT-5',
        'openrouter:anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
    }
    return mapping.get(system_id, system_id)


def save_figure(fig, output_dir: Path, name: str):
    for fmt in ['pdf', 'png', 'svg']:
        dpi = DPI if fmt == 'png' else None
        fig.savefig(output_dir / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight', 
                    pad_inches=0.02, facecolor='white', edgecolor='none')
    print(f"  Saved: {name}")


# =============================================================================
# FIGURE 1: Multi-turn + Human Evaluation (FULL WIDTH - figure*)
# =============================================================================
def fig1_main_results(data: Dict[str, Any], output_dir: Path):
    """
    Figure 1: Main results (full-width, 2 panels).
    (a) Multi-turn holistic scores
    (b) Human pairwise evaluation
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))
    
    # --- Panel (a): Multi-turn holistic scores ---
    ax_a = axes[0]
    df = data['multi_turn'].copy()
    df['system'] = df['system_id'].map(map_system_id)
    
    bar_width = 0.6
    for i, system in enumerate(SYSTEM_ORDER):
        system_data = df[df['system'] == system]['holistic_score']
        if len(system_data) == 0:
            continue
        mean_val = system_data.mean()
        std_val = system_data.std()
        ci = 1.96 * std_val / np.sqrt(len(system_data))
        
        color = SYSTEM_COLORS[system]
        short = SYSTEM_SHORT[system]
        
        # Bar
        ax_a.bar(i, mean_val, bar_width, color=color, alpha=0.9, 
                 edgecolor='white', linewidth=0.8)
        
        # Error bar
        ax_a.errorbar(i, mean_val, yerr=ci, color='#333333', capsize=2, 
                      capthick=0.8, linewidth=0.8, zorder=4)
        
        # Mean label INSIDE bar with white text
        ax_a.text(i, mean_val - 0.06, f'{mean_val:.2f}', 
                  ha='center', va='top', fontweight='bold', fontsize=9, color='white')
    
    # Threshold line
    ax_a.axhline(y=1.5, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax_a.set_xticks(range(len(SYSTEM_ORDER)))
    ax_a.set_xticklabels([SYSTEM_SHORT[s] for s in SYSTEM_ORDER], fontsize=7)
    ax_a.set_ylabel('Holistic Score (0-2)', fontsize=8)
    ax_a.set_ylim(1.3, 1.85)
    ax_a.set_title('(a) Multi-turn Mentoring Quality', fontsize=9, fontweight='bold', loc='left')
    
    # Annotations in top-right corner, clearly visible
    ax_a.text(0.97, 0.97, 'n=20 per system\nthreshold=1.5', transform=ax_a.transAxes, 
              ha='right', va='top', fontsize=6, color=COLORS['gray'], linespacing=1.3)
    
    # --- Panel (b): Human pairwise evaluation ---
    ax_b = axes[1]
    
    # Human evaluation data - simple horizontal bars showing MENTOR win rate
    # Clean design: just show win rates with 50% reference line, no significance markers
    matchups = [
        ('vs Claude', 79.7),
        ('vs GPT-5', 58.7),
        ('vs Gemini', 53.1),
        ('Overall', 64.7),
    ]
    
    y_pos = np.arange(len(matchups))
    bar_height = 0.6
    
    # Colors: use baseline color for comparison, MENTOR blue for overall
    bar_colors = [COLORS['reddish_purple'], COLORS['orange'], 
                  COLORS['bluish_green'], COLORS['blue']]
    
    for i, (label, win_rate) in enumerate(matchups):
        ax_b.barh(i, win_rate, bar_height, color=bar_colors[i], alpha=0.9)
        
        # Win rate label INSIDE bar with white text
        ax_b.text(win_rate - 2, i, f'{win_rate:.0f}%', 
                  ha='right', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([m[0] for m in matchups], fontsize=7)
    ax_b.set_xlabel('MENTOR Win Rate (%)', fontsize=8)
    ax_b.set_xlim(0, 100)
    ax_b.set_title('(b) Human Pairwise Evaluation', fontsize=9, fontweight='bold', loc='left')
    
    # Sample size annotation only
    ax_b.text(0.97, 0.97, 'n=218, 15 raters', 
              transform=ax_b.transAxes, ha='right', va='top', fontsize=6, 
              color=COLORS['gray'])
    
    plt.tight_layout(w_pad=1.5)
    save_figure(fig, output_dir, 'fig1_main_results')
    plt.close()


# =============================================================================
# FIGURE 2: Ablation Study (SINGLE COLUMN - figure)
# =============================================================================
def fig2_ablations(data: Dict[str, Any], output_dir: Path):
    """
    Figure 2: Ablation study (single column width).
    Shows impact of removing stage awareness and guidelines.
    """
    ablation_data = data.get('ablations_90', {})
    
    if not ablation_data:
        print("  Warning: No n=90 ablation data found")
        return
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.2))
    
    baseline_score = ablation_data['baseline']['holistic_score']
    no_stage = ablation_data['ablations']['no_stage']
    no_guidelines = ablation_data['ablations']['no_guidelines']
    
    conditions = ['Full\nMENTOR', '− Stage\nAwareness', '− Guidelines']
    scores = [baseline_score, no_stage['ablated_mean'], no_guidelines['ablated_mean']]
    deltas = [0, no_stage['delta_pct'], no_guidelines['delta_pct']]
    colors = [COLORS['blue'], COLORS['vermillion'], COLORS['orange']]
    
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, scores, 0.55, color=colors, alpha=0.85, 
                  edgecolor='white', linewidth=0.8)
    
    # Annotations
    for i, (score, delta) in enumerate(zip(scores, deltas)):
        # Score value above bar
        ax.text(i, score + 0.04, f'{score:.2f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
        # Delta inside bar (if negative)
        if delta != 0:
            delta_color = 'white'
            ax.text(i, score - 0.12, f'{delta:.1f}%', ha='center', va='top',
                    fontsize=7, fontweight='bold', color=delta_color)
    
    # Threshold line
    ax.axhline(y=1.5, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(2.4, 1.52, 'threshold', fontsize=6, color=COLORS['gray'], va='bottom')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=7)
    ax.set_ylabel('Holistic Score (0-2)', fontsize=8)
    ax.set_ylim(0, 1.85)
    
    # Sample size
    ax.text(0.97, 0.97, 'n=90', transform=ax.transAxes, ha='right', va='top',
            fontsize=6, color=COLORS['gray'])
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig2_ablations')
    plt.close()


# =============================================================================
# FIGURE 3: Stage-wise Analysis (FULL WIDTH - figure*)
# =============================================================================
def fig3_stage_analysis(data: Dict[str, Any], output_dir: Path):
    """
    Figure 3: Stage-wise analysis (full-width, 2 panels + shared legend).
    (a) Single-turn scores by research stage
    (b) Multi-turn dimension breakdown
    """
    # Create figure with extra space at bottom for shared legend
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6))
    
    # --- Panel (a): Single-turn by stage ---
    ax_a = axes[0]
    
    single_turn = data.get('single_turn', {}).get('stage_averages', {})
    
    stages = ['A', 'B', 'C', 'D', 'E', 'F']
    
    system_keys = {'MENTOR': 'mentor', 'Gemini 3 Pro': 'gemini', 
                   'GPT-5': 'gpt5', 'Claude Sonnet 4.5': 'claude'}
    
    x = np.arange(len(stages))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    
    bars_for_legend = []
    labels_for_legend = []
    
    for i, system in enumerate(SYSTEM_ORDER):
        key = system_keys.get(system)
        if key not in single_turn:
            continue
        
        scores = []
        errors = []
        for stage in stages:
            stage_data = single_turn[key].get(stage, {})
            scores.append(stage_data.get('avg', 0))
            score_list = stage_data.get('scores', [])
            if score_list:
                std_err = np.std(score_list) / np.sqrt(len(score_list)) * 1.96
            else:
                std_err = 0
            errors.append(std_err)
        
        color = SYSTEM_COLORS[system]
        short = SYSTEM_SHORT[system]
        
        bar = ax_a.bar(x + offsets[i] * width, scores, width,
                       color=color, alpha=0.85, yerr=errors, capsize=1.5, 
                       error_kw={'linewidth': 0.5, 'capthick': 0.5})
        bars_for_legend.append(bar[0])
        labels_for_legend.append(short)
    
    ax_a.axhline(y=1.5, color=COLORS['gray'], linestyle='--', linewidth=0.6, alpha=0.6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stages, fontsize=8)  # Just A, B, C, D, E, F
    ax_a.set_xlabel('Research Stage', fontsize=8)
    ax_a.set_ylabel('Holistic Score (0-2)', fontsize=8)
    ax_a.set_ylim(1.15, 1.72)
    ax_a.set_title('(a) Single-turn by Research Stage', fontsize=9, fontweight='bold', loc='left')
    ax_a.text(0.97, 0.97, 'n=15/stage', transform=ax_a.transAxes, 
              fontsize=6, color=COLORS['gray'], va='top', ha='right')
    
    # --- Panel (b): Multi-turn dimension breakdown ---
    ax_b = axes[1]
    
    df = data['multi_turn'].copy()
    df['system'] = df['system_id'].map(map_system_id)
    
    dimensions = ['overall_helpfulness', 'student_progress', 
                  'mentor_effectiveness', 'conversation_efficiency']
    dim_labels = ['Helpfulness', 'Progress', 'Effectiveness', 'Efficiency']
    
    x = np.arange(len(dimensions))
    width = 0.18
    
    for i, system in enumerate(SYSTEM_ORDER):
        system_df = df[df['system'] == system]
        if len(system_df) == 0:
            continue
        
        means = [system_df[dim].mean() for dim in dimensions]
        stds = [system_df[dim].std() / np.sqrt(len(system_df)) * 1.96 for dim in dimensions]
        
        color = SYSTEM_COLORS[system]
        
        ax_b.bar(x + offsets[i] * width, means, width,
                 color=color, alpha=0.85, yerr=stds, capsize=1.5,
                 error_kw={'linewidth': 0.5, 'capthick': 0.5})
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(dim_labels, fontsize=7)
    ax_b.set_ylabel('Score (0-2)', fontsize=8)
    ax_b.set_ylim(1.0, 1.9)
    ax_b.set_title('(b) Multi-turn Evaluation Dimensions', fontsize=9, fontweight='bold', loc='left')
    ax_b.text(0.97, 0.97, 'n=20/system', transform=ax_b.transAxes,
              ha='right', fontsize=6, color=COLORS['gray'], va='top')
    
    # Shared legend at the bottom center
    fig.legend(bars_for_legend, labels_for_legend, 
               loc='lower center', ncol=4, fontsize=7,
               framealpha=0.95, edgecolor='none',
               bbox_to_anchor=(0.5, 0.0), 
               columnspacing=1.5, handletextpad=0.4)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for legend
    save_figure(fig, output_dir, 'fig3_stage_analysis')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent
    output_dir = base_path / 'figures_v3'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {base_path}")
    data = load_data(base_path)
    
    print(f"\nGenerating ICML 2026 figures (v3) to: {output_dir}")
    print("=" * 60)
    print(f"  Single column width: {SINGLE_COL_WIDTH} inches")
    print(f"  Full width (figure*): {FULL_WIDTH} inches")
    print("=" * 60)
    
    print("\nFigure 1 (figure*): Main Results")
    fig1_main_results(data, output_dir)
    
    print("\nFigure 2 (figure): Ablation Study")
    fig2_ablations(data, output_dir)
    
    print("\nFigure 3 (figure*): Stage-wise Analysis")
    fig3_stage_analysis(data, output_dir)
    
    print("\n" + "=" * 60)
    print("ICML 2026 Figure Summary:")
    print("-" * 60)
    print("  fig1_main_results.pdf   -> figure* (full width)")
    print("                             (a) Multi-turn quality")
    print("                             (b) Human pairwise evaluation")
    print("")
    print("  fig2_ablations.pdf      -> figure (single column)")
    print("                             Ablation study (n=90)")
    print("")
    print("  fig3_stage_analysis.pdf -> figure* (full width)")
    print("                             (a) Single-turn by stage")
    print("                             (b) Multi-turn dimensions")
    print("=" * 60)
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
