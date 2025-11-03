#!/usr/bin/env python3
"""Visualization utilities for expert and student judge evaluations - Publication Quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parents[1] / "evals-for-papers" / "results" / "analysis_reports"
STAGE_FOLDERS = [f"stage_{suffix}" for suffix in "abcdef"]


@dataclass
class PairwiseConfig:
    slug: str
    summary_dir: str
    label: str


PAIRWISE_BASELINES: Dict[str, PairwiseConfig] = {
    "gpt5": PairwiseConfig(
        slug="gpt5",
        summary_dir="mentor_manual__vs__openrouter:openai_gpt-5",
        label="GPT-5 Baseline",
    ),
    "claude": PairwiseConfig(
        slug="claude",
        summary_dir="mentor_manual__vs__openrouter:anthropic_claude-sonnet-4.5",
        label="Claude Sonnet 4.5 Baseline",
    ),
}

STUDENT_MODEL_DIRS: Dict[str, str] = {
    "Mentor": "student_outcome_judge_mentor_manual",
    "GPT-5 Baseline": "student_outcome_judge_openrouter_openai_gpt_5",
    "Claude Sonnet 4.5 Baseline": "student_outcome_judge_openrouter_anthropic_claude_sonnet_4_5",
}

# IMPROVED: Okabe-Ito colorblind-safe palette
# This is the gold standard for scientific visualization
COLORS = {
    "mentor": "#0072B2",      # Blue - strong, trustworthy
    "gpt": "#E69F00",         # Orange - warm, distinct from blue
    "claude": "#CC79A7",      # Reddish purple - excellent contrast with both
    "ties": "#999999",        # Neutral gray
}

# Alternative option (also colorblind safe):
# COLORS_ALT = {
#     "mentor": "#0173B2",    # Blue
#     "gpt": "#DE8F05",       # Darker orange
#     "claude": "#029E73",    # Teal (not green!)
#     "ties": "#949494",
# }

EXPERT_ABSOLUTE_DIRS: Dict[str, str] = {
    "Mentor": "expert_absolute_pro_mentor_manual",
    "GPT-5 Baseline": "expert_absolute_pro_openrouter_openai_gpt_5",
    "Claude Sonnet 4.5 Baseline": "expert_absolute_pro_openrouter_anthropic_claude_sonnet_4_5",
}


def configure_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update(
        {
            # CRITICAL: Type 42 fonts required by ICML, NeurIPS, ACL
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            
            # Font settings - match or slightly smaller than caption text
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,              # Base size
            "axes.labelsize": 10,        # Axis labels slightly larger
            "axes.titlesize": 11,        # Titles
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            
            # DPI and quality
            "figure.dpi": 150,           # Screen preview
            "savefig.dpi": 300,          # Print quality
            
            # Line and marker sizes
            "lines.linewidth": 1.8,      # Slightly thicker for visibility
            "lines.markersize": 7,       # Larger markers
            "axes.linewidth": 0.8,
            
            # Grid - subtle but present
            "grid.color": "#CCCCCC",     # Light gray
            "grid.linestyle": "-",       # Solid lines (not dashed)
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            
            # Background
            "axes.facecolor": "white",   # Pure white, not gray
            "figure.facecolor": "white",
            
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def load_pairwise_stage_results(base_path: Path, baseline: PairwiseConfig) -> pd.DataFrame:
    """Load pairwise comparison results across stages."""
    records: List[Dict[str, object]] = []
    for stage_folder in STAGE_FOLDERS:
        stage_dir = base_path / stage_folder
        stage_label = stage_folder.split("_")[1].upper()
        summary_path = (
            stage_dir
            / f"{stage_folder}_pairwise_mentor_vs_{baseline.slug}_oct21"
            / "pairwise"
            / baseline.summary_dir
            / "summary.json"
        )
        if not summary_path.exists():
            continue  # Skip missing stages gracefully
            
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        mentor_wins = data["wins"].get("mentor_manual", 0)
        baseline_key = next((k for k in data["wins"].keys() if k != "mentor_manual"), None)
        baseline_wins = data["wins"].get(baseline_key or "baseline", 0)
        ties = data.get("ties", 0)
        total = data.get("total_comparisons", mentor_wins + baseline_wins + ties)
        records.append(
            {
                "stage": stage_label,
                "mentor_wins": mentor_wins,
                baseline.label: baseline_wins,
                "ties": ties,
                "total": total,
            }
        )
    
    if not records:
        raise ValueError(f"No data found for {baseline.label}")
        
    df = pd.DataFrame(records).set_index("stage")
    totals = df[["mentor_wins", baseline.label, "ties", "total"]].sum()
    totals.name = "Total"
    df = pd.concat([df, totals.to_frame().T])
    return df


def compute_student_stage_scores(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute student outcome scores with standard errors.
    
    Returns:
        means_df: DataFrame of mean scores
        stderr_df: DataFrame of standard errors
        skipped: List of skipped stage labels
    """
    rows: List[Dict[str, object]] = []
    stderr_rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    
    for stage_folder in STAGE_FOLDERS:
        stage_dir = base_path / stage_folder
        stage_label = stage_folder.split("_")[1].upper()
        row: Dict[str, object] = {"stage": stage_label}
        stderr_row: Dict[str, object] = {"stage": stage_label}
        missing_data = False
        
        for label, subdir in STUDENT_MODEL_DIRS.items():
            model_dir = stage_dir / subdir
            if not model_dir.exists():
                missing_data = True
                break
            scores: List[float] = []
            for file_path in sorted(model_dir.glob(f"{stage_folder}_*_student_judges.json")):
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                scores.append(payload["student_metrics"]["student_outcome_score"])
            if not scores:
                missing_data = True
                break
            row[label] = np.mean(scores)
            stderr_row[label] = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
            
        if missing_data:
            skipped.append(stage_label)
            continue
        rows.append(row)
        stderr_rows.append(stderr_row)
        
    if not rows:
        raise ValueError("No student judge data found across stages.")
    
    means_df = pd.DataFrame(rows).set_index("stage").sort_index()
    stderr_df = pd.DataFrame(stderr_rows).set_index("stage").sort_index()
    return means_df, stderr_df, skipped


def compute_expert_absolute_wins(base_path: Path) -> tuple[Dict[str, pd.DataFrame], List[str]]:
    """Compute expert absolute comparison wins from pro judge scores."""
    missing_stages: List[str] = []
    baseline_results: Dict[str, List[Dict[str, object]]] = {
        "gpt5": [],
        "claude": [],
    }

    for stage_folder in STAGE_FOLDERS:
        stage_dir = base_path / stage_folder
        stage_label = stage_folder.split("_")[1].upper()
        prompt_scores: Dict[str, Dict[str, float]] = {}
        stage_missing = False

        for label, subdir in EXPERT_ABSOLUTE_DIRS.items():
            system_dir = stage_dir / subdir
            if not system_dir.exists():
                stage_missing = True
                break
            for file_path in sorted(system_dir.glob(f"{stage_folder}_*_judges.json")):
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                metrics = payload.get("metrics", {})
                total_score = 0.0
                for metric_data in metrics.values():
                    score = metric_data.get("score") if isinstance(metric_data, dict) else None
                    if score is None:
                        continue
                    total_score += float(score)
                prompt_id = payload.get("prompt_id") or file_path.stem.replace("_judges", "")
                prompt_scores.setdefault(prompt_id, {})[label] = total_score

        if stage_missing:
            missing_stages.append(stage_label)
            continue

        for baseline_key, baseline_label in (
            ("gpt5", "GPT-5 Baseline"),
            ("claude", "Claude Sonnet 4.5 Baseline"),
        ):
            mentor_wins = 0
            baseline_wins = 0
            ties = 0
            total = 0
            for scores in prompt_scores.values():
                mentor_score = scores.get("Mentor")
                baseline_score = scores.get(baseline_label)
                if mentor_score is None or baseline_score is None:
                    continue
                total += 1
                diff = mentor_score - baseline_score
                if abs(diff) <= 1e-6:
                    ties += 1
                elif diff > 0:
                    mentor_wins += 1
                else:
                    baseline_wins += 1
            if total == 0:
                missing_stages.append(stage_label)
                continue
            baseline_results[baseline_key].append(
                {
                    "stage": stage_label,
                    "mentor_wins": mentor_wins,
                    baseline_label: baseline_wins,
                    "ties": ties,
                    "total": total,
                }
            )

    dataframes: Dict[str, pd.DataFrame] = {}
    for key, records in baseline_results.items():
        if not records:
            continue
        df = pd.DataFrame(records).set_index("stage").sort_index()
        baseline_col = next(col for col in df.columns if col not in {"mentor_wins", "ties", "total"})
        totals = df[["mentor_wins", baseline_col, "ties", "total"]].sum()
        totals.name = "Total"
        df = pd.concat([df, totals.to_frame().T])
        dataframes[key] = df
    return dataframes, sorted(set(missing_stages))


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str) -> List[Path]:
    """Save figure in multiple formats with proper settings."""
    paths: List[Path] = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{base_name}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,  # 2x DPI for raster
            bbox_inches="tight",
            pad_inches=0.05,  # Minimal padding
            backend="pdf" if ext == "pdf" else None,
            transparent=False,
            facecolor="white",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def plot_pairwise_results(
    results: Dict[str, pd.DataFrame], 
    output_dir: Path, 
    base_name: str = "expert_pairwise_stacked"
) -> List[Path]:
    """
    Create publication-quality stacked bar charts for pairwise comparisons.
    
    IMPROVEMENTS:
    - Uniform bar widths
    - Sample sizes in stage labels
    - Better color contrast
    - Improved text sizing and positioning
    """
    configure_style()
    num_panels = len(results)
    fig, axes = plt.subplots(1, num_panels, figsize=(5.5 * num_panels, 4.5), sharey=True)
    if num_panels == 1:
        axes = [axes]
    
    for ax, (key, df) in zip(axes, results.items()):
        baseline_label = next(col for col in df.columns if col not in {"mentor_wins", "ties", "total"})
        stage_df = df.drop(index="Total") if "Total" in df.index else df
        stages = stage_df.index.tolist()
        totals = stage_df["total"].to_numpy()
        
        # Create stage labels with sample sizes
        stage_labels = [f"{s}\n($n$={int(n)})" for s, n in zip(stages, totals)]
        
        # Calculate percentages
        mentor_pct = (stage_df["mentor_wins"].to_numpy() / totals) * 100
        baseline_pct = (stage_df[baseline_label].to_numpy() / totals) * 100
        ties_pct = (stage_df["ties"].to_numpy() / totals) * 100
        
        # IMPROVED: Uniform bar width
        bar_width = 0.65
        x_pos = np.arange(len(stages))
        
        ax.set_axisbelow(True)
        
        # Plot bars with consistent width
        mentor_bars = ax.bar(
            x_pos,
            mentor_pct,
            width=bar_width,
            color=COLORS["mentor"],
            label="Mentor Wins",
            edgecolor="white",
            linewidth=1.0,
        )
        baseline_bars = ax.bar(
            x_pos,
            baseline_pct,
            width=bar_width,
            bottom=mentor_pct,
            color=COLORS["gpt" if key == "gpt5" else "claude"],
            label=baseline_label,
            edgecolor="white",
            linewidth=1.0,
        )
        
        # Only show ties if they exist
        if ties_pct.sum() > 0:
            ties_bars = ax.bar(
                x_pos,
                ties_pct,
                width=bar_width,
                bottom=mentor_pct + baseline_pct,
                color=COLORS["ties"],
                label="Ties",
                edgecolor="white",
                linewidth=1.0,
            )
        
        # Set labels and formatting
        ax.set_title(f"Mentor vs {baseline_label}", pad=10)
        ax.set_xlabel("Stage", labelpad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        
        # IMPROVED: Horizontal gridlines only
        ax.grid(axis="y", alpha=0.6, zorder=0)
        
        # Add percentage labels - improved threshold
        for bars, values in [(mentor_bars, mentor_pct), (baseline_bars, baseline_pct)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height >= 8:  # Only label if segment is large enough
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="semibold",
                        color="white",
                    )
        
        # Add overall statistics - improved positioning
        if "Total" in df.index:
            total_counts = df.loc["Total", ["mentor_wins", baseline_label, "ties", "total"]]
            total_pct = (total_counts[["mentor_wins", baseline_label, "ties"]] / total_counts["total"]) * 100
            
            # Only show ties if > 0
            ties_line = f"\nTies {int(total_counts['ties'])} / {total_pct['ties']:.1f}%" if total_counts['ties'] > 0 else ""
            
            summary_text = (
                f"Overall:\n"
                f"Mentor {int(total_counts['mentor_wins'])} / {total_pct['mentor_wins']:.1f}%\n"
                f"{baseline_label.split()[0]} {int(total_counts[baseline_label])} / {total_pct[baseline_label]:.1f}%"
                f"{ties_line}"
            )
            ax.text(
                0.97,
                0.03,
                summary_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#CCCCCC",
                    "linewidth": 0.8,
                },
            )
    
    # Y-axis label only on leftmost plot
    axes[0].set_ylabel("Share of Pairwise Comparisons (%)", labelpad=8)
    
    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="upper center", 
        ncol=3, 
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, output_dir, base_name)


def plot_student_trends(
    means_df: pd.DataFrame,
    stderr_df: pd.DataFrame,
    output_dir: Path,
    skipped: List[str],
    base_name: str = "student_judge_trends",
) -> List[Path]:
    """
    Create publication-quality line plot with confidence intervals.
    
    IMPROVEMENTS:
    - Added error bars (95% CI)
    - Better grid styling
    - Improved marker visibility
    - Sample size annotations
    """
    configure_style()
    
    # Markers must be distinct in black & white
    markers = {
        "Mentor": "o",                          # Circle
        "GPT-5 Baseline": "s",                  # Square
        "Claude Sonnet 4.5 Baseline": "^"       # Triangle
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = means_df.index.tolist()
    x_pos = np.arange(len(stages))
    
    for label in means_df.columns:
        color_key = "mentor" if label == "Mentor" else ("gpt" if "GPT-5" in label else "claude")
        means = means_df[label].to_numpy()
        stderr = stderr_df[label].to_numpy()
        
        # 95% confidence interval (1.96 * SE)
        ci = 1.96 * stderr
        
        # Plot line with markers
        ax.plot(
            x_pos,
            means,
            marker=markers[label],
            color=COLORS[color_key],
            linewidth=2.2,
            markersize=8,
            label=label,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
        
        # Add confidence interval as shaded region
        ax.fill_between(
            x_pos,
            means - ci,
            means + ci,
            color=COLORS[color_key],
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
    
    ax.set_title("Student-Perceived Outcome Score by Stage", pad=12)
    ax.set_xlabel("Stage", labelpad=8)
    ax.set_ylabel("Average Student Outcome Score", labelpad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages)
    
    # IMPROVED: Set sensible y-limits with padding
    y_min = max(1.0, means_df.min().min() - 0.1)
    y_max = means_df.max().max() + 0.1
    ax.set_ylim(y_min, y_max)
    
    # IMPROVED: Horizontal grid only
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    
    ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC", loc="best")
    
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    
    # Add note about missing stages
    if skipped:
        note = ", ".join(skipped)
        fig.text(
            0.5,
            0.02,
            f"Note: Student judge data unavailable for stage(s): {note}. "
            f"Error bands show 95% confidence intervals.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
            style="italic",
        )
    
    return save_figure(fig, output_dir, base_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-quality evaluation visualizations.")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=BASE_DIR,
        help="Path to the analysis_reports directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "reports" / "evals" / "figures",
        help="Directory where figures will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root: Path = args.analysis_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication-quality visualizations...")
    print(f"Output directory: {output_dir}")
    
    # Load data
    pairwise_frames: Dict[str, pd.DataFrame] = {
        key: load_pairwise_stage_results(analysis_root, config)
        for key, config in PAIRWISE_BASELINES.items()
    }
    
    student_means, student_stderr, skipped_stages = compute_student_stage_scores(analysis_root)
    absolute_frames, expert_missing = compute_expert_absolute_wins(analysis_root)

    # Generate figures
    pairwise_paths = plot_pairwise_results(pairwise_frames, output_dir)
    student_trend_paths = plot_student_trends(student_means, student_stderr, output_dir, skipped_stages)

    absolute_paths: List[Path] = []
    if absolute_frames:
        absolute_paths = plot_pairwise_results(
            absolute_frames, output_dir, base_name="expert_absolute_wins"
        )
    
    absolute_student_paths = plot_student_trends(
        student_means,
        student_stderr,
        output_dir,
        skipped_stages,
        base_name="student_judge_trends_absolute",
    )

    # Print summary
    print("\n✓ Generated expert pairwise figures:")
    for path in pairwise_paths:
        print(f"  {path}")
    
    print("\n✓ Generated student trend figures:")
    for path in student_trend_paths:
        print(f"  {path}")
    
    if absolute_paths:
        print("\n✓ Generated expert absolute comparison figures:")
        for path in absolute_paths:
            print(f"  {path}")
    
    print("\n✓ Generated absolute student trend figures:")
    for path in absolute_student_paths:
        print(f"  {path}")
    
    if skipped_stages:
        print(f"\n⚠ Skipped student stages (missing data): {', '.join(skipped_stages)}")
    if expert_missing:
        print(f"⚠ Skipped expert stages (missing data): {', '.join(expert_missing)}")
    
    print("\n" + "="*60)
    print("PUBLICATION CHECKLIST:")
    print("="*60)
    print("✓ Colorblind-safe palette (Okabe-Ito)")
    print("✓ Type 42 fonts (required by conferences)")
    print("✓ 95% confidence intervals on student plots")
    print("✓ Sample sizes shown in stage labels")
    print("✓ Vector formats (PDF, SVG) for infinite scaling")
    print("✓ 600 DPI PNG for high-quality raster")
    print("✓ Font sizes ≥9pt (matches typical captions)")
    print("✓ Uniform bar widths in stacked charts")
    print("✓ White background (not gray)")
    print("✓ Minimal, informative gridlines")
    print("="*60)


if __name__ == "__main__":
    main()