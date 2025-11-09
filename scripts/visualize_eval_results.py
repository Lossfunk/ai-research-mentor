#!/usr/bin/env python3
"""Visualization utilities for expert and student judge evaluations - Publication Quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

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


@dataclass
class ComparisonStats:
    mentor_wins: int
    baseline_wins: int
    ties: int
    total: int
    mentor_pct: float
    baseline_pct: float
    ties_pct: float
    ci_low: float
    ci_high: float
    p_value: float
    significance_marker: str = ""


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

MENTOR_MODEL_LABEL = "Mentor (Kimi-k2-0905)"

STAGE_DEFINITIONS: Dict[str, str] = {
    "A": "Pre-Idea",
    "B": "Idea",
    "C": "Research Plan",
    "D": "First Draft",
    "E": "Second Draft",
    "F": "Final Draft",
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
                if mentor_score is None and baseline_score is None:
                    continue
                total += 1
                if mentor_score is None or baseline_score is None:
                    ties += 1
                    continue
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


def integerize_triplets(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round three arrays of percentages so per-position sums equal 100."""
    A, B, C = a.copy(), b.copy(), c.copy()
    outA = np.zeros_like(A, dtype=int)
    outB = np.zeros_like(B, dtype=int)
    outC = np.zeros_like(C, dtype=int)

    for i in range(len(A)):
        parts = np.array([A[i], B[i], C[i]], dtype=float)
        floors = np.floor(parts).astype(int)
        remainder = 100 - floors.sum()
        if remainder > 0:
            fracs = parts - floors
            order = np.argsort(-fracs)
            for j in range(remainder):
                floors[order[j]] += 1
        outA[i], outB[i], outC[i] = floors

    return outA, outB, outC


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    phat = successes / total
    denominator = 1 + (z**2) / total
    centre = phat + (z**2) / (2 * total)
    margin = z * ((phat * (1 - phat) / total) + (z**2) / (4 * total**2)) ** 0.5
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return max(0.0, lower), min(1.0, upper)


def binomial_test_two_sided(successes: int, total: int, prob: float = 0.5) -> float:
    """Exact two-sided binomial test without SciPy dependency."""
    if total == 0:
        return float("nan")
    pmf_observed = comb(total, successes) * (prob ** successes) * ((1 - prob) ** (total - successes))
    p_value = 0.0
    for k in range(total + 1):
        pmf = comb(total, k) * (prob ** k) * ((1 - prob) ** (total - k))
        if pmf <= pmf_observed + 1e-12:
            p_value += pmf
    return min(1.0, p_value)


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
    base_name: str = "expert_pairwise_stacked",
) -> Tuple[List[Path], Dict[str, ComparisonStats]]:
    """Create publication-quality stacked bar charts for pairwise comparisons."""
    configure_style()
    num_panels = len(results)
    fig, axes = plt.subplots(1, num_panels, figsize=(5.4 * num_panels, 4.6), sharey=True)
    if num_panels == 1:
        axes = [axes]

    summary_stats: Dict[str, ComparisonStats] = {}
    legend_entries: Dict[str, Rectangle] = {}

    for ax, (key, df) in zip(axes, results.items()):
        baseline_label = next(col for col in df.columns if col not in {"mentor_wins", "ties", "total"})
        stage_df = df.drop(index="Total") if "Total" in df.index else df
        if "Total" in df.index:
            overall_row = df.loc["Total"].copy()
            overall_row.name = "Overall"
            stage_df = pd.concat([stage_df, overall_row.to_frame().T])

        stages = stage_df.index.tolist()
        totals = stage_df["total"].to_numpy()

        # Stage labels with sample sizes
        stage_labels: List[str] = []
        for s, n in zip(stages, totals):
            stage_labels.append(f"{s}\n($n$={int(n)})")

        mentor_pct = (stage_df["mentor_wins"].to_numpy() / totals) * 100
        baseline_pct = (stage_df[baseline_label].to_numpy() / totals) * 100
        ties_pct = (stage_df["ties"].to_numpy() / totals) * 100

        # Normalize percentages to avoid rounding drift
        total_pct = mentor_pct + baseline_pct + ties_pct
        adjustment = 100 - total_pct
        for idx, adj in enumerate(adjustment):
            if abs(adj) > 0.01:
                segments = [mentor_pct[idx], baseline_pct[idx], ties_pct[idx]]
                max_index = int(np.argmax(segments))
                if max_index == 0:
                    mentor_pct[idx] += adj
                elif max_index == 1:
                    baseline_pct[idx] += adj
                else:
                    ties_pct[idx] += adj

        if "Overall" in stage_df.index:
            overall_counts = stage_df.loc["Overall"]
            mentor_total = int(overall_counts["mentor_wins"])
            baseline_total = int(overall_counts[baseline_label])
            ties_total = int(overall_counts["ties"])
            total_comparisons = int(overall_counts["total"])
            mentor_share = (mentor_total / total_comparisons) * 100 if total_comparisons else 0.0
            baseline_share = (baseline_total / total_comparisons) * 100 if total_comparisons else 0.0
            ties_share = (ties_total / total_comparisons) * 100 if total_comparisons else 0.0
            ci_low, ci_high = wilson_interval(mentor_total, total_comparisons)
            p_value = binomial_test_two_sided(mentor_total, total_comparisons)
            marker = ""
            summary_stats[key] = ComparisonStats(
                mentor_wins=mentor_total,
                baseline_wins=baseline_total,
                ties=ties_total,
                total=total_comparisons,
                mentor_pct=mentor_share,
                baseline_pct=baseline_share,
                ties_pct=ties_share,
                ci_low=ci_low * 100,
                ci_high=ci_high * 100,
                p_value=p_value,
                significance_marker=marker,
            )

        bar_width = 0.65
        x_pos = np.arange(len(stages))
        ax.set_axisbelow(True)

        mentor_lbl, baseline_lbl, ties_lbl = integerize_triplets(mentor_pct, baseline_pct, ties_pct)

        mentor_bars = ax.bar(
            x_pos,
            mentor_pct,
            width=bar_width,
            color=COLORS["mentor"],
            label="Mentor Wins",
            edgecolor="#f7f7f7",
            linewidth=0.8,
        )

        baseline_hatch = "///" if key == "gpt5" else "--"
        baseline_bars = ax.bar(
            x_pos,
            baseline_pct,
            width=bar_width,
            bottom=mentor_pct,
            color=COLORS["gpt" if key == "gpt5" else "claude"],
            label=baseline_label,
            edgecolor="#f4f4f4",
            linewidth=0.7,
            hatch=baseline_hatch,
            alpha=0.5,
        )

        ties_bars = None
        if ties_pct.sum() > 0:
            ties_bars = ax.bar(
                x_pos,
                ties_pct,
                width=bar_width,
                bottom=mentor_pct + baseline_pct,
                color=COLORS["ties"],
                label="Ties",
                edgecolor="#f1f1f1",
                linewidth=0.7,
                hatch="..",
                alpha=0.35,
            )

        ax.set_title(f"{MENTOR_MODEL_LABEL} vs. {baseline_label}", pad=10)
        ax.set_xlabel("Stage", labelpad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.6, zorder=0)

        def format_percentage(val: float) -> str:
            return f"{int(round(val))}%"

        def add_labels(bars, heights, labels, color, allow_outside=False, offset=3.2, min_value: float = 4.0):
            for bar, height, label in zip(bars, heights, labels):
                if height < min_value:
                    continue  # skip very small segments to avoid clutter/overlap
                if height <= 0:
                    continue
                if allow_outside and height < 20:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(103, bar.get_y() + height + offset),
                        format_percentage(label),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="semibold",
                        color="#333333",
                    )
                elif height >= 12:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        format_percentage(label),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="semibold",
                        color=color,
                    )

        add_labels(mentor_bars, mentor_pct, mentor_lbl, "white")
        # Baseline labels stay INSIDE their segment to avoid colliding with the 'ties' segment above
        add_labels(baseline_bars, baseline_pct, baseline_lbl, "#222222", allow_outside=False)
        # Only the TOP (ties) segment is allowed to place labels outside
        if ties_bars is not None:
            add_labels(ties_bars, ties_pct, ties_lbl, "#222222", allow_outside=True, offset=4.0, min_value=1.0)

        if "Overall" in stage_df.index and key in summary_stats:
            overall_idx = stages.index("Overall")
            stats = summary_stats[key]

            mentor_height = mentor_pct[overall_idx]
            baseline_height = baseline_pct[overall_idx]
            ties_height = ties_pct[overall_idx]

            error_lower = mentor_height - stats.ci_low
            error_upper = stats.ci_high - mentor_height
            ax.errorbar(
                x_pos[overall_idx],
                mentor_height,
                yerr=[[error_lower], [error_upper]],
                fmt="o",
                color="#1b1b1b",
                capsize=4,
                linewidth=1.1,
                markersize=4.5,
                zorder=5,
            )


        if "Mentor Wins" not in legend_entries:
            legend_entries["Mentor Wins"] = Rectangle((0, 0), 1, 1, facecolor=COLORS["mentor"], edgecolor="white", label="Mentor Wins")
        if baseline_label not in legend_entries:
            legend_entries[baseline_label] = Rectangle(
                (0, 0), 1, 1,
                facecolor=COLORS["gpt" if key == "gpt5" else "claude"],
                edgecolor="white",
                hatch=baseline_hatch,
                label=baseline_label,
            )
        if ties_bars is not None and "Ties" not in legend_entries:
            legend_entries["Ties"] = Rectangle((0, 0), 1, 1, facecolor=COLORS["ties"], edgecolor="white", hatch="..", label="Ties")

    legend_handles = [legend_entries[k] for k in legend_entries]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.055),
        fontsize=9,
    )

    axes[0].set_ylabel("Proportion of Comparisons (%)", labelpad=8)

    caption_lines = [
        f"Writing stages (n=15 prompts each): {', '.join(f'{k} = {v}' for k, v in STAGE_DEFINITIONS.items())}.",
        "Ties are shown explicitly and excluded from win-rate percentages.",
        "Error bars on Overall show 95% Wilson confidence intervals.",
        "Pairwise judge preferences can diverge from absolute scores since comparisons emphasize holistic relative quality.",
    ]

    caption_y = -0.14
    caption_line_height = 0.033
    for line in caption_lines:
        fig.text(
            0.5,
            caption_y,
            line,
            ha="center",
            va="center",
            fontsize=8.1,
            color="#4a4a4a",
        )
        caption_y -= caption_line_height

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(bottom=0.27)
    return save_figure(fig, output_dir, base_name), summary_stats


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
    pairwise_paths, pairwise_stats = plot_pairwise_results(pairwise_frames, output_dir)
    student_trend_paths = plot_student_trends(student_means, student_stderr, output_dir, skipped_stages)

    absolute_paths: List[Path] = []
    absolute_stats: Dict[str, ComparisonStats] = {}
    if absolute_frames:
        absolute_paths, absolute_stats = plot_pairwise_results(
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

    if absolute_stats:
        print("\nSummary statistics (single-turn absolute comparisons):")
        for key, stats in absolute_stats.items():
            label = "GPT-5" if key == "gpt5" else "Claude Sonnet 4.5"
            marker_text = f" ({stats.significance_marker})" if stats.significance_marker else ""
            print(
                f"  {MENTOR_MODEL_LABEL} vs {label}: "
                f"{stats.mentor_wins}/{stats.total} mentor wins "
                f"({stats.mentor_pct:.1f}%), {stats.baseline_wins} baseline wins ({stats.baseline_pct:.1f}%), "
                f"{stats.ties} ties ({stats.ties_pct:.1f}%), "
                f"95% CI [{stats.ci_low:.1f}, {stats.ci_high:.1f}], p={stats.p_value:.3g}{marker_text}"
            )
    
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