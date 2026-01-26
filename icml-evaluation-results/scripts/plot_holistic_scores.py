"""Publication-quality plots for holistic multi-turn evaluation results.

Generates figures following A* paper conventions:
- Okabe-Ito colorblind-safe palette
- Type 42 fonts (required by ICML, NeurIPS, ACL)
- 95% confidence intervals
- Clean, minimal aesthetic

Example::

    uv run python -m evaluation-results.scripts.evaluation-scripts.plot_holistic_scores \
        --input-dir icml-evaluation-results/holistic_scoring_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "kimi": "#0072B2",      # Blue - primary system
    "gemini": "#009E73",    # Bluish green
    "gpt5": "#E69F00",      # Orange
    "claude": "#CC79A7",    # Reddish purple
    "neutral": "#999999",   # Gray for reference lines
}

# System display names and colors
SYSTEM_CONFIG = {
    "openrouter:moonshotai/kimi-k2-thinking": {
        "label": "Kimi K2",
        "short": "Kimi K2",
        "color": OKABE_ITO["kimi"],
    },
    "openrouter:google/gemini-3-pro-preview": {
        "label": "Gemini 3 Pro",
        "short": "Gemini 3",
        "color": OKABE_ITO["gemini"],
    },
    "openrouter:openai/gpt-5": {
        "label": "GPT-5",
        "short": "GPT-5",
        "color": OKABE_ITO["gpt5"],
    },
    "openrouter:anthropic/claude-sonnet-4.5": {
        "label": "Claude Sonnet 4.5",
        "short": "Claude 4.5",
        "color": OKABE_ITO["claude"],
    },
}

# Dimension labels
DIMENSION_LABELS = {
    "overall_helpfulness": "Helpfulness",
    "student_progress": "Progress",
    "mentor_effectiveness": "Effectiveness",
    "conversation_efficiency": "Efficiency",
    "holistic_score": "Composite",
}


def configure_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        # CRITICAL: Type 42 fonts required by ICML, NeurIPS, ACL
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,

        # DPI and quality
        "figure.dpi": 150,
        "savefig.dpi": 600,

        # Line and marker sizes
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "axes.linewidth": 0.8,

        # Grid - subtle
        "grid.color": "#CCCCCC",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,

        # Spines - minimal
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Background
        "axes.facecolor": "white",
        "figure.facecolor": "white",

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def load_results(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load holistic results and agent summary."""
    results_path = input_dir / "holistic_results.csv"
    summary_path = input_dir / "agent_summary.csv"

    results_df = pd.read_csv(results_path)
    summary_df = pd.read_csv(summary_path)

    return results_df, summary_df


def compute_dimension_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and 95% CI for each dimension per system."""
    dimensions = ["overall_helpfulness", "student_progress", "mentor_effectiveness",
                  "conversation_efficiency", "holistic_score"]

    records = []
    for system_id in results_df["system_id"].unique():
        subset = results_df[results_df["system_id"] == system_id]
        n = len(subset)

        for dim in dimensions:
            values = subset[dim].dropna()
            if len(values) == 0:
                continue

            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0
            se = std / np.sqrt(len(values)) if len(values) > 1 else 0
            ci95 = 1.96 * se

            records.append({
                "system_id": system_id,
                "dimension": dim,
                "mean": mean,
                "std": std,
                "se": se,
                "ci95": ci95,
                "n": len(values),
            })

    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str) -> List[Path]:
    """Save figure in multiple formats."""
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{base_name}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            transparent=False,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def plot_holistic_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create main comparison bar chart with individual data points."""
    configure_style()

    # Order systems by score (descending)
    system_order = (
        results_df.groupby("system_id")["holistic_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_positions = np.arange(len(system_order))
    bar_width = 0.6

    means = []
    cis = []
    colors = []
    labels = []

    for system_id in system_order:
        subset = results_df[results_df["system_id"] == system_id]["holistic_score"].dropna()
        config = SYSTEM_CONFIG.get(system_id, {"label": system_id, "color": "#666666"})

        mean = subset.mean()
        std = subset.std(ddof=1) if len(subset) > 1 else 0
        se = std / np.sqrt(len(subset)) if len(subset) > 1 else 0
        ci = 1.96 * se

        means.append(mean)
        cis.append(ci)
        colors.append(config["color"])
        labels.append(config.get("short", config["label"]))

    # Draw bars
    bars = ax.bar(
        x_positions,
        means,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )

    # Add error bars (95% CI)
    ax.errorbar(
        x_positions,
        means,
        yerr=cis,
        fmt="none",
        color="#333333",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
        zorder=4,
    )

    # Add individual data points with jitter
    rng = np.random.default_rng(42)
    for idx, system_id in enumerate(system_order):
        subset = results_df[results_df["system_id"] == system_id]["holistic_score"].dropna()
        jitter = rng.uniform(-0.15, 0.15, size=len(subset))
        ax.scatter(
            np.full(len(subset), x_positions[idx]) + jitter,
            subset,
            color="#333333",
            s=25,
            alpha=0.5,
            zorder=5,
            linewidths=0.5,
            edgecolors="white",
        )

    # Add threshold line (no legend entry - label directly)
    threshold = 1.5
    ax.axhline(threshold, color=OKABE_ITO["neutral"], linestyle="--", linewidth=1.2, zorder=2)

    # Label threshold line directly near left edge (axes x, data y)
    ax.text(
        0.02, threshold + 0.02,
        "threshold",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="left",
    )

    # Add value labels on bars
    for idx, (mean, ci) in enumerate(zip(means, cis)):
        ax.text(
            x_positions[idx],
            mean + ci + 0.03,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#333333",
        )

    # Formatting - NO TITLE (ICML: caption serves as title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Holistic Score (0–2)", fontsize=12)
    ax.set_ylim(1.0, 2.0)
    ax.set_yticks(np.arange(1.0, 2.1, 0.2))

    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Add sample size annotations
    for idx, system_id in enumerate(system_order):
        n = len(results_df[results_df["system_id"] == system_id])
        ax.text(
            x_positions[idx],
            0.98,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#666666",
            transform=ax.get_xaxis_transform(),
        )

    # Add annotation for error bars (bottom right)
    ax.text(
        0.98, 0.98,
        "Error bars: 95% CI",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="right",
        va="top",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "holistic_comparison")


def plot_dimension_breakdown(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create grouped bar chart showing all dimensions per system."""
    configure_style()

    stats_df = compute_dimension_stats(results_df)

    # Order systems by holistic score
    system_order = (
        stats_df[stats_df["dimension"] == "holistic_score"]
        .sort_values("mean", ascending=False)["system_id"]
        .tolist()
    )

    dimensions = ["overall_helpfulness", "student_progress", "mentor_effectiveness",
                  "conversation_efficiency"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    num_systems = len(system_order)
    num_dims = len(dimensions)
    bar_width = 0.18
    x_base = np.arange(num_dims)

    for sys_idx, system_id in enumerate(system_order):
        config = SYSTEM_CONFIG.get(system_id, {"label": system_id, "color": "#666666"})

        means = []
        cis = []
        for dim in dimensions:
            row = stats_df[(stats_df["system_id"] == system_id) & (stats_df["dimension"] == dim)]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                cis.append(row["ci95"].values[0])
            else:
                means.append(0)
                cis.append(0)

        offset = (sys_idx - (num_systems - 1) / 2) * bar_width
        x_positions = x_base + offset

        ax.bar(
            x_positions,
            means,
            width=bar_width,
            color=config["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            label=config.get("short", config["label"]),
            zorder=3,
        )

        ax.errorbar(
            x_positions,
            means,
            yerr=cis,
            fmt="none",
            color="#333333",
            capsize=3,
            capthick=1,
            linewidth=1,
            zorder=4,
        )

    # Formatting - NO TITLE (ICML: caption serves as title)
    ax.set_xticks(x_base)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dimensions], fontsize=11)
    ax.set_ylabel("Score (0–2)", fontsize=12)
    ax.set_ylim(1.0, 2.0)
    ax.set_yticks(np.arange(1.0, 2.1, 0.2))

    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        ncol=2,
    )

    # Add annotation for error bars
    ax.text(
        0.02, 0.98,
        "Error bars: 95% CI",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="left",
        va="top",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "dimension_breakdown")


def plot_success_rates(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create simple success rate bar chart with 95% CI (Wilson interval)."""
    configure_style()

    # Order systems by holistic score
    system_order = (
        results_df.groupby("system_id")["holistic_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_positions = np.arange(len(system_order))
    bar_width = 0.6

    success_rates = []
    ci_lows = []
    ci_highs = []
    labels_list = []
    colors_list = []

    for system_id in system_order:
        subset = results_df[results_df["system_id"] == system_id]
        n = len(subset)
        successes = subset["is_success"].sum()
        rate = successes / n * 100
        config = SYSTEM_CONFIG.get(system_id, {"label": system_id, "color": "#666666"})
        labels_list.append(config.get("short", config["label"]))
        colors_list.append(config["color"])

        # Wilson score interval for binomial proportion
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.975)
        phat = successes / n
        denom = 1 + (z**2) / n
        centre = phat + (z**2) / (2 * n)
        margin = z * ((phat * (1 - phat) / n) + (z**2) / (4 * n**2)) ** 0.5
        ci_low = max(0, (centre - margin) / denom) * 100
        ci_high = min(1, (centre + margin) / denom) * 100

        success_rates.append(rate)
        ci_lows.append(rate - ci_low)
        ci_highs.append(ci_high - rate)

    # Draw bars with system colors
    bars = ax.bar(
        x_positions,
        success_rates,
        width=bar_width,
        color=colors_list,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )

    # Add error bars (Wilson 95% CI)
    ax.errorbar(
        x_positions,
        success_rates,
        yerr=[ci_lows, ci_highs],
        fmt="none",
        color="#333333",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
        zorder=4,
    )

    # Add value labels
    for idx, (rate, ci_h) in enumerate(zip(success_rates, ci_highs)):
        ax.text(
            x_positions[idx],
            rate + ci_h + 2,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#333333",
        )

    # Add success counts
    for idx, system_id in enumerate(system_order):
        subset = results_df[results_df["system_id"] == system_id]
        n = len(subset)
        successes = int(subset["is_success"].sum())
        ax.text(
            x_positions[idx],
            5,
            f"{successes}/{n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            fontweight="semibold",
        )

    # Formatting - NO TITLE (ICML: caption serves as title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Add annotation for error bars
    ax.text(
        0.98, 0.98,
        "Error bars: Wilson 95% CI",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="right",
        va="top",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "success_rates")


def plot_efficiency_vs_quality(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create scatter plot of efficiency (turns) vs quality (holistic score)."""
    configure_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Order by holistic score for consistent legend
    system_order = (
        results_df.groupby("system_id")["holistic_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Plot individual points with better alpha
    for system_id in system_order:
        subset = results_df[results_df["system_id"] == system_id]
        config = SYSTEM_CONFIG.get(system_id, {"label": system_id, "color": "#666666"})

        ax.scatter(
            subset["total_turns"],
            subset["holistic_score"],
            color=config["color"],
            s=50,
            alpha=0.6,
            label=config.get("short", config["label"]),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Add system means with larger markers (square instead of diamond to avoid font issues)
    for system_id in system_order:
        subset = results_df[results_df["system_id"] == system_id]
        config = SYSTEM_CONFIG.get(system_id, {"label": system_id, "color": "#666666"})

        mean_turns = subset["total_turns"].mean()
        mean_score = subset["holistic_score"].mean()
        ax.scatter(
            mean_turns,
            mean_score,
            color=config["color"],
            s=180,
            marker="s",
            edgecolors="#333333",
            linewidths=1.5,
            zorder=5,
        )

    # Add threshold line
    ax.axhline(1.5, color=OKABE_ITO["neutral"], linestyle="--", linewidth=1.2,
               zorder=2)

    # Add threshold label near left edge (axes x, data y)
    ax.text(
        0.02, 1.52,
        "threshold",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="left",
    )

    # Formatting - NO TITLE (ICML: caption serves as title)
    ax.set_xlabel("Conversation Length (turns)", fontsize=11)
    ax.set_ylabel("Holistic Score (0–2)", fontsize=11)
    ax.set_ylim(0.95, 1.85)
    ax.set_xlim(0, 85)

    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.xaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=9,
    )

    # Add annotation for square markers (using text, not unicode)
    ax.text(
        0.98, 0.02,
        "Large squares = system means",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="right",
        va="bottom",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "efficiency_vs_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality holistic evaluation plots")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing holistic_results.csv and agent_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for figures (default: <input-dir>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_dir}")
    results_df, summary_df = load_results(input_dir)

    print(f"Generating figures to: {output_dir}")

    # Generate all plots
    paths = []

    print("  Creating holistic comparison chart...")
    paths.extend(plot_holistic_comparison(results_df, output_dir))

    print("  Creating dimension breakdown chart...")
    paths.extend(plot_dimension_breakdown(results_df, output_dir))

    print("  Creating success rates chart...")
    paths.extend(plot_success_rates(results_df, output_dir))

    print("  Creating efficiency vs quality scatter...")
    paths.extend(plot_efficiency_vs_quality(results_df, output_dir))

    print(f"\nGenerated {len(paths)} files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
