"""Publication-quality plots for multi-turn evaluation results.

This script reads the outputs of ``score_multi_turn.py`` and produces the
figures recommended in the visualization review:

* ``multi_turn_efficiency_panel`` – a 2×2 summary panel showing
  turns-to-success, time-to-success, final scores, and the mean score trajectory
  (with 95 % confidence intervals).
* ``multi_turn_scenarios_faceted`` – small multiples (one per scenario) so the
  reviewer can inspect individual conversations.

Both PNG (for slides) and PDF (600 dpi, Type 42 fonts) versions are written to
``<scoring-dir>/plots``.

Example::

    uv run python -m evaluation.scripts.plot_multi_turn_scores \
        --scoring-dir reports/multi_turn_eval_all5/scoring --threshold 1.6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Use publication-friendly fonts (Type 42) and consistent styling.
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


OKABE_ITO = {
    "multi_turn_eval_mentor": "#0072B2",
    "multi_turn_eval_baseline_sonnet": "#E69F00",
    "multi_turn_eval_baselines_gpt5": "#CC79A7",
}

AGENT_DISPLAY = {
    "multi_turn_eval_mentor": "Mentor",
    "multi_turn_eval_baseline_sonnet": "Claude",
    "multi_turn_eval_baselines_gpt5": "GPT-5",
}

AGENT_CAPTION = {
    "multi_turn_eval_mentor": "Mentor (Kimi-k2-0905)",
    "multi_turn_eval_baseline_sonnet": "Baseline: Claude Sonnet 4.5",
    "multi_turn_eval_baselines_gpt5": "Baseline: GPT-5",
}


@dataclass
class ConversationTurn:
    agent_label: str
    scenario_id: str
    turn_index: int
    overall_score: Optional[float]
    success_at_turn: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-turn evaluation plots")
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        required=True,
        help="Directory containing scores/ and summary CSVs from score_multi_turn.py",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.6,
        help="Success threshold used to mark reward crossings",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for saved figures (PNG). PDF is always vector-based.",
    )
    return parser.parse_args()


def load_summary_dataframe(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df["agent_display"] = df["agent_label"].map(AGENT_DISPLAY)
    df["agent_caption"] = df["agent_label"].map(AGENT_CAPTION)
    df["scenario_display"] = df["scenario_id"].apply(lambda s: s.replace("_", " ").title())
    df["success_minutes"] = df["success_elapsed_seconds"] / 60.0
    return df


def load_turn_dataframe(scores_dir: Path) -> pd.DataFrame:
    records: List[ConversationTurn] = []
    for path in sorted(scores_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        agent = data["agent_label"]
        scenario = data["scenario_id"]
        for turn in data["turns"]:
            records.append(
                ConversationTurn(
                    agent_label=agent,
                    scenario_id=scenario,
                    turn_index=int(turn["turn_index"]),
                    overall_score=turn.get("overall_score"),
                    success_at_turn=bool(turn.get("success_at_turn")),
                )
            )
    df = pd.DataFrame([r.__dict__ for r in records])
    df["agent_display"] = df["agent_label"].map(AGENT_DISPLAY)
    df["scenario_display"] = df["scenario_id"].apply(lambda s: s.replace("_", " ").title())
    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def t_multiplier(n: int) -> float:
    if n <= 1:
        return 0.0
    lookup = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return lookup.get(n, 1.96)


def mean_and_ci(values: Iterable[float]) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, 0.0
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, 0.0
    se = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    return mean, t_multiplier(vals.size) * se


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=15)


def bar_with_points(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, order: List[str], colors: Dict[str, str]) -> List[int]:
    means = []
    cis = []
    counts = []
    for agent in order:
        subset = df.loc[df["agent_label"] == agent, metric].dropna()
        mean, ci = mean_and_ci(subset.values)
        means.append(mean)
        cis.append(ci)
        counts.append(len(subset))

    x = np.arange(len(order))
    ax.bar(x, means, yerr=cis, color=[colors[a] for a in order], alpha=0.85, capsize=6, width=0.6)

    rng = np.random.default_rng(42)
    for idx, agent in enumerate(order):
        subset = df.loc[df["agent_label"] == agent, metric].dropna()
        if subset.empty:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(subset))
        ax.scatter(
            np.full(len(subset), x[idx]) + jitter,
            subset,
            color="black",
            s=35,
            linewidth=0.3,
            alpha=0.8,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY[a] for a in order])
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.1, color="gray", linewidth=0.6)
    ax.xaxis.grid(False)
    _remove_spines(ax)
    return counts


def paired_stats(df: pd.DataFrame, metric: str, reference: str, comparators: List[str]) -> pd.DataFrame:
    rows = []
    ref_values = df.loc[df["agent_label"] == reference].set_index("scenario_id")[metric]
    for agent in comparators:
        comp_values = df.loc[df["agent_label"] == agent].set_index("scenario_id")[metric]
        joined = pd.concat([ref_values, comp_values], axis=1, join="inner", keys=["ref", "comp"]).dropna()
        if joined.empty:
            continue
        diff = joined["ref"] - joined["comp"]
        if np.allclose(diff, 0):
            p_value = 1.0
            t_stat = 0.0
            effect = 0.0
        else:
            try:
                t_stat, p_value = stats.ttest_rel(joined["ref"], joined["comp"])
            except Exception:
                t_stat, p_value = (np.nan, np.nan)
            if diff.std(ddof=1) == 0:
                effect = np.nan
            else:
                effect = diff.mean() / diff.std(ddof=1)
        rows.append(
            {
                "metric": metric,
                "reference": reference,
                "comparator": agent,
                "n": len(joined),
                "t_stat": t_stat,
                "p_value": p_value,
                "cohens_d_z": effect,
            }
        )
    return pd.DataFrame(rows)


def efficiency_panel(summary_df: pd.DataFrame, trajectory_df: pd.DataFrame, threshold: float, out_dir: Path, dpi: int) -> None:
    order = list(AGENT_DISPLAY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    counts_turns = bar_with_points(
        axes[0, 0],
        summary_df,
        metric="success_turn",
        ylabel="Turns to success",
        order=order,
        colors=OKABE_ITO,
    )
    axes[0, 0].set_ylim(bottom=0.5)
    axes[0, 0].set_title("Conversation turns to success")
    add_panel_label(axes[0, 0], "A")

    bar_with_points(
        axes[0, 1],
        summary_df,
        metric="success_minutes",
        ylabel="Minutes to success",
        order=order,
        colors=OKABE_ITO,
    )
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("Elapsed time to success")
    add_panel_label(axes[0, 1], "B")

    bar_with_points(
        axes[1, 0],
        summary_df,
        metric="final_score",
        ylabel="Final judge score",
        order=order,
        colors=OKABE_ITO,
    )
    axes[1, 0].axhline(threshold, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_ylim(1.5, 2.05)
    axes[1, 0].set_title("Final quality after conversation")
    add_panel_label(axes[1, 0], "C")

    for ax in axes.flat:
        _remove_spines(ax)

    traj_stats = (
        trajectory_df.dropna(subset=["overall_score"])
        .groupby(["agent_label", "turn_index"])["overall_score"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    traj_stats["ci"] = traj_stats.apply(
        lambda row: t_multiplier(int(row["count"])) * (row["std"] / np.sqrt(row["count"])) if row["count"] > 1 else 0.0,
        axis=1,
    )

    ax_traj = axes[1, 1]
    max_turn = traj_stats["turn_index"].max() if not traj_stats.empty else 0
    final_values: Dict[str, float] = {}
    for agent in order:
        stats = traj_stats.loc[traj_stats["agent_label"] == agent]
        if stats.empty:
            continue
        linestyle = "-" if agent == "multi_turn_eval_mentor" else "--" if agent == "multi_turn_eval_baseline_sonnet" else "-."
        ax_traj.plot(
            stats["turn_index"],
            stats["mean"],
            color=OKABE_ITO[agent],
            linewidth=3.0,
            linestyle=linestyle,
            label=AGENT_CAPTION[agent],
        )
        if stats["ci"].any():
            ax_traj.fill_between(
                stats["turn_index"],
                stats["mean"] - stats["ci"],
                stats["mean"] + stats["ci"],
                color=OKABE_ITO[agent],
                alpha=0.1,
            )
        final_values[agent] = float(stats["mean"].iloc[-1])

    label_x = max_turn + 0.6
    spacing = 0.05
    sorted_agents = sorted(final_values.items(), key=lambda item: item[1], reverse=True)
    last_y: Optional[float] = None
    for agent, final_y in sorted_agents:
        label_y = final_y if last_y is None else min(final_y, last_y - spacing)
        ax_traj.text(
            label_x,
            label_y,
            AGENT_CAPTION[agent],
            color=OKABE_ITO[agent],
            fontsize=11,
            fontweight="semibold",
            va="center",
            ha="left",
        )
        last_y = label_y

    ax_traj.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax_traj.set_xlabel("Turn index")
    ax_traj.set_ylabel("Overall judge score (mean ± 95% CI)")
    ax_traj.set_ylim(1.5, 2.05)
    ax_traj.set_xlim(1, max_turn + 2.0)
    ax_traj.yaxis.grid(True, alpha=0.1, color="gray", linewidth=0.6)
    ax_traj.xaxis.grid(False)
    ax_traj.legend_.remove() if ax_traj.legend_ else None
    _remove_spines(ax_traj)
    ax_traj.set_title("Average score trajectory across scenarios")
    add_panel_label(ax_traj, "D")

    fig.suptitle("Multi-turn conversation quality and efficiency", y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    png_path = out_dir / "multi_turn_efficiency_panel.png"
    pdf_path = out_dir / "multi_turn_efficiency_panel.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def facet_scenarios(turn_df: pd.DataFrame, threshold: float, out_dir: Path, dpi: int) -> None:
    scenarios = sorted(turn_df["scenario_display"].unique())
    num = len(scenarios)
    cols = 3
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6.5), sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // cols, idx % cols]
        subset = turn_df.loc[turn_df["scenario_display"] == scenario]
        for agent, group in subset.groupby("agent_label"):
            ax.plot(
                group["turn_index"],
                group["overall_score"],
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=OKABE_ITO[agent],
                linestyle="-" if agent == "multi_turn_eval_mentor" else "--" if agent == "multi_turn_eval_baseline_sonnet" else "-.",
                label=AGENT_CAPTION[agent],
            )
            success_turns = group.loc[group["success_at_turn"], "turn_index"]
            if not success_turns.empty:
                first_turn = success_turns.iloc[0]
                score = float(group.loc[group["turn_index"] == first_turn, "overall_score"].iloc[0])
                ax.scatter(
                    first_turn,
                    score,
                    color="white",
                    edgecolor=OKABE_ITO[agent],
                    s=160,
                    linewidth=2.0,
                    marker="o",
                    zorder=5,
                )
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_title(scenario)
        ax.set_xlabel("Turn index")
        ax.set_ylim(1.5, 2.05)
        ax.set_xlim(left=1)
        ax.yaxis.grid(True, alpha=0.1, color="gray", linewidth=0.6)
        ax.xaxis.grid(False)
        _remove_spines(ax)
        if idx % cols == 0:
            ax.set_ylabel("Judge score")

    for empty_idx in range(num, rows * cols):
        axes[empty_idx // cols, empty_idx % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(AGENT_DISPLAY),
        fontsize=12,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.suptitle("Per-scenario multi-turn trajectories", y=0.99)

    png_path = out_dir / "multi_turn_scenarios_faceted.png"
    pdf_path = out_dir / "multi_turn_scenarios_faceted.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def metrics_table(summary_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    order = list(AGENT_DISPLAY.keys())
    records = []
    for agent in order:
        subset = summary_df.loc[summary_df["agent_label"] == agent]
        for metric, label in [
            ("success_turn", "turns_to_success"),
            ("success_minutes", "minutes_to_success"),
            ("final_score", "final_score"),
            ("net_gain", "net_gain"),
        ]:
            mean, ci = mean_and_ci(subset[metric])
            records.append(
                {
                    "agent_label": agent,
                    "agent": AGENT_CAPTION[agent],
                    "metric": label,
                    "mean": mean,
                    "ci95": ci,
                    "n": int(subset[metric].notna().sum()),
                }
            )
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(out_dir / "multi_turn_metrics_summary.csv", index=False)
    return metrics_df


def stats_table(summary_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    reference = "multi_turn_eval_mentor"
    comparators = ["multi_turn_eval_baseline_sonnet", "multi_turn_eval_baselines_gpt5"]
    frames = []
    for metric in ["success_turn", "success_minutes", "final_score", "net_gain"]:
        frames.append(paired_stats(summary_df, metric, reference, comparators))
    stats_df = pd.concat(frames, ignore_index=True)
    stats_df.to_csv(out_dir / "multi_turn_stats_tests.csv", index=False)
    return stats_df


def _remove_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    scoring_dir = args.scoring_dir
    plots_dir = scoring_dir / "plots"
    ensure_output_dir(plots_dir)

    summary_df = load_summary_dataframe(scoring_dir / "summary_conversations.csv")
    turn_df = load_turn_dataframe(scoring_dir / "scores")

    efficiency_panel(summary_df, turn_df, args.threshold, plots_dir, args.dpi)
    facet_scenarios(turn_df, args.threshold, plots_dir, args.dpi)
    metrics_table(summary_df, plots_dir)
    stats_table(summary_df, plots_dir)


if __name__ == "__main__":
    main()
