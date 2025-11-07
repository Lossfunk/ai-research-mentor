"""Plot multi-turn evaluation results.

This script reads the aggregated scoring outputs produced by ``score_multi_turn.py``
and generates the visualizations described in the multi-turn evaluation spec:

* Scenario-level growth charts (overall score per turn) with success threshold
  markers for each agent
* Final score vs total turns scatter plot coloured by agent
* Time-to-success distribution plots (turn count and elapsed seconds)

Usage::

    uv run python -m evaluation.scripts.plot_multi_turn_scores \
        --scoring-dir reports/multi_turn_eval_all5/scoring \
        --threshold 1.6

All plots are written to ``<scoring-dir>/plots``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class ConversationSummaryRow:
    agent_label: str
    system_id: str
    scenario_id: str
    total_turns: float
    elapsed_seconds: float
    final_score: float
    net_gain: float
    success_turn: Optional[float]
    success_elapsed_seconds: Optional[float]


@dataclass
class ConversationTurn:
    turn_index: int
    overall_score: Optional[float]
    cumulative_avg: Optional[float]
    success_at_turn: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-turn evaluation results")
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
        help="Success threshold used when annotating the plots",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150)",
    )
    return parser.parse_args()


def load_summary_rows(path: Path) -> List[ConversationSummaryRow]:
    rows: List[ConversationSummaryRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            rows.append(
                ConversationSummaryRow(
                    agent_label=rec["agent_label"],
                    system_id=rec["system_id"],
                    scenario_id=rec["scenario_id"],
                    total_turns=float(rec["total_turns"]),
                    elapsed_seconds=float(rec["elapsed_seconds"]),
                    final_score=float(rec["final_score"]),
                    net_gain=float(rec["net_gain"]),
                    success_turn=_optional_float(rec.get("success_turn")),
                    success_elapsed_seconds=_optional_float(rec.get("success_elapsed_seconds")),
                )
            )
    return rows


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if value == "" or str(value).lower() == "none":
        return None
    return float(value)


def load_turns(scores_dir: Path) -> Dict[Tuple[str, str], List[ConversationTurn]]:
    turns_by_key: Dict[Tuple[str, str], List[ConversationTurn]] = {}
    for path in sorted(scores_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        agent_label = data["agent_label"]
        scenario_id = data["scenario_id"]
        key = (agent_label, scenario_id)
        turns: List[ConversationTurn] = []
        for item in data["turns"]:
            turns.append(
                ConversationTurn(
                    turn_index=int(item["turn_index"]),
                    overall_score=_optional_float(item.get("overall_score")),
                    cumulative_avg=_optional_float(item.get("cumulative_avg")),
                    success_at_turn=bool(item.get("success_at_turn")),
                )
            )
        turns_by_key[key] = turns
    return turns_by_key


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def scenario_growth_plots(
    output_dir: Path,
    threshold: float,
    turns_by_key: Dict[Tuple[str, str], List[ConversationTurn]],
    agent_colors: Dict[str, str],
) -> None:
    scenarios = sorted({scenario for (_, scenario) in turns_by_key.keys()})
    for scenario_id in scenarios:
        fig, ax = plt.subplots(figsize=(7, 4))
        for agent_label, color in agent_colors.items():
            turns = turns_by_key.get((agent_label, scenario_id))
            if not turns:
                continue
            xs = [t.turn_index for t in turns]
            ys = [t.overall_score for t in turns]
            ax.plot(xs, ys, marker="o", color=color, label=agent_label, linewidth=2)

            success_turns = [t for t in turns if t.success_at_turn]
            if success_turns:
                first_success = success_turns[0]
                ax.scatter(
                    first_success.turn_index,
                    first_success.overall_score,
                    color=color,
                    edgecolor="black",
                    zorder=5,
                    s=80,
                )

        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label="threshold")
        ax.set_title(f"Scenario: {scenario_id}")
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Overall judge score")
        ax.set_ylim(0, 2.05)
        ax.set_xlim(left=1)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out_path = output_dir / f"scenario_{scenario_id}_growth.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def final_score_vs_turns(
    output_path: Path,
    threshold: float,
    rows: Iterable[ConversationSummaryRow],
    agent_colors: Dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        color = agent_colors[row.agent_label]
        filled = row.final_score >= threshold
        ax.scatter(
            row.total_turns,
            row.final_score,
            color=color,
            edgecolor="black",
            s=80,
            marker="o" if filled else "x",
            label=row.agent_label,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Agent", loc="lower right")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Total turns")
    ax.set_ylabel("Final score")
    ax.set_ylim(0, 2.05)
    ax.set_title("Final score vs. conversation length")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def time_to_success_plots(
    output_path: Path,
    rows: Iterable[ConversationSummaryRow],
    agent_colors: Dict[str, str],
) -> None:
    data_turns: Dict[str, List[float]] = {}
    data_elapsed: Dict[str, List[float]] = {}
    for row in rows:
        if row.success_turn is not None:
            data_turns.setdefault(row.agent_label, []).append(row.success_turn)
        if row.success_elapsed_seconds is not None:
            data_elapsed.setdefault(row.agent_label, []).append(row.success_elapsed_seconds)

    agents = list(agent_colors.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # Success turns boxplot + strip overlay
    turn_data = [data_turns.get(agent, []) for agent in agents]
    axes[0].boxplot(turn_data, tick_labels=agents, patch_artist=True)
    for idx, agent in enumerate(agents, start=1):
        vals = data_turns.get(agent, [])
        if not vals:
            continue
        axes[0].scatter(
            [idx] * len(vals),
            vals,
            color=agent_colors[agent],
            edgecolor="black",
            s=50,
            zorder=3,
        )
    axes[0].set_title("Success turn distribution")
    axes[0].set_ylabel("Turn index")
    axes[0].grid(alpha=0.2)

    # Success elapsed seconds boxplot + strip overlay
    elapsed_data = [data_elapsed.get(agent, []) for agent in agents]
    axes[1].boxplot(elapsed_data, tick_labels=agents, patch_artist=True)
    for idx, agent in enumerate(agents, start=1):
        vals = data_elapsed.get(agent, [])
        if not vals:
            continue
        axes[1].scatter(
            [idx] * len(vals),
            vals,
            color=agent_colors[agent],
            edgecolor="black",
            s=50,
            zorder=3,
        )
    axes[1].set_title("Success time distribution")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Time-to-success by agent")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scoring_dir: Path = args.scoring_dir
    plots_dir = scoring_dir / "plots"
    ensure_output_dir(plots_dir)

    summary_rows = load_summary_rows(scoring_dir / "summary_conversations.csv")
    turns_by_key = load_turns(scoring_dir / "scores")

    agent_colors = {
        "multi_turn_eval_mentor": "#1b9e77",
        "multi_turn_eval_baseline_sonnet": "#d95f02",
        "multi_turn_eval_baselines_gpt5": "#7570b3",
    }

    scenario_growth_plots(plots_dir, args.threshold, turns_by_key, agent_colors)

    final_score_vs_turns(
        plots_dir / "final_score_vs_turns.png",
        args.threshold,
        summary_rows,
        agent_colors,
    )

    time_to_success_plots(
        plots_dir / "time_to_success.png",
        summary_rows,
        agent_colors,
    )


if __name__ == "__main__":
    main()
