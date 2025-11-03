#!/usr/bin/env python3
"""Visualization utilities for expert and student judge evaluations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


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

COLORS = {
    "mentor": "#1b6ca8",
    "gpt": "#f28e2b",
    "claude": "#59a14f",
    "ties": "#8f8f8f",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#f9f9f9",
            "grid.color": "#d7d7d7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def load_pairwise_stage_results(base_path: Path, baseline: PairwiseConfig) -> pd.DataFrame:
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
            raise FileNotFoundError(f"Missing summary for {baseline.label} in {stage_folder}: {summary_path}")
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
    df = pd.DataFrame(records).set_index("stage")
    totals = df[["mentor_wins", baseline.label, "ties", "total"]].sum()
    totals.name = "Total"
    df = pd.concat([df, totals.to_frame().T])
    return df


def compute_student_stage_scores(base_path: Path) -> tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for stage_folder in STAGE_FOLDERS:
        stage_dir = base_path / stage_folder
        stage_label = stage_folder.split("_")[1].upper()
        row: Dict[str, object] = {"stage": stage_label}
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
            row[label] = sum(scores) / len(scores)
        if missing_data:
            skipped.append(stage_label)
            continue
        rows.append(row)
    if not rows:
        raise ValueError("No student judge data found across stages.")
    return pd.DataFrame(rows).set_index("stage").sort_index(), skipped


EXPERT_ABSOLUTE_DIRS: Dict[str, str] = {
    "Mentor": "expert_absolute_pro_mentor_manual",
    "GPT-5 Baseline": "expert_absolute_pro_openrouter_openai_gpt_5",
    "Claude Sonnet 4.5 Baseline": "expert_absolute_pro_openrouter_anthropic_claude_sonnet_4_5",
}


def compute_expert_absolute_wins(base_path: Path) -> tuple[Dict[str, pd.DataFrame], List[str]]:
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
                    stage_missing = True
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
    paths: List[Path] = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{base_name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_pairwise_results(
    results: Dict[str, pd.DataFrame], output_dir: Path, base_name: str = "expert_pairwise_stacked"
) -> List[Path]:
    configure_style()
    num_panels = len(results)
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 5), sharey=True)
    if num_panels == 1:
        axes = [axes]
    for ax, (key, df) in zip(axes, results.items()):
        baseline_label = next(col for col in df.columns if col not in {"mentor_wins", "ties", "total"})
        stage_df = df.drop(index="Total") if "Total" in df.index else df
        stages = stage_df.index.tolist()
        totals = stage_df["total"].to_numpy()
        mentor_pct = (stage_df["mentor_wins"].to_numpy() / totals) * 100
        baseline_pct = (stage_df[baseline_label].to_numpy() / totals) * 100
        ties_pct = (stage_df["ties"].to_numpy() / totals) * 100
        ax.set_axisbelow(True)
        mentor_container = ax.bar(
            stages,
            mentor_pct,
            color=COLORS["mentor"],
            label="Mentor Wins",
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
        )
        baseline_container = ax.bar(
            stages,
            baseline_pct,
            bottom=mentor_pct,
            color=COLORS["gpt" if key == "gpt5" else "claude"],
            label=baseline_label,
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
        )
        ties_container = ax.bar(
            stages,
            ties_pct,
            bottom=mentor_pct + baseline_pct,
            color=COLORS["ties"],
            label="Ties",
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(f"Mentor vs {baseline_label}")
        ax.set_xlabel("Stage")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.45)

        for container, values in (
            (mentor_container, mentor_pct),
            (baseline_container, baseline_pct),
            (ties_container, ties_pct),
        ):
            labels = [f"{val:.0f}%" if val >= 6 else "" for val in values]
            ax.bar_label(
                container,
                labels=labels,
                label_type="center",
                fontsize=9,
                color="white",
                fontweight="semibold",
                padding=2,
            )

        if "Total" in df.index:
            total_counts = df.loc["Total", ["mentor_wins", baseline_label, "ties", "total"]]
            total_pct = (total_counts[["mentor_wins", baseline_label, "ties"]] / total_counts["total"]) * 100
            summary_text = (
                "Overall (wins / %):\n"
                f"Mentor {int(total_counts['mentor_wins'])} / {total_pct['mentor_wins']:.1f}\n"
                f"{baseline_label} {int(total_counts[baseline_label])} / {total_pct[baseline_label]:.1f}\n"
                f"Ties {int(total_counts['ties'])} / {total_pct['ties']:.1f}"
            )
            ax.text(
                0.98,
                0.05,
                summary_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#cfcfcf"},
            )
    axes[0].set_ylabel("Share of Pairwise Comparisons (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, base_name)


def plot_student_trends(
    df: pd.DataFrame,
    output_dir: Path,
    skipped: List[str],
    base_name: str = "student_judge_trends",
) -> List[Path]:
    configure_style()
    markers = {"Mentor": "o", "GPT-5 Baseline": "s", "Claude Sonnet 4.5 Baseline": "^"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    stages = df.index.tolist()
    for label in df.columns:
        color_key = "mentor" if label == "Mentor" else ("gpt" if "GPT-5" in label else "claude")
        ax.plot(
            stages,
            df[label].to_numpy(),
            marker=markers[label],
            color=COLORS[color_key],
            linewidth=2.8,
            markersize=8,
            label=label,
        )
    ax.set_title("Student Judge Outcome Score by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Average Student Outcome Score")
    ax.set_ylim(bottom=max(0.95, df.min().min() - 0.05), top=df.max().max() + 0.05)
    ax.grid(True, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if skipped:
        note = ", ".join(skipped)
        fig.text(
            0.5,
            0.015,
            f"Note: Student judge data unavailable for stage(s): {note}.",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
        ax.annotate(
            "F",
            xy=(len(stages) - 0.05, df.iloc[-1].max()),
            xytext=(len(stages) - 0.05, df.iloc[-1].max() + 0.05),
            textcoords="data",
            ha="right",
            va="bottom",
            color="#777777",
            fontsize=11,
            arrowprops={"arrowstyle": "-|>", "color": "#bbbbbb", "linewidth": 1.0},
        )
    return save_figure(fig, output_dir, base_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create evaluation visualizations.")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=BASE_DIR,
        help="Path to the analysis_reports directory (default: project evals-for-papers/results/analysis_reports)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "reports" / "evals" / "figures",
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root: Path = args.analysis_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_frames: Dict[str, pd.DataFrame] = {
        key: load_pairwise_stage_results(analysis_root, config)
        for key, config in PAIRWISE_BASELINES.items()
    }
    student_scores, skipped_stages = compute_student_stage_scores(analysis_root)
    absolute_frames, expert_missing = compute_expert_absolute_wins(analysis_root)

    pairwise_paths = plot_pairwise_results(pairwise_frames, output_dir)
    student_trend_paths = plot_student_trends(student_scores, output_dir, skipped_stages)

    absolute_paths: List[Path] = []
    if absolute_frames:
        absolute_paths = plot_pairwise_results(
            absolute_frames, output_dir, base_name="expert_absolute_wins"
        )
    absolute_student_paths = plot_student_trends(
        student_scores,
        output_dir,
        skipped_stages,
        base_name="student_judge_trends_absolute",
    )

    print("Generated pairwise figures:")
    for path in pairwise_paths:
        print(f"  {path}")
    print("Student trend figure:")
    for path in student_trend_paths:
        print(f"  {path}")
    if absolute_paths:
        print("Expert Absolute Pro comparison figures:")
        for path in absolute_paths:
            print(f"  {path}")
    print("Student trend figure (absolute run):")
    for path in absolute_student_paths:
        print(f"  {path}")
    if skipped_stages:
        print("Skipped student judge stages due to missing data:", ", ".join(skipped_stages))
    if expert_missing:
        print("Skipped expert stages due to missing data:", ", ".join(expert_missing))


if __name__ == "__main__":
    main()
