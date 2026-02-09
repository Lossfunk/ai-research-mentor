#!/usr/bin/env python3
"""Generate appendix-ready figures using Okabe-Ito palette."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "icml2026" / "figures" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}

SYSTEM_ORDER = ["mentor", "gemini", "gpt5", "claude"]
SYSTEM_LABELS = {
    "mentor": "METIS",
    "gemini": "Gemini 3 Pro",
    "gpt5": "GPT-5",
    "claude": "Claude Sonnet-4.5",
}
SYSTEM_COLORS = {
    "mentor": OKABE_ITO["blue"],
    "gemini": OKABE_ITO["green"],
    "gpt5": OKABE_ITO["orange"],
    "claude": OKABE_ITO["purple"],
}
SYSTEM_ID_MAP = {
    "openrouter:moonshotai/kimi-k2-thinking": "mentor",
    "openrouter:google/gemini-3-pro-preview": "gemini",
    "openrouter:openai/gpt-5": "gpt5",
    "openrouter:anthropic/claude-sonnet-4.5": "claude",
}


def configure_style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
    })


def save(fig: plt.Figure, name: str, *, do_tight_layout: bool = True) -> None:
    # Avoid overriding constrained layout when enabled
    if do_tight_layout and fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)


def _draw_stop_reasons(ax: plt.Axes) -> None:
    path = ROOT / "holistic_scoring_v2" / "holistic_results.csv"
    df = pd.read_csv(path)
    df["system"] = df["system_id"].map(SYSTEM_ID_MAP)
    df = df[df["system"].isin(SYSTEM_ORDER)]
    counts = (
        df.groupby(["system", "stop_reason_class"]).size().unstack(fill_value=0)
    )
    for col in ["positive", "ambiguous", "negative"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["positive", "ambiguous", "negative"]]
    fractions = counts.div(counts.sum(axis=1), axis=0)

    bottom = np.zeros(len(SYSTEM_ORDER))
    class_colors = {
        "positive": OKABE_ITO["green"],
        "ambiguous": OKABE_ITO["orange"],
        "negative": OKABE_ITO["vermillion"],
    }
    for key in ["positive", "ambiguous", "negative"]:
        values = [fractions.loc[s, key] for s in SYSTEM_ORDER]
        ax.bar(
            range(len(SYSTEM_ORDER)),
            values,
            bottom=bottom,
            color=class_colors[key],
            label=key.capitalize(),
        )
        bottom += np.array(values)

    ax.set_xticks(range(len(SYSTEM_ORDER)))
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Proportion of conversations")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y")


def _draw_iaa(ax: plt.Axes) -> None:
    path = ROOT / "inter_annotator_agreement" / "iaa_report.json"
    data = json.loads(path.read_text())
    llm = data["llm_judges"]
    dims = [
        ("overall_helpfulness", "Helpfulness"),
        ("student_progress", "Progress"),
        ("mentor_effectiveness", "Effectiveness"),
        ("conversation_efficiency", "Efficiency"),
    ]
    r_vals = [llm[key]["mean_pairwise_correlation"] for key, _ in dims]
    icc_vals = [llm[key]["icc"]["icc"] for key, _ in dims]

    x = np.arange(len(dims))
    width = 0.35

    ax.bar(x - width / 2, r_vals, width, color=OKABE_ITO["sky"], label="Mean r")
    ax.bar(x + width / 2, icc_vals, width, color=OKABE_ITO["purple"], label="ICC")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in dims])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Agreement")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    ax.grid(axis="y")


def _draw_alignment(ax: plt.Axes) -> None:
    path = ROOT / "robustness_checks" / "human_llm_alignment" / "alignment_summary.json"
    data = json.loads(path.read_text())
    order = ["overall", "gpt5", "claude", "gemini"]
    labels = ["Overall", "GPT-5", "Claude", "Gemini"]
    acc = []
    auc = []
    for key in order:
        block = data["overall"] if key == "overall" else data["by_baseline"][key]
        acc.append(block["accuracy"])
        auc.append(block["auc"])

    x = np.arange(len(order))
    width = 0.35

    ax.bar(x - width / 2, acc, width, color=OKABE_ITO["blue"], label="Accuracy")
    ax.bar(x + width / 2, auc, width, color=OKABE_ITO["orange"], label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    ax.grid(axis="y")


def _draw_length_bias(ax: plt.Axes) -> None:
    path = ROOT / "robustness_checks" / "length_bias" / "length_bias_summary.json"
    data = json.loads(path.read_text())
    vals = [data[s]["pearson_r_words"] for s in SYSTEM_ORDER]
    x = np.arange(len(SYSTEM_ORDER))
    ax.bar(x, vals, color=[SYSTEM_COLORS[s] for s in SYSTEM_ORDER])
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Pearson r (words)")
    ax.axhline(0, color=OKABE_ITO["gray"], linewidth=1)
    ax.grid(axis="y")


def _draw_order_bias(ax: plt.Axes) -> None:
    path = ROOT / "robustness_checks" / "order_bias" / "order_bias_summary.json"
    data = json.loads(path.read_text())
    labels = ["Displayed A", "Displayed B"]
    values = [data["displayed_A"]["win_rate"], data["displayed_B"]["win_rate"]]
    ax.bar([0, 1], values, color=[OKABE_ITO["blue"], OKABE_ITO["orange"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Win rate")
    ax.grid(axis="y")


def plot_alignment_and_agreement() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), layout="constrained")
    _draw_alignment(axes[0])
    _draw_iaa(axes[1])
    save(fig, "app_alignment_and_agreement")


def plot_bias_and_stop_reasons() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    _draw_length_bias(axes[0])
    _draw_stop_reasons(axes[1])
    # Legends may overlap in constrained layout; put stop-reasons legend below.
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.25)
    save(fig, "app_bias_and_stop_reasons", do_tight_layout=False)


def main() -> None:
    configure_style()
    plot_alignment_and_agreement()
    plot_bias_and_stop_reasons()


if __name__ == "__main__":
    main()
