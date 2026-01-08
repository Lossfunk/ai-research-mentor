from __future__ import annotations

"""
Render publication-ready tables (CSV/Markdown) and figures (PNG) from
tables_pilot_check_v2.json produced by generate_results_tables.py.

Usage:
  python -m evaluation.scripts.render_publication_assets \
    --input evals-for-papers/results/analysis_reports/stage_a/pilot_check_v2_derived/tables_pilot_check_v2.json \
    --out-dir evals-for-papers/results/analysis_reports/stage_a/pilot_check_v2_derived

Outputs:
  <out-dir>/tables/T{1,2,3,4}_*.csv
  <out-dir>/tables/T{1,2,3,4}_*.md (compact Markdown variants)
  <out-dir>/figures/F1_pareto.png, F2_metric_deltas.png, F3_pairwise_winrates.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _write_csv_and_md(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    # Compact Markdown (pipe table)
    md = "| " + " | ".join(df.columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(str(row[c]) if row[c] is not None else "" for c in df.columns) + " |\n"
    md_path.write_text(md, encoding="utf-8")


def render_t1_pairwise(payload: Dict, out_dir: Path) -> pd.DataFrame:
    rows = payload.get("tables", {}).get("T1_pairwise", [])
    records: List[Dict] = []
    for r in rows:
        pair = r.get("pair")
        total = r.get("total_comparisons")
        ties = r.get("ties")
        for tag in ("system_a", "system_b"):
            s = r.get(tag) or {}
            records.append(
                {
                    "pair": pair,
                    "system_label": s.get("label"),
                    "system_id": s.get("id"),
                    "wins": s.get("wins"),
                    "win_rate": round(s.get("win_rate", 0.0), 4) if s.get("win_rate") is not None else None,
                    "ci_low": round(s.get("ci_low", 0.0), 4) if s.get("ci_low") is not None else None,
                    "ci_high": round(s.get("ci_high", 0.0), 4) if s.get("ci_high") is not None else None,
                    "ties": ties,
                    "total": total,
                }
            )
    df = pd.DataFrame.from_records(records)
    tables_dir = out_dir / "tables"
    _write_csv_and_md(df, tables_dir / "T1_pairwise.csv", tables_dir / "T1_pairwise.md")
    return df


def render_t2_absolute(payload: Dict, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t2 = payload.get("tables", {}).get("T2_absolute", {})
    metrics = t2.get("metrics", {})  # {label: {metric: {mean, ci_low, ci_high, count}}}
    by_stage = t2.get("by_stage", {})  # {label: {stage: {metric: summary}}}
    weighted_overall = t2.get("weighted_overall", {})  # {label: value}

    rows: List[Dict] = []
    for label, metric_map in metrics.items():
        for metric, summary in (metric_map or {}).items():
            rows.append(
                {
                    "system_label": label,
                    "metric": metric,
                    "mean": round(summary.get("mean", 0.0), 4) if summary.get("mean") is not None else None,
                    "ci_low": round(summary.get("ci_low", 0.0), 4) if summary.get("ci_low") is not None else None,
                    "ci_high": round(summary.get("ci_high", 0.0), 4) if summary.get("ci_high") is not None else None,
                    "count": summary.get("count"),
                }
            )
    df_abs = pd.DataFrame.from_records(rows)

    rows_stage: List[Dict] = []
    for label, stage_map in by_stage.items():
        for stage, metric_map in (stage_map or {}).items():
            for metric, summary in (metric_map or {}).items():
                rows_stage.append(
                    {
                        "stage": stage,
                        "system_label": label,
                        "metric": metric,
                        "mean": round(summary.get("mean", 0.0), 4) if summary.get("mean") is not None else None,
                        "ci_low": round(summary.get("ci_low", 0.0), 4) if summary.get("ci_low") is not None else None,
                        "ci_high": round(summary.get("ci_high", 0.0), 4) if summary.get("ci_high") is not None else None,
                        "count": summary.get("count"),
                    }
                )
    df_stage = pd.DataFrame.from_records(rows_stage)

    df_overall = pd.DataFrame(
        [(label, round(value, 4) if value is not None else None) for label, value in weighted_overall.items()],
        columns=["system_label", "weighted_overall"],
    )

    tables_dir = out_dir / "tables"
    _write_csv_and_md(df_abs, tables_dir / "T2_absolute.csv", tables_dir / "T2_absolute.md")
    _write_csv_and_md(df_stage, tables_dir / "T2_by_stage.csv", tables_dir / "T2_by_stage.md")
    _write_csv_and_md(df_overall, tables_dir / "T2_weighted_overall.csv", tables_dir / "T2_weighted_overall.md")
    return df_abs, df_stage, df_overall


def render_t3_efficiency(payload: Dict, out_dir: Path) -> pd.DataFrame:
    t3 = payload.get("tables", {}).get("T3_efficiency", {})
    rows: List[Dict] = []
    for label, stats in (t3 or {}).items():
        rows.append(
            {
                "system_label": label,
                "elapsed_median": stats.get("elapsed_median"),
                "elapsed_p95": stats.get("elapsed_p95"),
                "elapsed_mean": stats.get("elapsed_mean"),
                "tool_runs_mean": stats.get("tool_runs_mean"),
                "tool_runs_median": stats.get("tool_runs_median"),
                "success_rate": stats.get("success_rate"),
                "timeout_rate": stats.get("timeout_rate"),
                "count": stats.get("count"),
            }
        )
    df = pd.DataFrame.from_records(rows)
    tables_dir = out_dir / "tables"
    _write_csv_and_md(df, tables_dir / "T3_efficiency.csv", tables_dir / "T3_efficiency.md")
    return df


def render_t4_compliance(payload: Dict, out_dir: Path) -> pd.DataFrame:
    t4 = payload.get("tables", {}).get("T4_compliance", {})
    rows: List[Dict] = []
    for label, metric_map in (t4 or {}).items():
        for metric, summary in (metric_map or {}).items():
            rows.append(
                {
                    "system_label": label,
                    "metric": metric,
                    "mean": round(summary.get("mean", 0.0), 4) if summary.get("mean") is not None else None,
                    "ci_low": round(summary.get("ci_low", 0.0), 4) if summary.get("ci_low") is not None else None,
                    "ci_high": round(summary.get("ci_high", 0.0), 4) if summary.get("ci_high") is not None else None,
                    "count": summary.get("count"),
                }
            )
    df = pd.DataFrame.from_records(rows)
    tables_dir = out_dir / "tables"
    _write_csv_and_md(df, tables_dir / "T4_compliance.csv", tables_dir / "T4_compliance.md")
    return df


def figure_pareto_quality_efficiency(df_overall: pd.DataFrame, df_eff: pd.DataFrame, figures_dir: Path) -> None:
    merged = pd.merge(df_overall, df_eff, on="system_label", how="inner")
    plt.figure(figsize=(6, 4))
    for _, row in merged.iterrows():
        x = row.get("elapsed_median")
        y = row.get("weighted_overall")
        if x is None or y is None:
            continue
        plt.scatter([x], [y], label=row["system_label"], s=60)
        plt.annotate(row["system_label"], (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Median latency (s)")
    plt.ylabel("Weighted quality score")
    plt.title("F1: Quality vs Efficiency (Pareto)")
    plt.tight_layout()
    (figures_dir / "F1_pareto.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "F1_pareto.png", dpi=150)
    plt.close()


def figure_metric_deltas(df_abs: pd.DataFrame, figures_dir: Path) -> None:
    # Expect two systems; compute mentor - baseline per metric when both present
    pivot = df_abs.pivot_table(index="metric", columns="system_label", values="mean")
    if pivot.shape[1] < 2:
        return
    systems = list(pivot.columns)
    a, b = systems[0], systems[1]
    deltas = (pivot[a] - pivot[b]).dropna().sort_values()
    plt.figure(figsize=(7, max(3, 0.4 * len(deltas))))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in deltas]
    plt.barh(deltas.index, deltas.values, color=colors)
    plt.axvline(0.0, color="#666", lw=1)
    plt.xlabel(f"Delta ({a} − {b})")
    plt.title("F2: Per-metric mean difference")
    plt.tight_layout()
    plt.savefig(figures_dir / "F2_metric_deltas.png", dpi=150)
    plt.close()


def figure_pairwise_winrates(df_t1: pd.DataFrame, figures_dir: Path) -> None:
    # One pair expected; plot both systems with error bars
    if df_t1.empty:
        return
    plt.figure(figsize=(5, 3))
    labels = df_t1["system_label"].tolist()
    rates = df_t1["win_rate"].tolist()
    err_low = df_t1["win_rate"].tolist()  # placeholder to compute symmetric error bars
    err_high = df_t1["win_rate"].tolist()
    # Build asymmetric errors
    lower = (df_t1["win_rate"] - df_t1["ci_low"]).clip(lower=0.0)
    upper = (df_t1["ci_high"] - df_t1["win_rate"]).clip(lower=0.0)
    yerr = [lower.to_list(), upper.to_list()]
    x = range(len(labels))
    plt.bar(x, rates, yerr=yerr, capsize=4, color=["#5bc0de", "#f0ad4e"])
    plt.xticks(list(x), labels)
    plt.ylim(0, 1)
    plt.ylabel("Win rate")
    plt.title("F3: Pairwise win rates (95% CI)")
    plt.tight_layout()
    plt.savefig(figures_dir / "F3_pairwise_winrates.png", dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Render publication tables and figures from tables JSON")
    ap.add_argument("--input", required=True, help="Path to tables JSON output")
    ap.add_argument("--out-dir", required=True, help="Directory to write tables/figures")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    tables_dir, figures_dir = _ensure_dirs(out_dir)

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    # Tables
    df_t1 = render_t1_pairwise(payload, out_dir)
    df_abs, df_stage, df_overall = render_t2_absolute(payload, out_dir)
    df_t3 = render_t3_efficiency(payload, out_dir)
    df_t4 = render_t4_compliance(payload, out_dir)

    # Figures
    figure_pareto_quality_efficiency(df_overall, df_t3, figures_dir)
    figure_metric_deltas(df_abs, figures_dir)
    figure_pairwise_winrates(df_t1, figures_dir)

    print(f"[render] wrote tables -> {tables_dir}")
    print(f"[render] wrote figures -> {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

