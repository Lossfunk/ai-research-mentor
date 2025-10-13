from __future__ import annotations

"""
Compute Spearman and Pearson correlations between two evaluation CSVs.

Usage:
  uv run python -m evaluation.scripts.compute_correlations \
    --csv-a evaluation/results/analysis_reports/stage_a/pilot_check_v2/annotation_placeholders.csv \
    --csv-b evaluation/results/analysis_reports/stage_a/pilot_check_v2_absolute_openrouter_anthropic_claude-sonnet-4.5/annotation_placeholders.csv \
    --out-dir evaluation/results/analysis_reports/stage_a/pilot_check_v2

Outputs:
  - correlations_gpt5_vs_claude.json
  - correlations_gpt5_vs_claude.csv
  - correlations_spearman_bar.png
  - correlations_pearson_bar.png
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


META_COLUMNS: Sequence[str] = (
    "prompt_id",
    "stage",
    "annotator",
    "system_id",
    "run_timestamp",
    "response_path",
    "tool_trace_path",
    "additional_notes",
)


@dataclass
class CorrEntry:
    metric: str
    n: int
    spearman_rho: float | None
    spearman_p: float | None
    pearson_r: float | None
    pearson_p: float | None


def _compute_pair(sa: pd.Series, sb: pd.Series) -> tuple[float | None, float | None, float | None, float | None, int]:
    """Compute Spearman/Pearson for two numeric series with NaN-safe handling.

    Returns (rho, rho_p, r, r_p, n). Returns Nones if not computable (e.g., constant inputs).
    """
    sa_num = pd.to_numeric(sa, errors="coerce")
    sb_num = pd.to_numeric(sb, errors="coerce")
    mask = sa_num.notna() & sb_num.notna()
    if mask.sum() < 3:
        return None, None, None, None, int(mask.sum())

    xa = sa_num[mask].to_numpy()
    xb = sb_num[mask].to_numpy()

    # Pearson (guard constant arrays)
    if np.std(xa) == 0 or np.std(xb) == 0:
        pear_r, pear_p = None, None
    else:
        pear_r, pear_p = pearsonr(xa, xb)

    # Spearman (rank-based; guard constant ranks)
    ra = pd.Series(xa).rank(method="average").to_numpy()
    rb = pd.Series(xb).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        spear_rho, spear_p = None, None
    else:
        spear_rho, spear_p = spearmanr(xa, xb)

    return (
        float(spear_rho) if spear_rho is not None else None,
        float(spear_p) if spear_p is not None else None,
        float(pear_r) if pear_r is not None else None,
        float(pear_p) if pear_p is not None else None,
        int(mask.sum()),
    )


def _plot_bar(values: List[CorrEntry], attr: str, title: str, output_path: Path) -> None:
    """Render horizontal bar chart for the given correlation attribute (spearman_rho or pearson_r)."""
    # Filter to available values
    filtered = [(v.metric, getattr(v, attr)) for v in values if getattr(v, attr) is not None]
    if not filtered:
        return
    filtered.sort(key=lambda x: x[1])
    labels = [m for m, _ in filtered]
    data = [float(v) for _, v in filtered]

    plt.figure(figsize=(10, max(4, 0.4 * len(filtered))))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in data]
    plt.barh(labels, data, color=colors)
    plt.axvline(0.0, color="#666", lw=1)
    plt.title(title)
    plt.xlabel("correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute correlations between two evaluation CSVs")
    ap.add_argument("--csv-a", required=True)
    ap.add_argument("--csv-b", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    a_path = Path(args.csv_a)
    b_path = Path(args.csv_b)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfa = pd.read_csv(a_path)
    dfb = pd.read_csv(b_path)

    if "prompt_id" not in dfa.columns or "prompt_id" not in dfb.columns:
        raise SystemExit("prompt_id column missing in one of the CSV files")

    dfa = dfa.set_index("prompt_id")
    dfb = dfb.set_index("prompt_id")
    common = sorted(set(dfa.index) & set(dfb.index))
    if not common:
        raise SystemExit("No overlapping prompt_id entries")
    dfa = dfa.loc[common]
    dfb = dfb.loc[common]

    candidates = [c for c in dfa.columns if c in dfb.columns and c not in META_COLUMNS]

    results: List[CorrEntry] = []
    for c in candidates:
        spear_rho, spear_p, pear_r, pear_p, n = _compute_pair(dfa[c], dfb[c])
        results.append(
            CorrEntry(
                metric=c,
                n=n,
                spearman_rho=spear_rho,
                spearman_p=spear_p,
                pearson_r=pear_r,
                pearson_p=pear_p,
            )
        )

    # Write JSON / CSV
    json_out = out_dir / "correlations_gpt5_vs_claude.json"
    csv_out = out_dir / "correlations_gpt5_vs_claude.csv"
    json_payload: Dict[str, List[Dict[str, object]]] = {
        "pairs": [
            {
                "metric": r.metric,
                "n": r.n,
                "spearman_rho": r.spearman_rho,
                "spearman_p": r.spearman_p,
                "pearson_r": r.pearson_r,
                "pearson_p": r.pearson_p,
            }
            for r in results
        ]
    }
    json_out.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "metric": r.metric,
                "n": r.n,
                "spearman_rho": r.spearman_rho,
                "spearman_p": r.spearman_p,
                "pearson_r": r.pearson_r,
                "pearson_p": r.pearson_p,
            }
            for r in results
        ]
    ).sort_values("metric")
    df.to_csv(csv_out, index=False)

    # Charts
    _plot_bar(results, "spearman_rho", "Spearman correlation: GPT-5 vs Claude", out_dir / "correlations_spearman_bar.png")
    _plot_bar(results, "pearson_r", "Pearson correlation: GPT-5 vs Claude", out_dir / "correlations_pearson_bar.png")

    # Simple console summary
    top_pos = sorted([r for r in results if r.spearman_rho is not None], key=lambda r: r.spearman_rho, reverse=True)[:5]
    top_neg = sorted([r for r in results if r.spearman_rho is not None], key=lambda r: r.spearman_rho)[:5]
    print("Wrote:")
    print(" ", json_out)
    print(" ", csv_out)
    print(" ", out_dir / "correlations_spearman_bar.png")
    print(" ", out_dir / "correlations_pearson_bar.png")
    print("Top positive (Spearman):", [(r.metric, r.spearman_rho) for r in top_pos])
    print("Top negative (Spearman):", [(r.metric, r.spearman_rho) for r in top_neg])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


