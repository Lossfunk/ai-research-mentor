from __future__ import annotations

"""
Compute simple inter-annotator agreement (Cohen's kappa) for evaluation CSVs.

Usage:
  uv run python -m evaluation.scripts.compute_iaa \
      --csv-a evaluation/results/analysis_reports/stage_a/annotation_placeholders.csv \
      --csv-b evaluation/results/analysis_reports/stage_a/claude4_vs_gemini25/annotation_placeholders.csv \
      --out evaluation/results/inter_annotator_agreement/stage_a/kappa.json

Notes:
  - Expects rows keyed by prompt_id; computes pairwise kappa per shared metric column.
  - Treats values as categorical strings; numeric values are rounded to two decimals first.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            pid = str(row.get("prompt_id") or "").strip()
            if pid:
                out[pid] = row
        return out


def _normalize(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    try:
        num = float(v)
        return f"{num:.2f}"
    except Exception:
        return v


def _cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    # Map categories to indices
    cats = sorted({*labels_a, *labels_b})
    if not cats:
        return 0.0
    idx = {c: i for i, c in enumerate(cats)}
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return 0.0
    # Confusion matrix
    m = [[0] * len(cats) for _ in cats]
    for a, b in zip(labels_a, labels_b):
        m[idx[a]][idx[b]] += 1
    # Observed agreement
    po = sum(m[i][i] for i in range(len(cats))) / n
    # Expected agreement
    row_sums = [sum(row) for row in m]
    col_sums = [sum(m[i][j] for i in range(len(cats))) for j in range(len(cats))]
    pe = sum((row_sums[i] * col_sums[i]) for i in range(len(cats))) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute Cohen's kappa for two annotator CSVs")
    ap.add_argument("--csv-a", required=True)
    ap.add_argument("--csv-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    a = _load_rows(Path(args.csv_a))
    b = _load_rows(Path(args.csv_b))
    common_ids = sorted(set(a.keys()) & set(b.keys()))
    if not common_ids:
        raise SystemExit("No overlapping prompt_id entries")

    headers = [h for h in a[common_ids[0]].keys() if h not in {"prompt_id", "stage", "annotator", "run_timestamp", "response_path", "tool_trace_path", "additional_notes"}]
    kappas: Dict[str, float] = {}
    for h in headers:
        la: List[str] = []
        lb: List[str] = []
        for pid in common_ids:
            va = _normalize(a[pid].get(h, ""))
            vb = _normalize(b[pid].get(h, ""))
            if va == "" or vb == "":
                continue
            la.append(va)
            lb.append(vb)
        if la and lb and len(la) == len(lb):
            kappas[h] = _cohen_kappa(la, lb)

    out = {
        "csv_a": args.csv_a,
        "csv_b": args.csv_b,
        "n_common": len(common_ids),
        "kappas": kappas,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[iaa] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

