from __future__ import annotations

"""
Aggregate evaluation artifacts into paper-ready result tables.

Inputs:
  --system mentor=path/to/annotation.csv  (repeatable)
  --meta-dir path/to/meta/jsons
  --pairwise path/to/pairwise/summary.json (optional, repeatable)
Outputs:
  JSON with T1–T4 style table data stored at --output.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statistics import NormalDist

from .config_loader import load_metrics_config


CI_ALPHA = 0.95
BOOTSTRAP_SAMPLES = 2000
RNG = np.random.default_rng(42)


@dataclass
class MetricSummary:
    mean: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    count: int

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "count": self.count,
        }


def wilson_interval(successes: int, total: int, confidence: float = CI_ALPHA) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else NormalDist().inv_cdf((1 + confidence) / 2)
    phat = successes / total
    denom = 1 + (z**2) / total
    centre = phat + (z**2) / (2 * total)
    adj = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * total)) / total)
    lower = max(0.0, (centre - adj) / denom)
    upper = min(1.0, (centre + adj) / denom)
    return (lower, upper)


def bootstrap_mean_ci(values: np.ndarray, confidence: float = CI_ALPHA, samples: int = BOOTSTRAP_SAMPLES) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(samples):
        sample = RNG.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return (float(lower), float(upper))


def summarize_scaled(values: Iterable[float]) -> MetricSummary:
    arr = np.array([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return MetricSummary(None, None, None, 0)
    mean = float(np.mean(arr))
    ci_low, ci_high = bootstrap_mean_ci(arr)
    return MetricSummary(mean, ci_low, ci_high, int(arr.size))


def summarize_binary(values: Iterable[float]) -> MetricSummary:
    arr = np.array([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return MetricSummary(None, None, None, 0)
    successes = int(np.sum(arr >= 0.999))
    total = int(arr.size)
    rate = successes / total
    ci_low, ci_high = wilson_interval(successes, total)
    return MetricSummary(rate, ci_low, ci_high, total)


def load_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_metric_tables(label: str, df: pd.DataFrame, metrics_cfg: Dict[str, any]) -> Dict[str, Dict[str, MetricSummary]]:
    scaled = metrics_cfg.get("absolute_metrics", {}).get("scaled") or []
    binary = metrics_cfg.get("absolute_metrics", {}).get("binary") or []
    target_columns = set(scaled) | set(binary)

    numeric_df = df.copy()
    for column in target_columns:
        if column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    result: Dict[str, Dict[str, MetricSummary]] = {
        "overall": {},
        "by_stage": {},
    }

    for metric in scaled:
        if metric not in numeric_df.columns:
            continue
        summary = summarize_scaled(numeric_df[metric].dropna().astype(float))
        result["overall"][metric] = summary

    for metric in binary:
        if metric not in numeric_df.columns:
            continue
        summary = summarize_binary(numeric_df[metric].dropna().astype(float))
        result["overall"][metric] = summary

    if "stage" in df.columns:
        for stage_value, stage_df in df.groupby("stage"):
            key = f"stage_{stage_value}"
            stage_metrics: Dict[str, MetricSummary] = {}
            numeric_stage = stage_df.copy()
            for column in target_columns:
                if column in numeric_stage.columns:
                    numeric_stage[column] = pd.to_numeric(numeric_stage[column], errors="coerce")
            for metric in scaled:
                if metric not in numeric_stage.columns:
                    continue
                stage_metrics[metric] = summarize_scaled(numeric_stage[metric].dropna().astype(float))
            for metric in binary:
                if metric not in numeric_stage.columns:
                    continue
                stage_metrics[metric] = summarize_binary(numeric_stage[metric].dropna().astype(float))
            result["by_stage"][key] = stage_metrics

    return result


def compute_weighted_overall(metric_summary: Dict[str, MetricSummary], weights: Dict[str, float]) -> Optional[float]:
    total_weight = 0.0
    weighted_sum = 0.0
    for metric, weight in weights.items():
        summary = metric_summary.get(metric)
        if not summary or summary.mean is None:
            continue
        weighted_sum += summary.mean * weight
        total_weight += weight
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def collect_efficiency(meta_dir: Path, system_ids: List[str]) -> Dict[str, float]:
    elapsed: List[float] = []
    tool_counts: List[float] = []
    successes: int = 0
    total: int = 0
    timeout_flags: int = 0

    for meta_path in meta_dir.glob("*_meta.json"):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        sys_id = payload.get("system_id") or payload.get("system")
        if sys_id not in system_ids:
            continue
        elapsed_seconds = payload.get("elapsed_seconds")
        tool_runs_count = payload.get("tool_runs_count")
        success_flag = payload.get("success")
        timeout = payload.get("timeout_exceeded")
        if isinstance(elapsed_seconds, (int, float)):
            elapsed.append(float(elapsed_seconds))
        if isinstance(tool_runs_count, (int, float)):
            tool_counts.append(float(tool_runs_count))
        if isinstance(success_flag, bool):
            successes += int(success_flag)
            total += 1
        if timeout is True:
            timeout_flags += 1

    if not elapsed and not tool_counts:
        return {}

    elapsed_arr = np.array(elapsed) if elapsed else np.array([])
    tool_arr = np.array(tool_counts) if tool_counts else np.array([])
    efficiency = {
        "elapsed_median": float(np.median(elapsed_arr)) if elapsed_arr.size else None,
        "elapsed_p95": float(np.percentile(elapsed_arr, 95)) if elapsed_arr.size else None,
        "elapsed_mean": float(np.mean(elapsed_arr)) if elapsed_arr.size else None,
        "tool_runs_mean": float(np.mean(tool_arr)) if tool_arr.size else None,
        "tool_runs_median": float(np.median(tool_arr)) if tool_arr.size else None,
        "success_rate": successes / total if total else None,
        "timeout_rate": timeout_flags / total if total else None,
        "count": total,
    }
    return efficiency


def summarize_pairwise(summary_paths: List[Path], system_map: Dict[str, str]) -> List[Dict[str, any]]:
    rows: List[Dict[str, any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        systems = payload.get("systems", [])
        wins = payload.get("wins", {})
        ties = int(payload.get("ties", 0))
        total = int(payload.get("total_comparisons", 0))
        if total <= 0 or len(systems) != 2:
            continue
        sys_a, sys_b = systems
        label_a = system_map.get(sys_a, sys_a)
        label_b = system_map.get(sys_b, sys_b)
        wins_a = int(wins.get(sys_a, 0))
        wins_b = int(wins.get(sys_b, 0))
        rate_a = wins_a / total
        rate_b = wins_b / total
        ci_a = wilson_interval(wins_a, total)
        ci_b = wilson_interval(wins_b, total)
        tie_rate = ties / total
        rows.append(
            {
                "pair": f"{label_a} vs {label_b}",
                "system_a": {
                    "id": sys_a,
                    "label": label_a,
                    "wins": wins_a,
                    "win_rate": rate_a,
                    "ci_low": ci_a[0],
                    "ci_high": ci_a[1],
                },
                "system_b": {
                    "id": sys_b,
                    "label": label_b,
                    "wins": wins_b,
                    "win_rate": rate_b,
                    "ci_low": ci_b[0],
                    "ci_high": ci_b[1],
                },
                "ties": ties,
                "tie_rate": tie_rate,
                "total_comparisons": total,
                "judges": payload.get("judges", []),
            }
        )
    return rows


def parse_system_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected format label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("System label cannot be empty")
    path_obj = Path(path.strip())
    if not path_obj.exists():
        raise argparse.ArgumentTypeError(f"Annotation path does not exist: {path}")
    return label, path_obj


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate consolidated evaluation tables (T1–T4).")
    parser.add_argument("--system", action="append", required=True, help="label=path/to/annotation.csv (repeatable)")
    parser.add_argument("--meta-dir", required=True, help="Directory containing *_meta.json files")
    parser.add_argument("--pairwise", action="append", help="Path to pairwise summary.json (repeatable)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    metrics_cfg = load_metrics_config()
    weights = metrics_cfg.get("weights", {}) or {}

    system_entries: Dict[str, Path] = {}
    for raw in args.system:
        label, path = parse_system_arg(raw)
        system_entries[label] = path

    meta_dir = Path(args.meta_dir)
    if not meta_dir.exists():
        raise SystemExit(f"Meta directory does not exist: {meta_dir}")

    system_results: Dict[str, Dict[str, any]] = {}
    system_id_map: Dict[str, str] = {}

    for label, csv_path in system_entries.items():
        df = load_annotation(csv_path)
        metric_tables = compute_metric_tables(label, df, metrics_cfg)
        overall_metrics = metric_tables["overall"]
        weighted_overall = compute_weighted_overall(overall_metrics, weights)
        system_ids = sorted({sid for sid in df.get("system_id", []) if isinstance(sid, str) and sid})
        if not system_ids and "system_id" in df.columns:
            unique_values = df["system_id"].dropna().unique()
            system_ids = [str(unique_values[0])] if len(unique_values) else []
        for sys_id in system_ids:
            system_id_map[sys_id] = label
        efficiency = collect_efficiency(meta_dir, system_ids)
        system_results[label] = {
            "annotation_path": str(csv_path),
            "system_ids": system_ids,
            "metrics": {metric: summary.to_dict() for metric, summary in overall_metrics.items()},
            "metrics_by_stage": {
                stage: {metric: summary.to_dict() for metric, summary in summaries.items()}
                for stage, summaries in metric_tables["by_stage"].items()
            },
            "weighted_overall": weighted_overall,
            "efficiency": efficiency,
        }

    pairwise_paths = [Path(p) for p in args.pairwise or []]
    for path in pairwise_paths:
        if not path.exists():
            raise SystemExit(f"Pairwise summary not found: {path}")

    pairwise_rows = summarize_pairwise(pairwise_paths, system_id_map)

    output_payload = {
        "metadata": {
            "systems": list(system_entries.keys()),
            "meta_dir": str(meta_dir),
            "pairwise_paths": [str(p) for p in pairwise_paths],
            "weights": weights,
            "confidence": CI_ALPHA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "systems": system_results,
        "tables": {
            "T1_pairwise": pairwise_rows,
            "T2_absolute": {
                "metrics": {
                    label: data["metrics"] for label, data in system_results.items()
                },
                "by_stage": {
                    label: data["metrics_by_stage"] for label, data in system_results.items()
                },
                "weighted_overall": {
                    label: data["weighted_overall"] for label, data in system_results.items()
                },
            },
            "T3_efficiency": {label: data["efficiency"] for label, data in system_results.items()},
            "T4_compliance": {
                label: {
                    key: val
                    for key, val in data["metrics"].items()
                    if key in {"tool_routing", "citation_validity", "evidence_integrity"}
                }
                for label, data in system_results.items()
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tables] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
