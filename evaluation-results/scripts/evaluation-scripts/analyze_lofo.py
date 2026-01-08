from __future__ import annotations

"""
Analyze judge-family bias via Leave-One-Family-Out (LOFO) aggregation.

Usage:
  uv run python -m evaluation.scripts.analyze_lofo \
    --stage stage_a \
    --label student_outcome_judge \
    --output evals-for-papers/results/analysis_reports/stage_a/student_outcome_judge/lofo_summary.json

Notes:
  - Auto-detects kind based on files under <stage>/<label>:
      * *_student_judges.json  => student metrics
      * *_judges.json          => expert absolute metrics
  - Groups by system_id (generator) and reports All-judges vs LOFO(generator_family).
  - Uses mean for scaled metrics and majority for binary; bootstrap CI for means, Wilson CI for rates.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .judge_metrics import METRIC_SPECS, MetricSpec  # type: ignore


# ---------------- CI helpers ----------------

def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + (z**2) / total
    centre = phat + (z**2) / (2 * total)
    adj = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * total)) / total)
    lower = max(0.0, (centre - adj) / denom)
    upper = min(1.0, (centre + adj) / denom)
    return (float(lower), float(upper))


def bootstrap_mean_ci(values: np.ndarray, confidence: float = 0.95, samples: int = 2000) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    means = []
    for _ in range(samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return (float(lower), float(upper))


# ---------------- Family utils ----------------

def _family_from_id(identifier: Optional[str]) -> str:
    if not identifier:
        return "unknown"
    value = identifier.strip().lower()
    if value.startswith("openrouter:"):
        rest = value.split(":", 1)[1]
        provider = rest.split("/", 1)[0]
        return provider  # google, openai, anthropic, deepseek, x-ai, etc.
    # heuristics for local ids
    if "claude" in value or "anthropic" in value:
        return "anthropic"
    if "gpt" in value or "openai" in value:
        return "openai"
    if "gemini" in value or "google" in value:
        return "google"
    if "deepseek" in value:
        return "deepseek"
    if "grok" in value or "x-ai" in value or "xai" in value:
        return "x-ai"
    if "mentor" in value:
        return "mentor"
    return "other"


def _is_binary(metric: str) -> bool:
    spec: Optional[MetricSpec] = METRIC_SPECS.get(metric) or METRIC_SPECS.get(metric.rstrip("_score"))
    return bool(spec and spec.kind == "binary")


def _scaled_bounds(metric: str) -> Tuple[float, float]:
    spec: Optional[MetricSpec] = METRIC_SPECS.get(metric) or METRIC_SPECS.get(metric.rstrip("_score"))
    if not spec:
        return (0.0, 2.0)
    return (float(spec.min_score), float(spec.max_score))


# ---------------- Parsing per kind ----------------

def _collect_student(path: Path) -> Optional[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    system_id = (payload.get("model_spec") or {}).get("system_id") or payload.get("system_id")
    gen_family = _family_from_id(system_id)
    judges = payload.get("judges") or []
    if not isinstance(judges, list):
        return None

    # Extract per-judge student metrics
    per_judge: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []
    for rec in judges:
        jname = rec.get("judge")
        fam = _family_from_id(jname)
        parsed = rec.get("parsed") or {}
        scores = parsed.get("scores") or {}
        bins = parsed.get("binary_checks") or {}
        sj_scores = {
            "student_actionability": float(scores.get("actionability_for_student", np.nan)) if scores else np.nan,
            "student_clarity": float(scores.get("clarity_for_student", np.nan)) if scores else np.nan,
            "student_constraint_fit": float(scores.get("constraint_fit_for_student", np.nan)) if scores else np.nan,
            "student_confidence_gain": float(scores.get("confidence_gain_for_student", np.nan)) if scores else np.nan,
        }
        sj_bins = {
            "student_path_ready": float(bins.get("path_ready", np.nan)) if bins else np.nan,
            "student_failure_modes": float(bins.get("failure_modes_flagged", np.nan)) if bins else np.nan,
        }
        per_judge.append((fam, sj_scores, sj_bins))

    return {
        "system_id": system_id or "unknown",
        "generator_family": gen_family,
        "per_judge": per_judge,
        "metrics": [
            "student_actionability",
            "student_clarity",
            "student_constraint_fit",
            "student_confidence_gain",
            "student_path_ready",
            "student_failure_modes",
        ],
        "kind": "student",
    }


def _collect_expert(path: Path) -> Optional[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    system_id = (payload.get("model_spec") or {}).get("system_id") or payload.get("system_id")
    gen_family = _family_from_id(system_id)
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None

    per_metric: Dict[str, List[Tuple[str, float]]] = {}
    for mkey, mval in metrics.items():
        judges = (mval or {}).get("judges") or []
        if not isinstance(judges, list):
            continue
        for rec in judges:
            jname = rec.get("judge")
            fam = _family_from_id(jname)
            score = rec.get("score")
            try:
                s = float(score)
            except Exception:
                continue
            per_metric.setdefault(mkey, []).append((fam, s))

    return {
        "system_id": system_id or "unknown",
        "generator_family": gen_family,
        "per_metric": per_metric,
        "metrics": list(per_metric.keys()),
        "kind": "expert",
    }


@dataclass
class Agg:
    mean: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n: int


def _aggregate_prompt(values: List[float], metric: str) -> Optional[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if not clean:
        return None
    if _is_binary(metric):
        rate = sum(1.0 for v in clean if v >= 0.5) / len(clean)
        return 1.0 if rate >= 0.5 else 0.0
    # scaled
    lo, hi = _scaled_bounds(metric)
    return max(lo, min(hi, float(sum(clean) / len(clean))))


def _summarize_across_prompts(per_prompt: List[Optional[float]], metric: str) -> Agg:
    clean = [v for v in per_prompt if v is not None]
    if not clean:
        return Agg(None, None, None, 0)
    if _is_binary(metric):
        successes = int(sum(1.0 for v in clean if v >= 0.999))
        total = len(clean)
        rate = successes / total
        lo, hi = wilson_interval(successes, total)
        return Agg(rate, lo, hi, total)
    arr = np.array(clean, dtype=float)
    mean = float(np.mean(arr))
    lo, hi = bootstrap_mean_ci(arr)
    return Agg(mean, lo, hi, len(clean))


def analyze(stage: str, label: str, system_filter: Optional[str]) -> Dict[str, Any]:
    stage_letter = stage.lower().replace("stage_", "").upper()
    stage_folder = f"stage_{stage_letter.lower()}"
    base = Path("evals-for-papers/results/analysis_reports") / stage_folder / label
    if not base.exists():
        raise SystemExit(f"Label path not found: {base}")

    student_files = list(base.glob("*_student_judges.json"))
    expert_files = list(base.glob("*_judges.json"))
    kind = "student" if student_files else "expert"
    files = student_files if kind == "student" else expert_files
    if not files:
        raise SystemExit("No judge payloads found for label")

    per_system: Dict[str, Dict[str, Any]] = {}
    for path in files:
        record = _collect_student(path) if kind == "student" else _collect_expert(path)
        if not record:
            continue
        sys_id = record["system_id"]
        if system_filter and sys_id != system_filter:
            continue
        gen_family = record["generator_family"]
        metrics: List[str] = record["metrics"]

        # Prepare accumulators
        entry = per_system.setdefault(sys_id, {"generator_family": gen_family, "metrics": {}})
        bucket = entry["metrics"]
        for m in metrics:
            b = bucket.setdefault(m, {"all": [], "lofo": []})
            # Aggregate per prompt over judges
            if kind == "student":
                fam_scores: Dict[str, List[float]] = {}
                fam_bins: Dict[str, List[float]] = {}
                for fam, s_scores, s_bins in record["per_judge"]:
                    if m in ("student_path_ready", "student_failure_modes"):
                        val = s_bins.get(m, np.nan)
                        fam_bins.setdefault(fam, []).append(val)
                    else:
                        val = s_scores.get(m, np.nan)
                        fam_scores.setdefault(fam, []).append(val)

                # All-judges aggregation
                all_vals = []
                if _is_binary(m):
                    for arr in fam_bins.values():
                        all_vals.extend(arr)
                else:
                    for arr in fam_scores.values():
                        all_vals.extend(arr)
                all_agg = _aggregate_prompt(all_vals, m)
                b["all"].append(all_agg)

                # LOFO aggregation (exclude generator family)
                lofo_vals = []
                if _is_binary(m):
                    for fam, arr in fam_bins.items():
                        if fam == gen_family:
                            continue
                        lofo_vals.extend(arr)
                else:
                    for fam, arr in fam_scores.items():
                        if fam == gen_family:
                            continue
                        lofo_vals.extend(arr)
                lofo_agg = _aggregate_prompt(lofo_vals, m)
                b["lofo"].append(lofo_agg)

            else:  # expert
                fam_scores2: Dict[str, List[float]] = {}
                for fam, s in record["per_metric"].get(m, []):
                    fam_scores2.setdefault(fam, []).append(s)
                all_vals = [s for arr in fam_scores2.values() for s in arr]
                b["all"].append(_aggregate_prompt(all_vals, m))
                lofo_vals = [s for fam, arr in fam_scores2.items() if fam != gen_family for s in arr]
                b["lofo"].append(_aggregate_prompt(lofo_vals, m))

    # Summarize across prompts
    summary: Dict[str, Any] = {
        "stage": stage_letter,
        "label": label,
        "kind": kind,
        "systems": {},
    }
    for sys_id, data in per_system.items():
        family = data["generator_family"]
        metrics = data["metrics"]
        m_out: Dict[str, Any] = {}
        for m, buckets in metrics.items():
            all_agg = _summarize_across_prompts(buckets["all"], m)
            lofo_agg = _summarize_across_prompts(buckets["lofo"], m)
            m_out[m] = {
                "all": {
                    "mean": all_agg.mean,
                    "ci_low": all_agg.ci_low,
                    "ci_high": all_agg.ci_high,
                    "count": all_agg.n,
                },
                "lofo": {
                    "mean": lofo_agg.mean,
                    "ci_low": lofo_agg.ci_low,
                    "ci_high": lofo_agg.ci_high,
                    "count": lofo_agg.n,
                },
            }
        summary["systems"][sys_id] = {"generator_family": family, "metrics": m_out}
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze LOFO (leave-one-family-out) bias for judges")
    p.add_argument("--stage", required=True, help="Stage (A/C/F or stage_a/stage_c/stage_f)")
    p.add_argument("--label", required=True, help="Label directory under analysis_reports/<stage>")
    p.add_argument("--system", dest="system_filter", help="Optional system_id filter")
    p.add_argument("--output", help="Output JSON path (default: <label>/lofo_summary.json)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage
    if not (stage.lower().startswith("stage_") or len(stage) == 1):
        stage = f"stage_{stage.lower()}"
    summary = analyze(stage, args.label, args.system_filter)

    stage_letter = stage.lower().replace("stage_", "").upper()
    out_path = args.output
    if not out_path:
        out_dir = Path("evals-for-papers/results/analysis_reports") / f"stage_{stage_letter.lower()}" / args.label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "lofo_summary.json")
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lofo] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

