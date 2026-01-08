"""Automated analysis and reporting for batch evaluation results."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config_loader import load_metrics_config, metrics_config_digest


class BatchAnalyzer:
    """Analyze and generate reports from batch evaluation results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.annotation_files = list(self.results_dir.glob("**/annotation_placeholders.csv"))
        self.summary_files = list(self.results_dir.glob("**/*_summary.json"))
        self.metrics_config = load_metrics_config()
        self.metrics_digest = metrics_config_digest()
        absolute = self.metrics_config.get("absolute_metrics", {}) or {}
        self.scaled_metrics: List[str] = list(absolute.get("scaled", []) or [])
        self.binary_metrics: List[str] = list(absolute.get("binary", []) or [])
        self.metric_weights: Dict[str, float] = {
            str(metric): float(weight)
            for metric, weight in (self.metrics_config.get("weights", {}) or {}).items()
        }
        iaa_candidates = [
            self.results_dir.parent / "inter_annotator_agreement",
            self.results_dir.parent.parent / "inter_annotator_agreement"
            if len(self.results_dir.parents) >= 2
            else None,
            Path("evals-for-papers/results/inter_annotator_agreement"),
        ]
        iaa_files: List[Path] = []
        for candidate in iaa_candidates:
            if candidate and candidate.exists():
                iaa_files = list(candidate.glob("**/*.json"))
                if iaa_files:
                    break
        self.iaa_files = iaa_files
    
    def load_all_annotations(self) -> Dict[str, pd.DataFrame]:
        """Load all annotation CSVs into DataFrames keyed by system."""
        annotations = {}
        
        for csv_file in self.annotation_files:
            # Extract system name from path
            parts = csv_file.parts
            system = None
            for part in parts:
                if "claude" in part or "gpt" in part or "gemini" in part:
                    system = part
                    break
            
            if not system:
                # Try to infer from parent directory
                system = csv_file.parent.name
            
            if system and csv_file.stat().st_size > 0:  # Not empty file
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty and 'prompt_id' in df.columns:
                        system_label = system
                        if "system_id" in df.columns:
                            non_null_ids = df["system_id"].dropna()
                            if not non_null_ids.empty:
                                system_label = str(non_null_ids.iloc[0])
                        annotations[system_label] = df
                except Exception as e:
                    print(f"Failed to load {csv_file}: {e}")
        
        return annotations
    
    def compute_system_metrics(self, annotations: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Compute aggregated metrics for each system using config-driven definitions."""

        system_metrics: Dict[str, Dict[str, Any]] = {}
        numeric_columns = set(self.scaled_metrics + self.binary_metrics)

        for system, df in annotations.items():
            if df.empty:
                continue

            coerced: Dict[str, pd.Series] = {}
            for column in numeric_columns:
                if column in df.columns:
                    coerced[column] = pd.to_numeric(df[column], errors="coerce")

            metrics: Dict[str, Any] = {
                "prompt_count": int(len(df)),
                "core_scores": {},
                "binary_success": {},
            }

            for metric in self.scaled_metrics:
                series = coerced.get(metric)
                if series is None:
                    continue
                values = series.dropna()
                if values.empty:
                    continue
                metrics["core_scores"][metric] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": int(values.count()),
                }

            for metric in self.binary_metrics:
                series = coerced.get(metric)
                if series is None:
                    continue
                values = series.dropna()
                if values.empty:
                    continue
                success_rate = float((values >= 0.999).mean())
                metrics["binary_success"][metric] = {
                    "success_rate": success_rate,
                    "count": int(values.count()),
                }

            weighted_sum = 0.0
            total_weight = 0.0
            for metric, weight in self.metric_weights.items():
                core_entry = metrics["core_scores"].get(metric)
                if not core_entry:
                    continue
                weighted_sum += core_entry["mean"] * weight
                total_weight += weight

            if total_weight > 0:
                metrics["overall_score"] = {
                    "weighted_mean": weighted_sum / total_weight,
                    "total_weight": float(total_weight),
                }

            system_metrics[system] = metrics

        return system_metrics
    
    def generate_comparison_report(self, system_metrics: Dict[str, Dict]) -> Dict:
        """Generate system comparison analysis."""
        if len(system_metrics) < 2:
            return {"error": "Need at least 2 systems for comparison"}
        
        comparison = {
            "systems": list(system_metrics.keys()),
            "metric_rankings": {},
            "statistical_tests": {},
            "winner_analysis": {},
            "metrics_config_digest": self.metrics_digest,
        }
        
        # Rank systems by each metric
        all_metrics = set()
        for system_data in system_metrics.values():
            all_metrics.update(system_data.get("core_scores", {}).keys())
            all_metrics.update(system_data.get("binary_success", {}).keys())
        
        for metric in sorted(all_metrics):
            rankings = []
            for system, data in system_metrics.items():
                if metric in data.get("core_scores", {}):
                    score = data["core_scores"][metric]["mean"]
                    rankings.append((system, score))
                elif metric in data.get("binary_success", {}):
                    score = data["binary_success"][metric]["success_rate"]
                    rankings.append((system, score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            comparison["metric_rankings"][metric] = rankings
        
        # Determine overall winners
        overall_scores = []
        for system, data in system_metrics.items():
            if "overall_score" in data:
                overall_scores.append((system, data["overall_score"]["weighted_mean"]))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["winner_analysis"]["overall"] = overall_scores
        
        return comparison
    
    def generate_failure_analysis(self, annotations: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze common failure modes across systems."""
        failure_data: Dict[str, Any] = {
            "by_stage": defaultdict(list),
            "low_scores": defaultdict(lambda: defaultdict(list)),
            "binary_failures": defaultdict(lambda: defaultdict(list)),
        }

        for system, df in annotations.items():
            if df.empty:
                continue

            for _, row in df.iterrows():
                stage = str(row.get("stage") or "UNKNOWN").upper()
                failure_data["by_stage"][stage].append(
                    {
                        "system": system,
                        "prompt_id": row.get("prompt_id"),
                    }
                )

            for metric in self.scaled_metrics:
                if metric not in df.columns:
                    continue
                series = pd.to_numeric(df[metric], errors="coerce")
                low_threshold = 0.8  # configurable threshold for scaled metrics
                mask = (series < low_threshold) & series.notna()
                if mask.any():
                    subset = df.loc[mask, ["prompt_id", metric]]
                    failure_data["low_scores"][metric][system].extend(subset.to_dict("records"))

            for metric in self.binary_metrics:
                if metric not in df.columns:
                    continue
                series = pd.to_numeric(df[metric], errors="coerce")
                mask = (series < 0.5) & series.notna()
                if mask.any():
                    subset = df.loc[mask, ["prompt_id", metric]]
                    failure_data["binary_failures"][metric][system].extend(subset.to_dict("records"))

        return {
            "by_stage": {k: list(v) for k, v in failure_data["by_stage"].items()},
            "low_scores": {
                metric: {sys: records for sys, records in systems.items()}
                for metric, systems in failure_data["low_scores"].items()
            },
            "binary_failures": {
                metric: {sys: records for sys, records in systems.items()}
                for metric, systems in failure_data["binary_failures"].items()
            },
        }
    
    def create_detailed_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        annotations = self.load_all_annotations()
        system_metrics = self.compute_system_metrics(annotations)
        comparison = self.generate_comparison_report(system_metrics)
        failures = self.generate_failure_analysis(annotations)
        
        report = {
            "metadata": {
                "analysis_date": pd.Timestamp.now().isoformat(),
                "systems_analyzed": list(annotations.keys()),
                "total_annotations": sum(len(df) for df in annotations.values()),
                "core_metrics_count": len(next(iter(system_metrics.values()), {}).get("core_scores", {})),
                "metrics_version": self.metrics_config.get("version"),
                "metrics_config_digest": self.metrics_digest,
            },
            "system_metrics": system_metrics,
            "comparison": comparison,
            "failure_analysis": failures,
            "recommendations": self._generate_recommendations(system_metrics, failures)
        }

        human_calibration: List[Dict[str, Any]] = []
        for path in self.iaa_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["_source_path"] = str(path)
                human_calibration.append(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to load human calibration artifact {path}: {exc}")
        if human_calibration:
            report["human_calibration"] = human_calibration
            report["metadata"]["human_calibration_count"] = len(human_calibration)
        
        return report
    
    def _generate_recommendations(self, system_metrics: Dict, failures: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check overall performance
        for system, metrics in system_metrics.items():
            overall = metrics.get("overall_score", {}).get("weighted_mean", 0)
            if overall < 1.5:
                recommendations.append(
                    f"{system} shows below-expected performance (overall: {overall:.2f}). Focus on improving core research quality and actionability."
                )
        
        # Check citation validity issues
        for system, metrics in system_metrics.items():
            citation_rate = metrics.get("binary_success", {}).get("citation_validity", {}).get("success_rate", 1.0)
            if citation_rate < 0.9:
                recommendations.append(
                    f"{system} has citation validation issues (success rate: {citation_rate:.1%}). Source verification needs attention."
                )
        
        # Tool routing issues
        for system, metrics in system_metrics.items():
            tool_rate = metrics.get("binary_success", {}).get("tool_routing", {}).get("success_rate", 1.0)
            if tool_rate < 0.8:
                recommendations.append(
                    f"{system} struggles with appropriate tool selection (success rate: {tool_rate:.1%}). Review tool decision logic."
                )
        
        # Stage-specific recommendations
        stage_issues = failures.get("by_stage", {})
        for stage, issues in stage_issues.items():
            if len(issues) > 10:  # Arbitrary threshold for "many issues"
                recommendations.append(
                    f"Stage {stage} shows recurring issues across systems. Review stage-specific guidance and complexity."
                )
        
        return recommendations
    
    def save_report(self, report: Dict, output_path: Optional[Path] = None) -> Path:
        """Save analysis report to file."""
        if output_path is None:
            output_path = self.results_dir / "comprehensive_analysis.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory containing batch results")
    parser.add_argument("--output", help="Output file path for analysis report")
    args = parser.parse_args()
    
    analyzer = BatchAnalyzer(Path(args.results_dir))
    report = analyzer.create_detailed_report()
    
    output_path = analyzer.save_report(report, Path(args.output) if args.output else None)
    print(f"Analysis report saved: {output_path}")
    print(f"Analyzed {report['metadata']['total_annotations']} annotations across {len(report['metadata']['systems_analyzed'])} systems")
    
    # Print key findings
    print("\n=== Key Findings ===")
    for system, metrics in report["system_metrics"].items():
        overall = metrics.get("overall_score", {}).get("weighted_mean", "N/A")
        print(f"{system}: Overall score = {overall}")
    
    if report["recommendations"]:
        print("\n=== Recommendations ===")
        for rec in report["recommendations"]:
            print(f"• {rec}")


if __name__ == "__main__":
    main()
