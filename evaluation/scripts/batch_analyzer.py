"""Automated analysis and reporting for batch evaluation results."""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd
import numpy as np


class BatchAnalyzer:
    """Analyze and generate reports from batch evaluation results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.annotation_files = list(self.results_dir.glob("**/annotation_placeholders.csv"))
        self.summary_files = list(self.results_dir.glob("**/*_summary.json"))
    
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
                        annotations[system] = df
                except Exception as e:
                    print(f"Failed to load {csv_file}: {e}")
        
        return annotations
    
    def compute_system_metrics(self, annotations: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Compute aggregated metrics for each system."""
        system_metrics = {}
        
        # Core metrics to analyze
        core_metrics = [
            "research_quality_score",
            "actionability_score", 
            "persona_fit_score",
            "evidence_quality_score",
            "question_asking_score"
        ]
        
        binary_metrics = [
            "citation_validity",
            "tool_routing",
            "stage_appropriateness", 
            "constraint_handling"
        ]
        
        for system, df in annotations.items():
            if df.empty:
                continue
            
            metrics = {
                "prompt_count": len(df),
                "success_rate": 1.0,  # All annotated = successful generation
                "core_scores": {},
                "binary_success": {}
            }
            
            # Core metric statistics
            for metric in core_metrics:
                if metric in df.columns:
                    scores = df[metric].dropna()
                    if not scores.empty:
                        metrics["core_scores"][metric] = {
                            "mean": float(scores.mean()),
                            "median": float(scores.median()),
                            "std": float(scores.std()),
                            "min": float(scores.min()),
                            "max": float(scores.max()),
                            "count": int(scores.count())
                        }
            
            # Binary metric success rates  
            for metric in binary_metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if not values.empty:
                        success_rate = (values == 1.0).mean()
                        metrics["binary_success"][metric] = {
                            "success_rate": float(success_rate),
                            "count": int(values.count())
                        }
            
            # Compute overall score (weighted average)
            overall_scores = []
            metric_weights = {
                "research_quality_score": 1.2,
                "actionability_score": 1.0,
                "persona_fit_score": 1.0, 
                "evidence_quality_score": 0.8,
                "question_asking_score": 0.8
            }
            
            total_weight = 0
            weighted_sum = 0
            for metric, weight in metric_weights.items():
                if metric in metrics["core_scores"]:
                    weighted_sum += metrics["core_scores"][metric]["mean"] * weight
                    total_weight += weight
            
            if total_weight > 0:
                metrics["overall_score"] = {
                    "weighted_mean": weighted_sum / total_weight,
                    "total_weight": float(total_weight)
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
            " statistical_tests": {},
            "winner_analysis": {}
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
        failure_data = {
            "by_stage": defaultdict(list),
            "by_persona": defaultdict(list),
            "low_scores": defaultdict(lambda: defaultdict(list)),
            "binary_failures": defaultdict(lambda: defaultdict(list))
        }
        
        for system, df in annotations.items():
            if df.empty:
                continue
            
            for _, row in df.iterrows():
                prompt_id = row.get('prompt_id', '')
                stage = prompt_id.split('_')[1] if '_' in prompt_id else 'unknown'
                
                # Filter prompts with metadata if available
                metadata = {}
                if hasattr(df, '_metadata_cache'):
                    metadata = df._metadata_cache.get(prompt_id, {})
                
                # Track by stage
                failure_data["by_stage"][stage].append((system, row))
                
                # Track by persona if metadata available
                persona = metadata.get('persona', 'unknown')
                failure_data["by_persona"][persona].append((system, row))
        
        # Analyze low-scoring prompts
        core_metrics = ["research_quality_score", "actionability_score", "persona_fit_score"]
        
        for system, df in annotations.items():
            for metric in core_metrics:
                if metric in df.columns:
                    low_score_threshold = 0.8  # Below 80% of max score
                    low_score_mask = (df[metric] < low_score_threshold) & df[metric].notna()
                    low_scoring = df[low_score_mask]
                    
                    if not low_scoring.empty:
                        for _, row in low_scoring.iterrows():
                            failure_data["low_scores"][metric][system].append(row)
        
        return {k: dict(v) for k, v in failure_data.items()}  # Convert defaultdict to dict
    
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
                "core_metrics_count": len(system_metrics.get(list(system_metrics.keys())[0], {}).get("core_scores", {}))
            },
            "system_metrics": system_metrics,
            "comparison": comparison,
            "failure_analysis": failures,
            "recommendations": self._generate_recommendations(system_metrics, failures)
        }
        
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
