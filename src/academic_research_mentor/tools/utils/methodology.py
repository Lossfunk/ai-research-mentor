from __future__ import annotations

from typing import Any, Dict, List, Optional


def methodology_validate(plan: str, checklist: Optional[List[str]] = None) -> Dict[str, Any]:
    text = plan.lower() if plan else ""

    risks: List[str] = []
    missing_controls: List[str] = []
    ablation_suggestions: List[str] = []
    reproducibility_gaps: List[str] = []
    sample_size_notes: Optional[str] = None

    if "leak" in text or "test set" in text and "train" in text:
        risks.append("Potential data leakage between train/test; ensure strict splits.")
    if "baseline" not in text:
        missing_controls.append("Add at least two strong baselines.")
    if "ablation" not in text:
        ablation_suggestions.append("Plan ablations for key components and hyperparameters.")
    if "seed" not in text:
        reproducibility_gaps.append("Specify seeds and report variance across ≥3 runs.")
    if "compute" in text or "gpu" in text:
        reproducibility_gaps.append("Document compute budget and runtime per experiment.")

    return {
        "report": {
            "risks": risks,
            "missing_controls": missing_controls,
            "ablation_suggestions": ablation_suggestions,
            "reproducibility_gaps": reproducibility_gaps,
            "sample_size_notes": sample_size_notes,
        }
    }
