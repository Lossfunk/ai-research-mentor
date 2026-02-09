from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_FALLBACK_GUIDELINES: List[Dict[str, Any]] = [
    {
        "id": "G-METHOD-1",
        "title": "Define hypotheses and success metrics up front",
        "content": "State clear hypotheses and measurable success criteria before running experiments.",
        "tags": ["hypothesis", "metrics", "evaluation"],
        "source": "internal_kb",
    },
    {
        "id": "G-METHOD-2",
        "title": "Use strong baselines and controls",
        "content": "Include competitive baselines and appropriate control conditions to isolate effects.",
        "tags": ["baseline", "controls", "comparison"],
        "source": "internal_kb",
    },
    {
        "id": "G-METHOD-3",
        "title": "Justify sample size or power",
        "content": "Report sample sizes and justify them (power analysis or prior benchmarks).",
        "tags": ["sample size", "power", "statistics"],
        "source": "internal_kb",
    },
    {
        "id": "G-METHOD-4",
        "title": "Plan ablations to test key components",
        "content": "Run ablations to attribute gains to specific design choices.",
        "tags": ["ablation", "components"],
        "source": "internal_kb",
    },
    {
        "id": "G-METHOD-5",
        "title": "Ensure reproducibility and reporting",
        "content": "Document seeds, hyperparameters, compute budget, and evaluation protocol.",
        "tags": ["reproducibility", "seeds", "compute"],
        "source": "internal_kb",
    },
]


def _tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 3]


def _load_guidelines() -> Tuple[List[Dict[str, Any]], str]:
    try:
        from ...guidelines_engine.loader import GuidelinesLoader

        loader = GuidelinesLoader()
        return loader.load_guidelines(), "guidelines_kb"
    except Exception:
        return _FALLBACK_GUIDELINES, "fallback_kb"


def _select_guidelines(
    plan_text: str,
    guidelines: List[Dict[str, Any]],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    if not guidelines:
        return []
    tokens = _tokenize(plan_text)
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for guideline in guidelines:
        haystack = " ".join(
            [
                str(guideline.get("title", "")),
                str(guideline.get("content", "")),
                " ".join(guideline.get("tags") or []),
                str(guideline.get("category", "")),
            ]
        ).lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            scored.append((score, guideline))
    if not scored:
        return guidelines[:max_items]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [g for _, g in scored[:max_items]]


def methodology_validate(plan: str, checklist: Optional[List[str]] = None) -> Dict[str, Any]:
    """Guideline-assisted methodology check with heuristic flags."""
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
        reproducibility_gaps.append("Specify seeds and report variance across >=3 runs.")
    if "compute" in text or "gpu" in text:
        reproducibility_gaps.append("Document compute budget and runtime per experiment.")
    if not re.search(r"\b(sample size|participants|subjects|n\s*=\s*\d+)\b", text):
        sample_size_notes = "Report and justify sample size (e.g., power analysis or prior benchmarks)."

    guidelines, guidelines_source = _load_guidelines()
    selected = _select_guidelines(plan or "", guidelines, max_items=5)
    guideline_snippets: List[Dict[str, str]] = []
    for g in selected:
        title = str(g.get("title") or g.get("id") or "Guideline").strip()
        content = str(g.get("content", "")).strip()
        if len(content) > 240:
            content = content[:240] + "..."
        guideline_snippets.append(
            {
                "title": title,
                "content": content,
                "source": str(g.get("source") or g.get("source_domain") or guidelines_source),
            }
        )

    alignment_prompt = (
        "Assess alignment between the student's methodology plan and the guidelines above. "
        "Flag potential issues (e.g., insufficient sample size, missing controls), "
        "note uncertainties, and avoid guarantees. Recommend expert review for final validation."
    )

    return {
        "report": {
            "risks": risks,
            "missing_controls": missing_controls,
            "ablation_suggestions": ablation_suggestions,
            "reproducibility_gaps": reproducibility_gaps,
            "sample_size_notes": sample_size_notes,
            "guidelines": guideline_snippets,
            "guidelines_source": guidelines_source,
            "alignment_prompt": alignment_prompt,
        }
    }
