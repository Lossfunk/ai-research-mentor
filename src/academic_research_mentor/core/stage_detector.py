from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# Simple heuristic stage detector based on user turn content.
# Stages:
# A – Pre idea
# B – Idea
# C – Research plan
# D – First draft
# E – Second draft
# F – Final


_STAGE_DEFS: Dict[str, Tuple[str, List[str]]] = {
    "A": (
        "Pre idea",
        [
            "what should i work on",
            "clarify",
            "clarifying",
            "disambiguate",
            "scope",
            "problem selection",
            "brainstorm",
            "explore ideas",
            "open ended",
            "getting started",
            "not sure",
        ],
    ),
    "B": (
        "Idea",
        [
            "idea",
            "hypothesis",
            "proposal",
            "novel",
            "intuition",
            "direction",
            "angle",
            "approach sketch",
            "framing",
            "baseline choice",
            "compare baselines",
        ],
    ),
    "C": (
        "Research plan",
        [
            "plan",
            "methodology",
            "evaluation",
            "metrics",
            "dataset",
            "experiment",
            "experiments",
            "feasibility",
            "baseline plan",
            "ablation plan",
            "risks",
            "design",
            "hyperparameter",
            "setup",
            "configuration",
            "planning",
            "study design",
        ],
    ),
    "D": (
        "First draft",
        [
            "draft",
            "baseline",
            "ablations",
            "initial results",
            "preliminary results",
            "writeup",
            "writing first draft",
            "figure draft",
            "analysis",
            "interpret",
            "reporting",
            "results",
            "visualization",
            "summary",
        ],
    ),
    "E": (
        "Second draft",
        [
            "revision",
            "revise",
            "reviewer",
            "critic",
            "checklist",
            "figure checks",
            "math check",
            "proof check",
            "polish",
            "presentation",
            "flow",
            "clarity",
            "narrative",
            "story",
            "tone",
            "visuals",
            "communication",
        ],
    ),
    "F": (
        "Final",
        [
            "venue",
            "submission",
            "camera ready",
            "final",
            "simulate reviews",
            "ready to submit",
            "cover letter",
            "author response",
            "checklist",
            "last mile",
        ],
    ),
}


def _stage_marker(text: str) -> Optional[str]:
    lowered = text.lower()
    for code in ("A", "B", "C", "D", "E", "F"):
        marker = f"stage {code.lower()}"
        if marker in lowered or marker + "+" in lowered:
            return code
    return None


def detect_stage(user_text: str) -> Dict[str, object]:
    """Detect an approximate research stage for the current user turn.

    Returns a dict with keys: code, name, confidence.
    """
    text = (user_text or "").strip().lower()
    if not text:
        return {"code": "A", "name": _STAGE_DEFS["A"][0], "confidence": 0.30}

    marker = _stage_marker(text)
    if marker:
        return {"code": marker, "name": _STAGE_DEFS[marker][0], "confidence": 0.8}

    best_code = "A"
    best_score = 0
    total_hits = 0
    for code, (name, keywords) in _STAGE_DEFS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        total_hits += score
        if score > best_score:
            best_score = score
            best_code = code

    # Confidence: basic normalization by hits
    if best_score == 0:
        conf = 0.35 if any(w in text for w in ("idea", "plan", "draft", "submit")) else 0.30
    else:
        conf = min(0.9, 0.45 + 0.1 * best_score)

    return {"code": best_code, "name": _STAGE_DEFS[best_code][0], "confidence": round(conf, 2)}


__all__ = ["detect_stage"]


