from __future__ import annotations

from typing import Any, Dict, Optional


def math_ground(text_or_math: str, options: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    text = text_or_math or ""
    findings: Dict[str, Any] = {
        "assumptions": [],
        "symbol_glossary": [],
        "dimensional_issues": [],
        "proof_skeleton": [],
        "references": [],
    }

    if "=>" in text or "implies" in text:
        findings["assumptions"].append("Ensure premises for implications are stated.")
    if "O(" in text or "Theta(" in text:
        findings["assumptions"].append("State complexity assumptions and input size definitions.")
    if any(tok in text for tok in ["d/dx", "∂", "partial"]):
        findings["symbol_glossary"].append("Define variables and constants used in derivatives.")
    if any(tok in text for tok in ["||", "norm", "L2", "L1"]):
        findings["symbol_glossary"].append("Clarify norm definitions and spaces.")

    findings["proof_skeleton"].extend([
        "State assumptions and definitions.",
        "Outline lemma(s) with clear dependencies.",
        "Provide main argument and bound(s).",
        "Conclude with conditions for equality or tightness.",
    ])

    return {"findings": findings}
