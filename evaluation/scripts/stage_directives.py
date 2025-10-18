"""Stage-specific prompt directives for evaluation agents."""

from __future__ import annotations

from typing import Optional, Tuple


_STAGE_C_DIRECTIVE = (
    "You are preparing a turnkey research execution plan with publication-grade rigor. "
    "Do not request additional context unless absolutely required to avoid catastrophic mistakes. "
    "Provide an EXTREMELY DETAILED, falsifiable plan with the following sections:\n\n"
    "1. Problem framing and goals:\n"
    "   - Specific, measurable objectives with numerical targets (e.g., '≥95% accuracy retention with 4-8× compression')\n"
    "   - Clear success criteria\n\n"
    "2. Experiments (target 5-7 experiments, each with 3-5 ablations):\n"
    "   For EACH experiment provide:\n"
    "   - Falsifiable hypothesis with NUMERICAL acceptance criteria (e.g., 'H1: Method X achieves Y±Z on metric M. Falsify if result <Y-Z.')\n"
    "   - Detailed setup with specific models, datasets, hyperparameters\n"
    "   - Comprehensive ablation studies (list 3-5 variations per experiment: architectures, hyperparameters, data scales, etc.)\n"
    "   - Multiple baselines (include state-of-the-art comparisons)\n"
    "   - Precise evaluation metrics with thresholds\n"
    "   - Expected outcomes with numerical ranges\n\n"
    "3. Timeline:\n"
    "   - Break 6 months into BI-WEEKLY or WEEKLY sprints (not just monthly)\n"
    "   - Each sprint has specific deliverables and measurable milestones\n"
    "   - Include integrated evaluation phases and checkpoint criteria\n\n"
    "4. Resources:\n"
    "   - Specific compute requirements (GPU types, hours, memory)\n"
    "   - Exact datasets with versions/splits\n"
    "   - Precise tooling and libraries with versions\n\n"
    "5. Risks and mitigations:\n"
    "   - Detailed table with probability, impact, and specific mitigation strategies\n\n"
    "6. Integrated recipe or scaling study:\n"
    "   - Explain how experiments combine\n"
    "   - Pareto analysis or sensitivity studies\n\n"
    "Use tools when they yield clearly relevant sources. If retrieved evidence is generic or off-topic, note the limitation and propose how to gather better references instead of forcing a citation. "
    "When citing, prefer [file:page] for attachments and [n] for web results. If no high-confidence references exist, state that explicitly. "
    "CRITICAL: Maximize depth, specificity, and falsifiability. Include 20-30% more detail than you think is needed."
)


def apply_stage_directives(stage_letter: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    stage = (stage_letter or "").strip().upper()
    if stage == "C":
        return f"{_STAGE_C_DIRECTIVE}\n\n{prompt_text}", "stage_c_gpt5_competitive_v2"
    return prompt_text, None
