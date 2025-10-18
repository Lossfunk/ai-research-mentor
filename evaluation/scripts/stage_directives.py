"""Stage-specific prompt directives for evaluation agents."""

from __future__ import annotations

from typing import Optional, Tuple


_STAGE_C_DIRECTIVE = (
    "You are preparing a turnkey research execution plan with publication-grade rigor. "
    "Do not request additional context unless absolutely required to avoid catastrophic mistakes. "
    "Provide a highly detailed, falsifiable plan with the following sections:\n\n"
    "1. Problem framing and goals:\n"
    "   - Specific, measurable objectives with numerical targets (e.g., '≥95% accuracy retention with 4-8× compression')\n"
    "   - Clear success criteria\n\n"
    "2. Experiments (target 4-6 experiments, each with 2-4 ablations):\n"
    "   For EACH experiment provide:\n"
    "   - Falsifiable hypothesis with NUMERICAL acceptance criteria (e.g., 'H1: Method X achieves Y±Z on metric M. Falsify if result <Y-Z.')\n"
    "   - Detailed setup with specific models, datasets, hyperparameters\n"
    "   - Focused ablation studies (list 2-4 variations per experiment: architectures, hyperparameters, data scales, etc.)\n"
    "   - Multiple baselines (include state-of-the-art comparisons)\n"
    "   - Precise evaluation metrics with thresholds\n"
    "   - Expected outcomes with numerical ranges\n\n"
    "3. Timeline:\n"
    "   - Break 6 months into monthly milestones, adding bi-weekly checkpoints where it is natural\n"
    "   - Each milestone has specific deliverables and measurable criteria for success\n"
    "   - Include evaluation phases and go/no-go decision points\n\n"
    "4. Resources:\n"
    "   - Specific compute requirements (GPU types, hours, memory)\n"
    "   - Exact datasets with versions/splits\n"
    "   - Tooling and libraries with versions\n\n"
    "5. Risks and mitigations:\n"
    "   - Table with probability, impact, and mitigation actions\n\n"
    "6. Integrated recipe or scaling study:\n"
    "   - Explain how experiments combine and inform each other\n"
    "   - Mention planned sensitivity or Pareto analysis where appropriate\n\n"
    "Use tools when they yield clearly relevant sources. If retrieved evidence is generic or off-topic, note the limitation and propose how to gather better references instead of forcing a citation. "
    "When invoking web search, craft domain-specific queries that combine method + task keywords and, when appropriate, filters such as 'arXiv', 'state-of-the-art', or key dataset names to surface precise academic sources. "
    "When citing, prefer [file:page] for attachments and [n] for web results. If no high-confidence references exist, state that explicitly and describe how you would acquire authoritative evidence. "
    "Aim for expert-level depth without unnecessary repetition."
)


def apply_stage_directives(stage_letter: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    stage = (stage_letter or "").strip().upper()
    if stage == "C":
        return f"{_STAGE_C_DIRECTIVE}\n\n{prompt_text}", "stage_c_precision_balance_v1"
    return prompt_text, None
