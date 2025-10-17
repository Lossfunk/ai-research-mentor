"""Stage-specific prompt directives for evaluation agents."""

from __future__ import annotations

from typing import Optional, Tuple


_STAGE_C_DIRECTIVE = (
    "You are preparing a turnkey research execution plan. "
    "Do not request additional context unless it is absolutely required to avoid catastrophic mistakes. "
    "Respond with the following sections in order: \n"
    "1. Problem framing and goals.\n"
    "2. Experiments (each with hypothesis, setup, baselines, evaluation metrics, and expected outcomes).\n"
    "3. Timeline for the next 6 months with monthly milestones.\n"
    "4. Resources (people, compute, tools, datasets).\n"
    "5. Risks and mitigations table.\n"
    "6. Stretch ideas or follow-up directions.\n"
    "Include only citations that come from tools (use [file:page] or [n] style). If no evidence is available, state that explicitly. "
    "Finish with a single optional follow-up suggestion labelled 'Optional next step'."
)


def apply_stage_directives(stage_letter: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    stage = (stage_letter or "").strip().upper()
    if stage == "C":
        return f"{_STAGE_C_DIRECTIVE}\n\n{prompt_text}", "stage_c_direct_plan_v1"
    return prompt_text, None
