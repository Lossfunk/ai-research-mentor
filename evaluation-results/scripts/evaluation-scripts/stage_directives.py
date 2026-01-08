"""Stage-specific prompt directives for evaluation agents."""

from __future__ import annotations

from typing import Optional, Tuple


_STAGE_C_DIRECTIVE = (
    "You are preparing a turnkey research execution plan with publication-grade rigor. "
    "Do not request additional context unless it is absolutely required to avoid catastrophic mistakes. "
    "Respond with the following sections in order:\n\n"
    "1. Problem framing and goals\n"
    "2. Experiments (each with hypothesis, setup, baselines, evaluation metrics, and expected outcomes)\n"
    "3. Timeline for the next 6 months with milestones\n"
    "4. Resources (compute, tools, datasets)\n"
    "5. Risks and mitigations table\n"
    "6. Stretch ideas or follow-up directions\n\n"
    "Use tools when they yield clearly relevant sources. If retrieved evidence is generic or off-topic, note the limitation and propose how to gather better references instead of forcing a citation. "
    "When invoking web search, craft domain-specific queries that combine method + task keywords and, when appropriate, filters such as 'arXiv', 'state-of-the-art', or key dataset names to surface precise academic sources. "
    "When citing, prefer [file:page] for attachments and [n] for web results. If no high-confidence references exist, state that explicitly and describe how you would acquire authoritative evidence. "
    "Finish with a single optional follow-up suggestion labelled 'Optional next step'."
)


def apply_stage_directives(stage_letter: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    stage = (stage_letter or "").strip().upper()
    if stage == "C":
        return f"{_STAGE_C_DIRECTIVE}\n\n{prompt_text}", "stage_c_natural_balance_v1"
    return prompt_text, None
