from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..prompts_loader import load_instructions_from_prompt_md
from ..rich_formatter import print_error, print_info
from ..guidelines_engine import create_guidelines_injector  # type: ignore
from .builder import build_agent


@dataclass
class AgentPreparationResult:
    agent: Optional[Any]
    offline_reason: Optional[str]
    loaded_variant: str
    effective_instructions: str


def _compose_runtime_prelude() -> str:
    prelude = (
        "Use the selected core prompt variant only; never combine prompts. "
        "Default to conversational answers; call tools only when they would materially change advice. "
        "When user-attached PDFs are present, FIRST use attachments_search to ground your answer with [file:page] citations. "
        "For research queries about papers, literature, or getting started in a field: PREFER unified_research tool which combines papers and guidelines with [P#] and [G#] citations. "
        "For mentorship, hypothesis-generation, getting-started, novelty, experiments, methodology: AFTER grounding, call mentorship_guidelines (research_guidelines) BEFORE any literature_search; "
        "then, if helpful, run literature_search. In your final answer include (1) at least three concrete, falsifiable experiments and (2) one to two literature anchors (titles with links). "
        "Always keep claims grounded in attached snippets with [file:page] citations. "
        "IMPORTANT: Your advice must avoid hyperbole, and claims must be substantiated by evidence presented. "
        "Science is evidence-based; never present unsubstantiated claims. If a claim is speculative, pose it as conjecture, not a conclusion."
    )
    prelude += (
        " Always include citations to sources when giving research advice. "
        "When using unified_research tool: embed inline bracketed citations [P#] for papers and [G#] for guidelines right after the specific sentences they support. "
        "When using other tools: embed inline bracketed citations [n] right after the specific sentences they support, where [n] refers to the numbered source from the tool output. "
        "Soft guidance: Prefer citing relevant papers [P#] when available for research recommendations. If no relevant papers exist, use guidelines [G#] for methodology advice. "
        "Also include a final 'Citations' section listing [ID] Title — URL."
    )
    return prelude


def prepare_agent(prompt_arg: Optional[str] = None, ascii_override: Optional[bool] = None) -> AgentPreparationResult:
    prompt_variant = (
        prompt_arg
        or os.environ.get("ARM_PROMPT")
        or os.environ.get("LC_PROMPT")
        or os.environ.get("AGNO_PROMPT")
        or "mentor"
    ).strip().lower()

    if ascii_override is None:
        ascii_normalize = bool(
            os.environ.get("ARM_PROMPT_ASCII")
            or os.environ.get("LC_PROMPT_ASCII")
            or os.environ.get("AGNO_PROMPT_ASCII")
        )
    else:
        ascii_normalize = bool(ascii_override)

    instructions, loaded_variant = load_instructions_from_prompt_md(prompt_variant, ascii_normalize)
    if not instructions:
        instructions = (
            "You are an expert research mentor. Ask high-impact questions first, then provide concise, actionable guidance."
        )
        loaded_variant = "fallback"

    effective_instructions = f"{_compose_runtime_prelude()}\n\n{instructions}"

    try:
        injector = create_guidelines_injector()
        stats = injector.get_stats()
        cfg = stats.get("config", {}) if isinstance(stats, dict) else {}
        gs = stats.get("guidelines_stats", {}) if isinstance(stats, dict) else {}
        enabled = bool(cfg.get("is_enabled"))
        if enabled:
            total = gs.get("total_guidelines")
            token_estimate = stats.get("token_estimate", 0)
            print_info(
                f"Guidelines: enabled (mode={cfg.get('mode')}, total={total}, tokens≈{token_estimate})"
            )
            effective_instructions = injector.inject_guidelines(effective_instructions)  # type: ignore[attr-defined]
        else:
            print_info("Guidelines: disabled")
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        print_error(f"Guidelines injector error: {exc}")

    agent, offline_reason = build_agent(effective_instructions)
    return AgentPreparationResult(agent, offline_reason, loaded_variant, effective_instructions)


__all__ = ["AgentPreparationResult", "prepare_agent"]
