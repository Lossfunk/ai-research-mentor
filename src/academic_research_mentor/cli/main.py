from __future__ import annotations

import os
import signal
from typing import Optional, Tuple, Any

from ..prompts_loader import load_instructions_from_prompt_md
from ..runtime import build_agent
from ..core.bootstrap import bootstrap_registry_if_enabled
from ..guidelines_engine import create_guidelines_injector  # type: ignore
from ..rich_formatter import print_error

from .args import build_parser
from .commands import (
    verify_environment,
    show_env_help,
    list_tools_command,
    show_candidates_command,
    recommend_command,
    show_runs_command,
)
from .repl import online_repl, offline_repl
from .session import load_env_file, signal_handler


def main() -> None:
    # Signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # .env
    load_env_file()

    # Feature-flagged registry
    discovered = bootstrap_registry_if_enabled()
    if discovered:
        from ..rich_formatter import print_info
        print_info(f"Tool registry initialized: {', '.join(discovered)}")

    # Args
    parser = build_parser()
    try:
        args, _unknown = parser.parse_known_args()
    except SystemExit:
        class _Args:  # type: ignore
            prompt: Optional[str] = None
            ascii: bool = False
            check_env: bool = False
            env_help: bool = False
            list_tools: bool = False
            show_candidates: Optional[str] = None
            recommend: Optional[str] = None
            show_runs: bool = False
        args = _Args()

    # Commands
    if getattr(args, 'check_env', False):
        verify_environment()
        return
    if getattr(args, 'env_help', False):
        show_env_help()
        return
    if getattr(args, 'list_tools', False):
        list_tools_command()
        return
    if getattr(args, 'show_candidates', None):
        show_candidates_command(str(getattr(args, 'show_candidates')))
        return
    if getattr(args, 'recommend', None):
        recommend_command(str(getattr(args, 'recommend')))
        return
    if getattr(args, 'show_runs', False):
        show_runs_command()
        return

    # Prompt selection
    prompt_variant = (
        args.prompt
        or os.environ.get("ARM_PROMPT")
        or os.environ.get("LC_PROMPT")
        or os.environ.get("AGNO_PROMPT")
        or "mentor"
    ).strip().lower()
    ascii_normalize = bool(
        args.ascii
        or os.environ.get("ARM_PROMPT_ASCII")
        or os.environ.get("LC_PROMPT_ASCII")
        or os.environ.get("AGNO_PROMPT_ASCII")
    )

    instructions, loaded_variant = load_instructions_from_prompt_md(prompt_variant, ascii_normalize)
    if not instructions:
        instructions = (
            "You are an expert research mentor. Ask high-impact questions first, then provide concise, actionable guidance."
        )
        loaded_variant = "fallback"

    runtime_prelude = (
        "Use the selected core prompt variant only; never combine prompts. "
        "Default to conversational answers; call tools only when they would materially change advice. "
        "When user-attached PDFs are present, FIRST use the attachments_search tool to ground your answer with [file:page] citations before any external search."
    )
    effective_instructions = f"{runtime_prelude}\n\n{instructions}"

    # Guidelines injection (optional)
    try:
        injector = create_guidelines_injector()
        stats = injector.get_stats()
        cfg = stats.get("config", {}) if isinstance(stats, dict) else {}
        gs = stats.get("guidelines_stats", {}) if isinstance(stats, dict) else {}
        enabled = bool(cfg.get("is_enabled"))
        if enabled:
            from ..rich_formatter import print_info
            total = gs.get("total_guidelines")
            categories = gs.get("categories")
            token_estimate = stats.get("token_estimate", 0)
            print_info(
                f"Guidelines: enabled (mode={cfg.get('mode')}, total={total}, categories={categories}, tokens≈{token_estimate})"
            )
            effective_instructions = injector.inject_guidelines(effective_instructions)  # type: ignore[attr-defined]
        else:
            from ..rich_formatter import print_info
            print_info("Guidelines: disabled")
    except Exception:
        pass

    # Agent
    agent, offline_reason = build_agent(effective_instructions)
    if agent is None:
        print_error(offline_reason or "Model initialization failed. Set one of the API keys in your .env (OPENROUTER_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, or MISTRAL_API_KEY). Then re-run: uv run academic-research-mentor --check-env")
        return

    # Attach PDFs if provided
    try:
        pdfs = getattr(args, 'attach_pdf', None)
        if pdfs:
            from ..attachments import attach_pdfs, get_summary
            attach_pdfs([str(p) for p in pdfs if p])
            from ..rich_formatter import print_info
            summ = get_summary()
            msg = (
                f"Attachments loaded: files={summ.get('files')}, pages={summ.get('pages')}, chunks={summ.get('chunks')} "
                f"(backend={summ.get('backend')})"
            )
            if (summ.get('skipped_large') or 0) > 0:
                msg += f" | skipped_large={summ.get('skipped_large')} (> {50} MB)"
            if (summ.get('truncated') or 0) > 0:
                msg += f" | truncated_files={summ.get('truncated')} (> 500 pages)"
            print_info(msg)
    except Exception:
        pass

    # REPL
    online_repl(agent, loaded_variant)
