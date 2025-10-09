from __future__ import annotations

import os
import signal
from ..core.bootstrap import bootstrap_registry_if_enabled
from ..rich_formatter import print_error
from ..runtime.context import prepare_agent

from .args import build_parser
from .commands import (
    verify_environment,
    show_env_help,
    list_tools_command,
    show_candidates_command,
    recommend_command,
    show_runs_command,
)
from .repl import online_repl
from .openrouter_setup import maybe_run_openrouter_setup
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

    maybe_run_openrouter_setup(force=getattr(args, "interactive_setup", False))

    # Attach PDFs if provided (do this BEFORE building the agent so tools can reflect attachment presence)
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

    ascii_override = True if getattr(args, "ascii", False) else None
    prep = prepare_agent(prompt_arg=getattr(args, "prompt", None), ascii_override=ascii_override)

    if prep.agent is None:
        print_error(prep.offline_reason or "Model initialization failed. Set OPENROUTER_API_KEY in your .env and re-run: uv run academic-research-mentor --check-env")
        return

    # REPL
    online_repl(prep.agent, prep.loaded_variant)
