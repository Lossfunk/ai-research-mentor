from __future__ import annotations

import os
import argparse
from typing import Optional, Tuple, Any

from dotenv import load_dotenv

from .prompts_loader import load_instructions_from_prompt_md
from .runtime import build_agent
from .router import route_and_maybe_run_tool
from .rich_formatter import print_formatted_response, print_info, print_error, get_formatter
from .core.bootstrap import bootstrap_registry_if_enabled
from .literature_review import build_research_context


def _load_env_file() -> None:
    """Load environment variables from .env file with error handling."""
    debug_env = os.environ.get("ARM_DEBUG_ENV", "").lower() in ("1", "true", "yes")

    try:
        # Try to find .env file in current directory first
        if os.path.exists(".env"):
            load_dotenv(".env", verbose=False, override=False)
            if debug_env:
                print(f"Debug: Loaded .env from current directory: {os.path.abspath('.env')}")
            return

        # Try to find .env file in parent directories (common when running from subdirs)
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            env_path = os.path.join(current_dir, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, verbose=False, override=False)
                if debug_env:
                    print(f"Debug: Loaded .env from: {env_path}")
                return
            current_dir = os.path.dirname(current_dir)

        # If no .env file found, that's okay - just continue with system env vars
        if debug_env:
            print("Debug: No .env file found, using system environment variables only")

    except Exception as e:
        # Don't fail the application if .env loading fails
        # Just print a warning and continue
        print(f"Warning: Failed to load .env file: {e}")


def _verify_environment() -> None:
    """Verify environment configuration and show status."""
    from .rich_formatter import print_info, print_error, get_formatter

    formatter = get_formatter()
    formatter.print_rule("Environment Configuration Status")

    # Check API keys
    api_keys = [
        ("OPENROUTER_API_KEY", "OpenRouter (recommended for O3 access)"),
        ("OPENAI_API_KEY", "OpenAI GPT models"),
        ("GOOGLE_API_KEY", "Google Gemini models"),
        ("ANTHROPIC_API_KEY", "Anthropic Claude models"),
        ("MISTRAL_API_KEY", "Mistral models"),
    ]

    configured_keys = []
    for key, description in api_keys:
        value = os.environ.get(key)
        if value:
            # Show only first 6 and last 4 chars for security
            masked = f"{value[:6]}...{value[-4:]}" if len(value) > 10 else "***"
            print_info(f"✓ {key}: {masked} ({description})")
            configured_keys.append(key)
        else:
            print_error(f"✗ {key}: Not configured ({description})")

    if not configured_keys:
        print_error("No API keys configured! Please set at least one API key in your .env file.")
        return

    formatter.console.print("")

    # Check model configurations
    model_configs = [
        ("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
        ("OPENAI_MODEL", "gpt-4o-mini"),
        ("GEMINI_MODEL", "gemini-2.5-flash-latest"),
        ("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        ("MISTRAL_MODEL", "mistral-large-latest"),
    ]

    print_info("Model Configuration:")
    for key, default in model_configs:
        value = os.environ.get(key, default)
        status = "custom" if os.environ.get(key) else "default"
        print_info(f"  {key}: {value} ({status})")

    formatter.console.print("")

    # Check agent configuration
    agent_mode = os.environ.get("LC_AGENT_MODE", "chat")
    prompt_variant = os.environ.get("ARM_PROMPT", os.environ.get("LC_PROMPT", "mentor"))
    ascii_mode = bool(os.environ.get("ARM_PROMPT_ASCII", os.environ.get("LC_PROMPT_ASCII")))

    print_info("Agent Configuration:")
    print_info(f"  Agent Mode: {agent_mode}")
    print_info(f"  Prompt Variant: {prompt_variant}")
    print_info(f"  ASCII Mode: {ascii_mode}")

    formatter.console.print("")


def _show_env_help() -> None:
    """Show help about environment variables and .env file usage."""
    from .rich_formatter import get_formatter, print_info

    formatter = get_formatter()
    formatter.print_rule("Environment Variables Help")

    formatter.console.print("""
[bold cyan]Using .env File:[/bold cyan]

The Academic Research Mentor automatically loads environment variables from a .env file.
Place your .env file in the project root directory or any parent directory.

[bold cyan]Required API Keys (at least one):[/bold cyan]

• [bold]OPENROUTER_API_KEY[/bold] - Recommended for O3-powered literature review
• [bold]OPENAI_API_KEY[/bold] - For OpenAI GPT models
• [bold]GOOGLE_API_KEY[/bold] - For Google Gemini models
• [bold]ANTHROPIC_API_KEY[/bold] - For Anthropic Claude models
• [bold]MISTRAL_API_KEY[/bold] - For Mistral models

[bold cyan]Optional Model Configuration:[/bold cyan]

• [bold]OPENROUTER_MODEL[/bold] (default: anthropic/claude-sonnet-4)
• [bold]OPENAI_MODEL[/bold] (default: gpt-4o-mini)
• [bold]GEMINI_MODEL[/bold] (default: gemini-2.5-flash-latest)
• [bold]ANTHROPIC_MODEL[/bold] (default: claude-3-5-sonnet-latest)
• [bold]MISTRAL_MODEL[/bold] (default: mistral-large-latest)

[bold cyan]Agent Configuration:[/bold cyan]

• [bold]LC_AGENT_MODE[/bold] - "chat" (default), "react", or "router"
• [bold]ARM_PROMPT[/bold] - "mentor" or "system" prompt variant
• [bold]ARM_PROMPT_ASCII[/bold] - Set to "1" for ASCII-friendly symbols

[bold cyan]Debug Options:[/bold cyan]

• [bold]ARM_DEBUG_ENV[/bold] - Set to "1" to show .env file loading debug info

[bold cyan]Example .env file:[/bold cyan]

```
# Primary API key (recommended)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Agent configuration
LC_AGENT_MODE=chat
ARM_PROMPT=mentor

# Optional: Custom models
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

Use --check-env to verify your current configuration.
""")


def main() -> None:
    """CLI entrypoint for Academic Research Mentor (thin wrapper)."""
    # Load environment variables from .env file
    _load_env_file()

    # Optional: initialize tool registry (WS2) behind a feature flag
    discovered = bootstrap_registry_if_enabled()
    if discovered:
        print_info(f"Tool registry initialized: {', '.join(discovered)}")

    parser = argparse.ArgumentParser(
        description="Academic Research Mentor - AI-powered research assistance with O3-powered literature review",
        epilog="Environment variables are automatically loaded from .env file. Use --env-help for configuration details."
    )
    parser.add_argument(
        "--prompt",
        choices=["mentor", "system"],
        default=None,
        help="Select prompt variant: 'mentor' for conversational guidance, 'system' for technical assistance (default: from ARM_PROMPT env var or 'mentor')"
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Normalize prompt symbols to ASCII-friendly characters for better terminal compatibility"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Verify and display current environment configuration including API keys and agent settings, then exit"
    )
    parser.add_argument(
        "--env-help",
        action="store_true",
        help="Show comprehensive help about environment variables, .env file usage, and configuration options, then exit"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Discover and list available tools (ignores FF_REGISTRY_ENABLED), then exit",
    )
    parser.add_argument(
        "--show-candidates",
        type=str,
        default=None,
        help="Show tool candidates for a goal (uses orchestrator selection), then exit",
    )

    try:
        args, _unknown = parser.parse_known_args()
    except SystemExit:
        class _Args:  # type: ignore
            prompt: Optional[str] = None
            ascii: bool = False
            check_env: bool = False
            env_help: bool = False

        args = _Args()

    # Handle environment check command
    if getattr(args, 'check_env', False):
        _verify_environment()
        return

    # Handle environment help command
    if getattr(args, 'env_help', False):
        _show_env_help()
        return

    # Handle tool listing (forces discovery)
    if getattr(args, 'list_tools', False):
        try:
            from .tools import auto_discover as _auto, list_tools as _list
            _auto()
            names = sorted(list(_list().keys()))
            if names:
                print_info(f"Discovered tools ({len(names)}): {', '.join(names)}")
            else:
                print_info("No tools discovered.")
        except Exception as e:  # noqa: BLE001
            print_error(f"Tool listing failed: {e}")
        return

    # Handle candidates display (forces discovery, runs selection-only)
    if getattr(args, 'show_candidates', None):
        goal = str(getattr(args, 'show_candidates'))
        try:
            from .tools import auto_discover as _auto
            from .core.orchestrator import Orchestrator
            _auto()
            orch = Orchestrator()
            out = orch.run_task("literature_search", context={"goal": goal})
            cands = out.get("candidates", [])
            if cands:
                pretty = ", ".join(f"{n}:{s}" for n, s in cands)
                print_info(f"Candidates for goal -> {goal}: {pretty}")
            else:
                print_info(f"No candidates for goal -> {goal}")
        except Exception as e:  # noqa: BLE001
            print_error(f"Show candidates failed: {e}")
        return

    # Prefer new env vars; keep backward-compatible AGNO_* fallback
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

    instructions, loaded_variant = load_instructions_from_prompt_md(
        prompt_variant, ascii_normalize
    )
    if not instructions:
        instructions = (
            "You are an expert research mentor. Ask high-impact questions first, then provide concise, actionable guidance."
        )
        loaded_variant = "fallback"

    # Small runtime prelude to enforce single-variant usage and tool discipline
    runtime_prelude = (
        "Use the selected core prompt variant only; never combine prompts. "
        "Default to conversational answers; call tools only when they would materially change advice."
    )
    effective_instructions = f"{runtime_prelude}\n\n{instructions}"

    # Debug: report guidelines status (enabled/disabled and basic stats)
    guidelines_injector = None
    try:
        from .guidelines_engine import create_guidelines_injector  # type: ignore
        injector = create_guidelines_injector()
        guidelines_injector = injector
        stats = injector.get_stats()
        cfg = stats.get("config", {}) if isinstance(stats, dict) else {}
        gs = stats.get("guidelines_stats", {}) if isinstance(stats, dict) else {}
        enabled = bool(cfg.get("is_enabled"))
        if enabled:
            total = gs.get("total_guidelines")
            categories = gs.get("categories")
            token_estimate = stats.get("token_estimate", 0)
            print_info(
                f"Guidelines: enabled (mode={cfg.get('mode')}, total={total}, categories={categories}, tokens≈{token_estimate})"
            )
        else:
            print_info("Guidelines: disabled")
    except Exception:
        # Silent: guidelines engine optional
        pass

    # Inject guidelines into the system instructions (if configured)
    try:
        if guidelines_injector is not None:
            effective_instructions = guidelines_injector.inject_guidelines(effective_instructions)  # type: ignore[attr-defined]
    except Exception:
        # Non-fatal: proceed without injected guidelines
        pass

    agent, offline_reason = build_agent(effective_instructions)

    if agent is None:
        _offline_repl(offline_reason or "Unknown reason")
        return

    # Check agent mode to determine routing behavior
    agent_mode = os.environ.get("LC_AGENT_MODE", "chat").strip().lower()
    use_manual_routing = agent_mode == "chat"

    # Use Rich formatting for the welcome message
    formatter = get_formatter()
    formatter.print_rule("Academic Research Mentor")
    print_info(f"Loaded prompt variant: {loaded_variant}")
    print_info(f"Agent mode: {agent_mode}")
    print_info("Type 'exit' to quit")
    formatter.console.print("")
    while True:
        try:
            # Use Rich console for input prompt
            formatter.console.print("[bold cyan]You:[/bold cyan] ", end="")
            user = input().strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        # Step 1: Build research context using O3-powered literature review
        research_context = build_research_context(user)

        # Step 2: If manual routing is enabled and this is a simple tool request, handle it
        if use_manual_routing and route_and_maybe_run_tool(user):
            continue

        # Step 3: Provide research context to the agent if available
        if research_context.get("has_research_context", False):
            # Enhance the agent's instructions with research context
            context_prompt = research_context.get("context_for_agent", "")
            enhanced_user_input = f"{context_prompt}\n\nUser Query: {user}"
        else:
            enhanced_user_input = user

        try:
            # Prefer streaming if available
            agent.print_response(enhanced_user_input, stream=True)  # type: ignore[attr-defined]
        except Exception:
            try:
                reply = agent.run(enhanced_user_input)
                content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
                print_formatted_response(content, "Mentor")
            except Exception as exc:  # noqa: BLE001
                print_error(f"Mentor response failed: {exc}")

        # Add spacing between conversation turns
        formatter.console.print("")


def _offline_repl(reason: str) -> None:
    formatter = get_formatter()
    formatter.print_rule("Academic Research Mentor (Offline Mode)")
    print_info("Type 'exit' to quit")
    if reason:
        print_error(f"Offline reason: {reason}")
    print_info("Falling back to a simple echo mentor")
    formatter.console.print("")

    while True:
        try:
            formatter.console.print("[bold cyan]You:[/bold cyan] ", end="")
            user = input().strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if route_and_maybe_run_tool(user):
            continue
        print_formatted_response(
            "A few quick questions to calibrate: What is your goal, compute budget, and target venue? Then I can suggest next steps.",
            "Mentor"
        )

        # Add spacing between conversation turns
        formatter.console.print("")
