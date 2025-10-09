from __future__ import annotations

import os
from ..rich_formatter import print_info, print_error, get_formatter


def verify_environment() -> None:
    formatter = get_formatter()
    formatter.print_rule("Environment Configuration Status")

    api_keys = [
        ("OPENROUTER_API_KEY", "OpenRouter (required)"),
    ]

    configured_keys = []
    for key, description in api_keys:
        value = os.environ.get(key)
        if value:
            masked = f"{value[:6]}...{value[-4:]}" if len(value) > 10 else "***"
            print_info(f"✓ {key}: {masked} ({description})")
            configured_keys.append(key)
        else:
            print_error(f"✗ {key}: Not configured ({description})")

    if not configured_keys:
        print_error("OpenRouter API key missing. Set OPENROUTER_API_KEY in your .env file.")
        return

    formatter.console.print("")

    model_configs = [
        ("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
    ]

    print_info("Model Configuration:")
    for key, default in model_configs:
        value = os.environ.get(key, default)
        status = "custom" if os.environ.get(key) else "default"
        print_info(f"  {key}: {value} ({status})")

    formatter.console.print("")

    prompt_variant = os.environ.get("ARM_PROMPT", os.environ.get("LC_PROMPT", "mentor"))
    ascii_mode = bool(os.environ.get("ARM_PROMPT_ASCII", os.environ.get("LC_PROMPT_ASCII")))

    print_info("Agent Configuration:")
    print_info(f"  Prompt Variant: {prompt_variant}")
    print_info(f"  ASCII Mode: {ascii_mode}")

    formatter.console.print("")


def show_env_help() -> None:
    formatter = get_formatter()
    formatter.print_rule("Environment Variables Help")

    formatter.console.print(
        """
[bold cyan]Using .env File:[/bold cyan]

The Academic Research Mentor automatically loads environment variables from a .env file.
Place your .env file in the project root directory or any parent directory.

[bold cyan]Required API Keys:[/bold cyan]

• [bold]OPENROUTER_API_KEY[/bold] - Required for OpenRouter-based mentoring

[bold cyan]Optional Model Configuration:[/bold cyan]

• [bold]OPENROUTER_MODEL[/bold] (default: anthropic/claude-sonnet-4)

[bold cyan]Agent Configuration:[/bold cyan]

• [bold]ARM_PROMPT[/bold] - "mentor" or "system" prompt variant
• [bold]ARM_PROMPT_ASCII[/bold] - Set to "1" for ASCII-friendly symbols

[bold cyan]Debug Options:[/bold cyan]

• [bold]ARM_DEBUG_ENV[/bold] - Set to "1" to show .env file loading debug info

[bold cyan]Example .env file:[/bold cyan]

```
# Primary API key (required)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Agent configuration
ARM_PROMPT=system

# Optional: Custom models
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

Use --check-env to verify your current configuration.
"""
    )


def list_tools_command() -> None:
    try:
        from ..tools import auto_discover as _auto, list_tools as _list
        _auto()
        names = sorted(list(_list().keys()))
        if names:
            print_info(f"Discovered tools ({len(names)}): {', '.join(names)}")
        else:
            print_info("No tools discovered.")
    except Exception as e:  # noqa: BLE001
        print_error(f"Tool listing failed: {e}")


def show_candidates_command(goal: str) -> None:
    try:
        from ..tools import auto_discover as _auto
        from ..core.orchestrator import Orchestrator
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


def recommend_command(goal: str) -> None:
    try:
        from ..tools import auto_discover as _auto, list_tools as _list
        from ..core.recommendation import score_tools
        _auto()
        scored = score_tools(goal, _list())
        if scored:
            top = scored[0]
            print_info(f"Top tool: {top[0]} (score={top[1]:.2f}, why={top[2]})")
            others = ", ".join(f"{n}:{s:.2f}" for n, s, _r in scored[1:])
            if others:
                print_info(f"Others: {others}")
        else:
            print_info(f"No suitable tools for goal -> {goal}")
    except Exception as e:  # noqa: BLE001
        print_error(f"Recommend failed: {e}")


def show_runs_command() -> None:
    try:
        from ..core.transparency import get_transparency_store
        store = get_transparency_store()
        runs = store.list_runs()[:10]
        if not runs:
            print_info("No tool runs recorded in this session.")
        else:
            for r in runs:
                print_info(f"run={r.run_id} tool={r.tool_name} status={r.status} events={len(r.events)}")
    except Exception as e:  # noqa: BLE001
        print_error(f"Show runs failed: {e}")
