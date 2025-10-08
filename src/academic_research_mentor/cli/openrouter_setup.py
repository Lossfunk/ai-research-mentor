from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path
from typing import Callable, Iterable, Tuple

from ..rich_formatter import print_error, print_info, print_success


_MODEL_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("anthropic/claude-sonnet-4.5", "Anthropic Claude Sonnet 4.5"),
    ("openai/gpt-5", "OpenAI GPT-5"),
    ("moonshotai/kimi-k2-0905", "MoonshotAI Kimi K2 0905"),
)


def maybe_run_openrouter_setup(
    force: bool = False,
    input_fn: Callable[[str], str] | None = None,
    getpass_fn: Callable[[str], str] | None = None,
) -> bool:
    input_fn = input_fn or input
    getpass_fn = getpass_fn or getpass

    if not force:
        if os.environ.get("ARM_SKIP_INTERACTIVE_SETUP", "").lower() in {"1", "true", "yes"}:
            return False
        if os.environ.get("OPENROUTER_API_KEY"):
            return False
        if not _is_interactive_terminal():
            return False
        if _has_alternative_provider_configured():
            return False
        if not _prompt_yes_no("No OpenRouter API key detected. Configure it now? [Y/n]: ", True, input_fn):
            print_info("Skipping OpenRouter setup. You can rerun with --interactive-setup to configure later.")
            return False

    api_key = _prompt_api_key(getpass_fn, input_fn)
    if not api_key:
        print_info("OpenRouter setup cancelled. Agent will start without OpenRouter configuration.")
        return False

    model = _prompt_model_choice(input_fn)
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_MODEL"] = model
    print_success(f"Using OpenRouter model: {model}")

    if _prompt_yes_no("Persist these settings for future runs? [y/N]: ", False, input_fn):
        path = _persist_credentials(api_key, model)
        print_success(f"Saved OpenRouter credentials to {path}")
    else:
        print_info("Configuration applied only for this session.")

    return True


def _prompt_api_key(getpass_fn: Callable[[str], str], input_fn: Callable[[str], str]) -> str:
    while True:
        api_key = getpass_fn("Enter OpenRouter API key: ").strip()
        if api_key:
            return api_key
        if _prompt_yes_no("No key entered. Cancel setup? [Y/n]: ", True, input_fn):
            return ""


def _prompt_model_choice(input_fn: Callable[[str], str]) -> str:
    print_info("Select the default OpenRouter model:")
    for idx, (slug, label) in enumerate(_MODEL_CHOICES, start=1):
        print_info(f"  {idx}) {label} [{slug}]")

    while True:
        choice = input_fn("Enter choice number or model id (default 1): ").strip()
        if not choice:
            return _MODEL_CHOICES[0][0]
        for idx, (slug, _label) in enumerate(_MODEL_CHOICES, start=1):
            if choice == str(idx) or choice.lower() == slug.lower():
                return slug
        print_error("Invalid selection. Choose one of the listed options.")


def _prompt_yes_no(prompt: str, default: bool, input_fn: Callable[[str], str]) -> bool:
    suffix = "Y/n" if default else "y/N"
    prompt = prompt if prompt.strip().endswith(('?', ':')) else f"{prompt.rstrip()} "
    while True:
        raw = input_fn(prompt if prompt else f"[{suffix}] ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print_error("Please answer with 'y' or 'n'.")


def _persist_credentials(api_key: str, model: str) -> Path:
    path = _get_config_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: Iterable[str]
    if path.exists():
        existing_lines = path.read_text().splitlines()
    else:
        existing_lines = []

    lines = []
    found_key = False
    found_model = False
    for line in existing_lines:
        if line.startswith("OPENROUTER_API_KEY="):
            lines.append(f"OPENROUTER_API_KEY={api_key}")
            found_key = True
        elif line.startswith("OPENROUTER_MODEL="):
            lines.append(f"OPENROUTER_MODEL={model}")
            found_model = True
        else:
            lines.append(line)
    if not found_key:
        lines.append(f"OPENROUTER_API_KEY={api_key}")
    if not found_model:
        lines.append(f"OPENROUTER_MODEL={model}")

    path.write_text("\n".join(lines) + "\n")
    if os.name == "posix":
        os.chmod(path, 0o600)
    return path


def _get_config_env_path() -> Path:
    override = os.environ.get("ARM_CONFIG_HOME")
    if override:
        base = Path(override).expanduser()
    else:
        base = Path.cwd() / ".config"
    return base / "academic-research-mentor" / ".env"


def _is_interactive_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _has_alternative_provider_configured() -> bool:
    other_keys = (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
    )
    return any(os.environ.get(name) for name in other_keys)


__all__ = ["maybe_run_openrouter_setup"]
