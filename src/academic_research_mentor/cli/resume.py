from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Optional

from ..rich_formatter import print_info, get_formatter


def _load_turns_from_path(path: Optional[str]) -> tuple[list[dict], Optional[Path]]:
    p: Optional[Path]
    if path:
        p = Path(path)
        if not p.exists():
            # Allow bare filename under default log dir
            p = Path("convo-logs") / path
            if not p.exists():
                return [], None
    else:
        log_dir = Path("convo-logs")
        if not log_dir.exists():
            return [], None
        files = sorted(log_dir.glob("chat_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        p = files[0] if files else None
        if not p:
            return [], None

    try:
        with open(p, "r", encoding="utf-8") as f:
            turns = json.load(f)
        # Filter to real turns
        filtered = [
            t for t in turns
            if isinstance(t, dict)
            and t.get("ai_response")
            and t.get("user_prompt")
            and t.get("user_prompt", "").lower() not in {"exit", "quit", "eof (ctrl+d)", "unexpected_exit"}
        ]
        return filtered, p
    except Exception:
        return [], p


def _select_log_interactively(limit: int = 10) -> Optional[Path]:
    log_dir = Path("convo-logs")
    if not log_dir.exists():
        return None
    files = sorted(log_dir.glob("chat_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    files = files[:limit]
    fmt = get_formatter()
    fmt.print_rule("Resume: Select a conversation log")
    for idx, fp in enumerate(files, start=1):
        try:
            ts = fp.stat().st_mtime
            import datetime as _dt
            when = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            when = "unknown"
        fmt.console.print(f"[bold]{idx}[/bold]. {fp.name}  ([blue]{when}[/blue])")
    fmt.console.print("Enter a number to resume, or press Enter to cancel:", end=" ")
    try:
        choice = input().strip()
    except EOFError:
        return None
    if not choice:
        return None
    try:
        index = int(choice)
        if 1 <= index <= len(files):
            return files[index - 1]
    except Exception:
        pass
    print_info("Invalid selection; resume cancelled.")
    return None


def handle_resume_command(agent: Any, user_input: str) -> None:
    """Handle a /resume command in the REPL.

    Usage: /resume [filename-or-path]
    """
    try:
        parts = user_input.split(maxsplit=1)
        path = parts[1] if len(parts) > 1 else ""
        if path:
            turns, p = _load_turns_from_path(path)
        else:
            p = _select_log_interactively()
            turns = []
            if p:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        turns = json.load(f)
                    turns = [
                        t for t in turns
                        if isinstance(t, dict)
                        and t.get("ai_response")
                        and t.get("user_prompt")
                        and t.get("user_prompt", "").lower() not in {"exit", "quit", "eof (ctrl+d)", "unexpected_exit"}
                    ]
                except Exception:
                    turns = []
        if not p:
            print_info("No conversation logs found to resume.")
            return
        if hasattr(agent, 'preload_history_from_chatlog'):
            loaded = agent.preload_history_from_chatlog(turns)  # type: ignore[attr-defined]
            print_info(f"Loaded {loaded} prior turns from: {p}")
        else:
            print_info("This agent does not support resuming history in this mode.")
    except Exception as exc:
        print_info(f"Failed to resume: {exc}")


