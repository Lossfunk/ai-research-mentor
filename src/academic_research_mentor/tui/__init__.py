from __future__ import annotations

import sys

from ..cli.session import load_env_file
from ..core.bootstrap import bootstrap_registry_if_enabled
from ..rich_formatter import RichFormatter, SilentRichFormatter, set_formatter
from ..runtime.context import prepare_agent
from ..runtime.events import subscribe_events


def _ensure_textual_available() -> None:
    try:
        __import__("textual")
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        if exc.name == "textual":
            message = (
                "Textual is not installed. Install the TUI extra with "
                "`pip install academic-research-mentor[tui]` and try again."
            )
            raise SystemExit(message) from exc
        raise


def main() -> None:
    _ensure_textual_available()

    from ..rich_formatter import print_info
    from .app import MentorTUI
    from .session import TUISessionManager

    load_env_file()

    discovered = bootstrap_registry_if_enabled()
    if discovered:
        print_info(f"Tool registry initialized: {', '.join(discovered)}")

    subscription = subscribe_events()
    set_formatter(SilentRichFormatter())

    try:
        prep = prepare_agent()
        session = None
        offline_reason = prep.offline_reason
        if prep.agent is not None:
            session = TUISessionManager(prep.agent, prep.loaded_variant)
            offline_reason = None

        app = MentorTUI(session=session, subscription=subscription, offline_reason=offline_reason)
        app.run()
    finally:
        subscription.close()
        set_formatter(RichFormatter())


if __name__ == "__main__":  # pragma: no cover
    main()
