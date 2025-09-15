from __future__ import annotations

import os
import threading
from typing import Any

from .session_store import GLOBAL_SESSION_STORE
from ...prompts_loader import load_instructions_from_prompt_md
from ...runtime import build_agent


def _build_agent_for_slack() -> tuple[Any | None, str]:
    prompt_variant = (
        os.environ.get("ARM_PROMPT")
        or os.environ.get("LC_PROMPT")
        or os.environ.get("AGNO_PROMPT")
        or "mentor"
    ).strip().lower()
    ascii_normalize = bool(
        os.environ.get("ARM_PROMPT_ASCII")
        or os.environ.get("LC_PROMPT_ASCII")
        or os.environ.get("AGNO_PROMPT_ASCII")
    )
    instructions, _loaded = load_instructions_from_prompt_md(prompt_variant, ascii_normalize)
    if not instructions:
        instructions = (
            "You are an expert research mentor. Ask high-impact questions first, then provide concise, actionable guidance."
        )
    agent, _reason = build_agent(instructions)
    return agent, instructions


def run_agent_async(team_id: str, channel_id: str, thread_ts: str, user_text: str, client, message_ts: str) -> None:
    """Run the agent in a background thread, update Slack message as a simple non-streaming MVP."""

    def _task() -> None:
        sess = GLOBAL_SESSION_STORE.create_or_get(team_id, channel_id, thread_ts)
        if sess.agent is None:
            agent, _ = _build_agent_for_slack()
            sess.agent = agent
        if sess.agent is None:
            try:
                client.chat_update(channel=channel_id, ts=message_ts, text="Model initialization failed. Configure API keys and retry.")
            except Exception:
                pass
            return
        try:
            reply = sess.agent.run(user_text)
            content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
        except Exception as exc:  # noqa: BLE001
            content = f"Mentor failed to respond: {exc}"
        try:
            client.chat_update(channel=channel_id, ts=message_ts, text=content)
        except Exception:
            pass

    t = threading.Thread(target=_task, daemon=True)
    t.start()


