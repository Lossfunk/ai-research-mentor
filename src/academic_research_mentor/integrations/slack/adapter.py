from __future__ import annotations

import os
import threading
import time
from typing import Any

from .session_store import GLOBAL_SESSION_STORE
from ...prompts_loader import load_instructions_from_prompt_md
from ...runtime import build_agent
from .context import build_thread_prelude


def _strip_wrapping_code_fence(text: str) -> str:
    try:
        s = (text or "").strip()
        if s.startswith("```") and s.endswith("```"):
            # Remove the first line fence and trailing fence
            lines = s.splitlines()
            if len(lines) >= 3:
                # Drop first and last line
                inner = "\n".join(lines[1:-1])
                return inner
        return text
    except Exception:
        return text


def _stream_llm_to_slack(agent: Any, effective_input: str, client, channel_id: str, message_ts: str, user_text: str) -> None:
    """Token-stream to Slack via chat.update with simple throttling. Falls back to single-shot if streaming not supported."""
    try:
        llm = getattr(agent, "_llm", None)
        build_messages = getattr(agent, "_build_messages", None)
        if llm is None or not hasattr(llm, "stream") or build_messages is None:
            # Fallback to single invoke
            reply = agent.run(effective_input)
            content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
            content = _strip_wrapping_code_fence(content)
            final_text = f"Q: {user_text}\n\n{content}"
            client.chat_update(channel=channel_id, ts=message_ts, text=final_text, mrkdwn=True)
            return

        accumulated: list[str] = []
        last_sent = 0.0
        try:
            from langchain_core.messages import AIMessageChunk  # type: ignore
        except Exception:  # pragma: no cover
            AIMessageChunk = None  # type: ignore
        for chunk in llm.stream(build_messages(effective_input)):
            piece = None
            if AIMessageChunk is not None and isinstance(chunk, AIMessageChunk):
                piece = getattr(chunk, "content", "") or ""
            else:
                # Some providers stream dict-like objects
                piece = getattr(chunk, "content", None)
                if not isinstance(piece, str):
                    piece = None
            if not piece:
                continue
            accumulated.append(piece)
            now = time.time()
            if (now - last_sent) >= 0.8:
                preview = _strip_wrapping_code_fence("".join(accumulated))
                body = f"Q: {user_text}\n\n{preview}"
                client.chat_update(channel=channel_id, ts=message_ts, text=body, mrkdwn=True)
                last_sent = now
        # Final flush
        content = _strip_wrapping_code_fence("".join(accumulated))
        final_text = f"Q: {user_text}\n\n{content}"
        client.chat_update(channel=channel_id, ts=message_ts, text=final_text, mrkdwn=True)
    except Exception:
        # Best-effort fallback
        try:
            reply = agent.run(effective_input)
            content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
            content = _strip_wrapping_code_fence(content)
            final_text = f"Q: {user_text}\n\n{content}"
            client.chat_update(channel=channel_id, ts=message_ts, text=final_text, mrkdwn=True)
        except Exception:
            pass


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
        # Build a concise prelude from the recent thread on first run per session
        prelude = ""
        if not sess.history:
            try:
                prelude = build_thread_prelude(client, channel_id, thread_ts or None)
            except Exception:
                prelude = ""
        effective_input = (prelude + "\n\n" + user_text).strip() if prelude else user_text
        _stream_llm_to_slack(sess.agent, effective_input, client, channel_id, message_ts, user_text)

    t = threading.Thread(target=_task, daemon=True)
    t.start()


