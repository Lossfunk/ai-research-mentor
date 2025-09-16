from __future__ import annotations

from typing import Any, List, Dict, Optional


def _get_bot_user_id(client) -> Optional[str]:
    try:
        res = client.auth_test()
        return res.get("user_id")
    except Exception:
        return None


def fetch_thread_messages(client, channel_id: str, thread_ts: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent messages for a thread or channel history when no thread_ts.

    Returns messages sorted ascending by ts.
    """
    try:
        if thread_ts:
            res = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=limit)
            messages = res.get("messages", []) or []
        else:
            res = client.conversations_history(channel=channel_id, limit=limit)
            messages = res.get("messages", []) or []
    except Exception:
        messages = []
    # sort ascending by ts
    try:
        messages = sorted(messages, key=lambda m: float(m.get("ts", 0)))
    except Exception:
        pass
    return messages


def _truncate(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def build_concise_prelude(messages: List[Dict[str, Any]], *, bot_user_id: Optional[str], window: int = 15, max_chars: int = 1200) -> str:
    """Create a short textual prelude summarizing the last N messages in the thread.

    Format keeps roles as User/Mentor and trims to fit max_chars.
    """
    role_lines: List[str] = []
    # Consider only the last `window` messages
    tail = messages[-window:] if len(messages) > window else messages
    for m in tail:
        user = m.get("user") or m.get("bot_id") or "user"
        text = (m.get("text") or "").strip()
        # Skip empty and Slack system/service messages
        if not text or m.get("subtype") in {"channel_join", "channel_topic", "message_changed"}:
            continue
        role = "Mentor" if bot_user_id and user == bot_user_id else "User"
        # Avoid including our own placeholder/status lines
        if role == "Mentor" and ("Mentor is thinking…" in text or "Starting mentor…" in text):
            continue
        role_lines.append(f"- {role}: {text}")

    header = "Context from recent thread messages:"
    prelude = header + "\n" + "\n".join(role_lines)
    return _truncate(prelude, max_chars)


def build_thread_prelude(client, channel_id: str, thread_ts: Optional[str], *, window: int = 15, max_chars: int = 1200) -> str:
    """Fetch thread or channel history and build a concise prelude string."""
    messages = fetch_thread_messages(client, channel_id, thread_ts, limit=max(window * 3, 50))
    bot_user_id = _get_bot_user_id(client)
    return build_concise_prelude(messages, bot_user_id=bot_user_id, window=window, max_chars=max_chars)


