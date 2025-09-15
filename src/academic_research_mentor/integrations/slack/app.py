from __future__ import annotations

import os
from typing import Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from .adapter import run_agent_async


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def run_slack_bot() -> None:
    """Run Slack bot using Socket Mode.

    Env vars required:
    - SLACK_APP_TOKEN (xapp-...)
    - SLACK_BOT_TOKEN (xoxb-...)
    Optional:
    - SLACK_ALLOWED_TEAM_IDS (comma-separated)
    - SLACK_DEFAULT_THREAD_WINDOW
    """

    bot_token = _require_env("SLACK_BOT_TOKEN")
    app = App(token=bot_token)

    allowed_teams = {
        t.strip()
        for t in (os.environ.get("SLACK_ALLOWED_TEAM_IDS") or "").split(",")
        if t.strip()
    }

    def _team_allowed(team_id: str | None) -> bool:
        if not allowed_teams:
            return True
        return bool(team_id and team_id in allowed_teams)

    @app.command("/mentor")
    def handle_mentor_command(ack, body, respond, client, logger):  # type: ignore[no-redef]
        try:
            ack()
        except Exception:
            pass
        team_id = (body or {}).get("team_id")
        if not _team_allowed(team_id):
            respond(text="This workspace is not authorized for Mentor.")
            return
        text = ((body or {}).get("text") or "").strip() or "How can I help you today?"
        channel_id = (body or {}).get("channel_id")
        user_id = (body or {}).get("user_id")
        thread_ts = None
        try:
            res = client.chat_postMessage(channel=channel_id, text=f"Starting mentor… <@{user_id}>", reply_broadcast=False)
            thread_ts = res.get("ts")
        except SlackApiError as exc:  # pragma: no cover - Slack API errors at runtime
            err = None
            try:
                err = exc.response.get("error")  # type: ignore[attr-defined]
            except Exception:
                err = None
            if err == "not_in_channel":
                try:
                    client.conversations_join(channel=channel_id)
                    res = client.chat_postMessage(channel=channel_id, text=f"Starting mentor… <@{user_id}>", reply_broadcast=False)
                    thread_ts = res.get("ts")
                except SlackApiError as exc2:
                    logger.error(f"Failed to post initial message after join attempt: {exc2}")
                    respond(text=(
                        "I’m not a member of this channel. Please add me with /invite @Research Mentor, "
                        "or start a DM with me, then try again."
                    ))
                    return
            else:
                logger.error(f"Failed to post initial message: {exc}")
                respond(text="Sorry, I couldn’t post in this channel. Please try again or DM me.")
                return

        # Route to agent in background and update message when done
        try:
            client.chat_update(channel=channel_id, ts=thread_ts, text=f"Mentor is getting ready…\n> {text}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to update message: {exc}")
        run_agent_async(team_id or "T_UNKNOWN", channel_id, thread_ts or "", text, client, thread_ts or "")

    @app.event("app_mention")
    def handle_app_mention(body, say, logger):  # type: ignore[no-redef]
        team_id = (body or {}).get("team", {}).get("id") or (body or {}).get("team_id")
        if not _team_allowed(team_id):
            return
        event = (body or {}).get("event", {})
        text = (event.get("text") or "").strip()
        user = event.get("user")
        ts = event.get("thread_ts") or event.get("ts")
        chan = event.get("channel")
        try:
            res = say(thread_ts=ts, text=f"Mentor is thinking…")
        except Exception:
            return
        run_agent_async(team_id or "T_UNKNOWN", chan, ts, text, app.client, getattr(res, "ts", ts))

    @app.event("message")
    def handle_dm_messages(body, say, logger):  # type: ignore[no-redef]
        event = (body or {}).get("event", {})
        # Only react in IMs without @mention
        if event.get("channel_type") != "im":
            return
        text = (event.get("text") or "").strip()
        if not text:
            return
        ts = event.get("thread_ts") or event.get("ts")
        chan = event.get("channel")
        try:
            res = say(thread_ts=ts, text=f"Mentor is thinking…")
        except Exception:
            return
        team_id = (body or {}).get("team", {}).get("id") or (body or {}).get("team_id")
        run_agent_async(team_id or "T_UNKNOWN", chan, ts, text, app.client, getattr(res, "ts", ts))

    app_token = _require_env("SLACK_APP_TOKEN")
    handler = SocketModeHandler(app, app_token)
    handler.start()


