# Slack Integration Plan for Research Mentor Agent

This document describes how to integrate the **Research Mentor Agent** into Slack. It extends the existing architecture (sessions, history, attachments, CLI entrypoints) and adds improvements for concurrency, persistence, attachments, and reliability.

---

## Goals and UX

* **Slack-first workflow**

  * Slash command `/mentor …` in any channel starts or continues a thread with the mentor.
  * App mentions `@Mentor …` inside a thread continue the session.
  * DMs with the bot are private threaded chats.
* **Minimal ceremony**

  * Bot always replies in a thread.
  * Streaming-like updates with throttled `chat.update`.
  * Session persists per Slack thread (or per DM channel).
  * Optional commands: `/mentor reset`, `/mentor resume`, `/mentor help`.
* **Future phases**

  * Phase 2: Ingest PDFs posted to thread; reference with `[file:page]`.
  * Phase 3: Admin/health commands, telemetry.

---

## Architecture Overview

### Slack Adapter

* Use **Bolt for Python** with **Socket Mode** (no public URL required).
* Directory: `src/academic_research_mentor/integrations/slack/`.

### Event Flow

* `/mentor`: ack within 3s → post initial placeholder → create or find session → route to agent.
* App mention or DM: route to existing session; create if none exists.
* Session key:

  ```
  f"{team_id}:{channel_id}:{thread_ts_or_dm}"
  ```

  * For threads: `thread_ts`.
  * For DMs: `im_channel_id`.

### Session Store

* Interface with pluggable backends:

  * In-memory LRU for dev/testing.
  * Redis/Dynamo later for production.
* Holds:

  * Per-session agent wrapper.
  * Bounded recent message history (last N turns).
  * Attachment store ID.
  * Metadata (last active, user IDs).

### Agent Lifecycle

* Built via `runtime.build_agent(...)` with same flags/guidelines.
* On session creation:

  * Pre-seed with concise context (recent Slack messages).
* On resume:

  * Rehydrate by replaying last N turns or loading checkpointed history.

### Context Extractor

* Fetch last N Slack messages (default 15).
* Build a concise prelude injected once at session start.

### Streaming

* Post placeholder → update the same message incrementally.
* Throttle updates to \~1/sec.
* On 429 errors, respect `Retry-After` and back off.
* Always ensure a final update with full content.

---

## Attachments (Phase 2)

* **Slack file ingestion**

  * Scopes: `files:read`.
  * Download files (PDFs only, size/page capped).
  * Quarantine + optional malware scan.
  * OCR or PDF text extraction.
* **Workerization**

  * Run ingestion in background queue (Celery/RQ).
  * Do not block event handler.
* **Per-session AttachmentStore**

  * Store extracted text/index.
  * Provide search and summary APIs.
  * Inject into agent tools.

---

## CLI Integration

* New subcommand:

  ```bash
  uv run academic-research-mentor slack
  ```
* Flags consistent with existing CLI (guidelines, registry, agent mode).
* Env vars:

  * `SLACK_APP_TOKEN` (xapp-…)
  * `SLACK_BOT_TOKEN` (xoxb-…)
  * `SLACK_ALLOWED_TEAM_IDS`
  * `SLACK_DEFAULT_THREAD_WINDOW=15`

---

## Session and Logging Model

* **Session key**: `team_id:channel_id:thread_ts_or_dm`.

* **History**: keep last N turns in store, feed into agent on init/resume.

* **Logs**:

  * Path: `convo-logs/slack/{team_id}/{channel_id}/{thread_ts}.json`
  * Each turn:

    ```json
    {
      "turn": 3,
      "user_prompt": "...",
      "ai_response": "...",
      "tool_calls": [],
      "slack_meta": {"user":"U123","ts":"1699999999.000200"}
    }
    ```

* **Retention**:

  * Decide expiry (e.g., 30 days).
  * Auto-prune idle sessions >X days.

---

## Error Handling and Reliability

* Ack slash commands immediately; process async.
* Rate limit protection:

  * Retry/backoff on Slack 429s.
  * Throttle streaming updates.
* Session concurrency:

  * Ensure atomic `get/create` in store.
  * Use per-session mutex if needed.
* Clear/reset session via `/mentor reset`.
* Resume session with `/mentor resume`.

---

## Security and Permissions

* Slack app scopes:

  * Core: `app_mentions:read`, `chat:write`, `commands`, `channels:history`, `im:history`.
  * Attachments: `files:read`.
* Validate `team_id` against allowlist.
* Mask PII in logs if required.
* Store tokens securely (never log).

---

## Testing

### Unit

* SessionStore lifecycle & TTL.
* Slash command handler logic.
* Streaming updater (rate limiting, retries).
* ChatLogger path & metadata.

### Integration

* End-to-end: slash command → session → agent → streaming reply.
* Race condition simulation (two simultaneous messages).
* Attachments: ingest large, empty, malformed, and OCR PDFs.

---

## Deployment

* **Dev**: Socket Mode, single worker.
* **Prod**:

  * Socket Mode or Events API behind HTTPS.
  * Optional health endpoint (`--health-port`).
* Commands:

  ```bash
  FF_REGISTRY_ENABLED=1 \
  FF_AGENT_RECOMMENDATION=1 \
  SLACK_APP_TOKEN=... \
  SLACK_BOT_TOKEN=... \
  uv run academic-research-mentor slack
  ```

---

## Phased Delivery

* **V1 (Core)**

  * Slash command, mentions/DMs.
  * Per-thread sessions.
  * Streaming updates.
  * Logging + resume/reset.
  * CLI entry.

* **V2 (Attachments)**

  * Per-session AttachmentStore.
  * Slack file ingestion.
  * Search + citation in responses.

* **V3 (Admin/Health)**

  * `/mentor status`, `/mentor help`.
  * Simple telemetry/metrics.

---

## TODOs

* [ ] Scaffold Slack Bolt app at `src/.../integrations/slack`.
* [ ] Add CLI entry to start Slack bot.
* [ ] Implement SessionStore interface (in-memory first).
* [ ] Wire session lifecycle into slash/mention handlers.
* [ ] Implement response streaming with throttled updates.
* [ ] Persist chat logs per thread.
* [ ] Add `/mentor reset` and `/mentor resume`.
* [ ] Document setup (`docs/slack.md`).
* [ ] Unit tests for handlers and session store.
* [ ] Phase 2: build AttachmentStore + Slack file ingestion.
* [ ] Phase 3: add admin/health commands.
* [ ] Deployment scripts + runbook.

---

## Priority Order

1. Scaffold Bolt app + CLI.
2. Implement SessionStore (in-memory).
3. Wire slash/mention handlers.
4. Add streaming updater.
5. Logging + resume/reset.
6. Attachments (Phase 2).
7. Admin/health (Phase 3).

---
