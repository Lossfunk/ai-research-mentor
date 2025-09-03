# TODO - System Architecture Restructuring Implementation Plan

### Executive objectives
- **Autonomy**: Agent picks tools dynamically via metadata and recommendations.
- **Transparency**: Users and developers see search results and intermediate steps.
- **Consolidation**: Deprecate redundant search tools; unify on O3.
- **Organization**: Standardize tool shape, discovery, and structure for maintainability.
- **Quality**: Strong testing, observability, and performance guardrails.

## Scope and non-goals
- **In scope**: Tool interface/metadata standardization, tool registry & discovery, agent recommendation engine, O3 transparency (UI/streaming/logging/export), deprecations (arxiv/openreview), file structure reorg, tests, docs, dashboards, feature flags, rollout.
- **Out of scope (for now)**: ML-based learning to rank beyond a rules+heuristics baseline, cross-org permissions/integrations, paid compliance (e.g., HIPAA/GDPR DPIA) beyond basic PII safety.

## Assumptions
- Codebase is Python-first (due to `main.py`/`core/*.py` paths).
- Existing O3 integration already returns raw results internally (we’ll expose/stream/log).
- Frontend UI exists under `ui/` and can render new components.
- We can introduce a lightweight event layer (in-process pub/sub or WebSocket/SSE) without infra changes.

## Workstreams
- **WS1: Architecture & Filesystem**
- **WS2: Tool Interface, Metadata, Registry**
- **WS3: Agent Selection & Orchestrator**
- **WS4: Transparency & Storage & Streaming**
- **WS5: UI & Developer Dashboard**
- **WS6: Tool Consolidation & Deprecations**
- **WS7: Testing & Performance**
- **WS8: Documentation & Change Management**
- **WS9: Rollout, Flags, Observability, Security**

## Timeline and phases
- **Phase 1 (Weeks 1–2): Foundation**
  - WS1, WS2 core deliverables.
- **Phase 2 (Weeks 2–3): Tool Migration**
  - WS6 deprecations, partial WS4 (O3 transparency plumbing).
- **Phase 3 (Weeks 3–4): Agent Intelligence**
  - WS3 recommendation engine & fallback.
- **Phase 4 (Weeks 4–5): Transparency Features**
  - WS4, WS5 full UI/streaming/export/dashboard.
- **Phase 5 (Weeks 5–6): Testing & Docs**
  - WS7, WS8 hardening and performance.

## Detailed plan by workstream

### WS1: Architecture & Filesystem
- **Deliverables**
  - Create `tools/` with `__init__.py`, `base_tool.py`, `utils/`.
  - Add `core/agent.py`, `core/orchestrator.py`, `core/transparency.py`.
  - Create `main.py` entrypoint compatible with orchestrator.
  - Update `tests/`, `docs/`, `ui/` directories as needed.
- **Design decisions**
  - Tools are packages under `tools/<tool_name>/` with `tool.py` and optional `config.py`.
  - All imports reference new module paths.
- **Acceptance criteria**
  - Project imports succeed; CI builds pass.
  - All legacy paths removed or shimmed.
  - ADR recorded for new structure (ADR-001).
- **Dependencies**
  - None; unblock WS2 and WS3.

### WS2: Tool Interface, Metadata, Registry
- **Tool interface (in `base_tool.py`)**
  - Required methods: initialize(config), can_handle(task_context), execute(task_context, inputs), cleanup().
  - Result object: status (success/failure), outputs (structured), artifacts (URIs), metrics, errors.
- **Metadata (exposed by each tool and used by registry)**
  - Identity: name, version, owner.
  - Capabilities: task_types, domains, input_schema, output_schema.
  - Operational: cost_estimate, latency_profile, rate_limits.
  - Quality: reliability_score (historical), confidence_estimation flag.
  - Usage guidance: ideal_inputs, anti_patterns, prerequisites.
- **Registry & discovery (`tools/__init__.py`)**
  - Auto-discovery by scanning `tools/*/tool.py` for a `Tool` class implementing the base interface and exposing a `metadata` property.
  - Validation step: metadata schema validation; interface conformance check.
  - Registration: index of tool name → instance + metadata.
  - Health check on startup; lazy init supported.
- **Acceptance criteria**
  - Adding a new tool under `tools/<name>/` is auto-registered with zero orchestrator changes.
  - Invalid tools fail validation with actionable errors.
  - ADR for metadata schema and lifecycle (ADR-002).

### WS3: Agent Selection & Orchestrator
- **Recommendation engine**
  - Inputs: task_context (goal, constraints), recent outcomes, tool metadata, historical success stats.
  - Scoring function (MVP): weighted sum of capability match, latency, reliability, cost; tie-break with exploration probability.
  - Policy: top-1 execute with backoff; optionally top-k with parallel speculative exec for low-cost tasks behind a flag.
- **Fallback logic**
  - If primary fails or confidence low: choose next best; cap retries; circuit breaker for flapping tools.
- **Orchestrator responsibilities (`core/orchestrator.py`)**
  - Resolve tools via registry; call agent recommend; orchestrate execution; capture events to transparency layer; manage cancellations/timeouts.
- **Acceptance criteria**
  - Agent can run a task with no hardcoded tool references.
  - Fallback triggers and logs decisions.
  - Unit tests cover scoring and fallback paths.
  - ADR for selection strategy & fallbacks (ADR-003).

### WS4: Transparency, Storage, Streaming
- **Storage model (tool runs and artifacts)**
  - Entities:
    - ToolRun: id, task_id, tool_name, inputs_hash, start/end timestamps, status, scores, errors, metadata_snapshot.
    - Artifact: id, tool_run_id, type (text/json/file), uri or inline payload, size, checksum.
    - Event: id, tool_run_id, event_type (started/partial_result/final_result/error), timestamp, payload_summary.
  - Backends: pluggable; start with filesystem + sqlite/postgres via existing infra; abstract via `core/transparency.py`.
- **Streaming**
  - Event bus abstraction with in-process pub/sub.
  - Server push via WebSocket or SSE from backend to UI.
  - Backpressure: sample or coalesce partials for large streams.
- **Result exposure**
  - O3 search: surface top-N results with provenance; intermediate traces (queries, retries, rate-limit backs).
  - Export formats: JSON and CSV; include run metadata for audit.
- **Acceptance criteria**
  - Users can see O3 results and intermediate steps in UI.
  - Developers can query past ToolRun entries.
  - Real-time updates appear in the UI within latency budgets (<300ms server-to-client).
  - ADR for event model & streaming (ADR-004).

### WS5: UI & Developer Dashboard
- **User-facing components (in `ui/`)**
  - Search Results Panel: list + detail with source, snippet, score, timestamp.
  - Execution Timeline: chronological events (started, retries, partials, final).
  - Live Stream Indicator: status, errors, reconnect handling.
  - Export button (JSON/CSV) with confirmation and size warning.
- **Developer dashboard**
  - Tool Runs Explorer: filters by tool, status, time, duration, error codes.
  - Tool Health: latency histograms, error rates, success ratios per tool version.
  - Selection Decisions: sampled logs of agent scores and rationale.
- **Acceptance criteria**
  - UI renders with mocked data and then live data.
  - Dashboard charts render within 1s on typical datasets.
  - Access control: dashboard gated to developer role.

### WS6: Tool Consolidation & Deprecations
- **Steps**
  - Audit usage of arxiv/openreview tools; enumerate call sites and tests.
  - Introduce flags: `FF_REMOVE_ARXIV`, `FF_REMOVE_OPENREVIEW` to disable at runtime.
  - Migrate any unique behaviors into O3 wrapper (e.g., field filters).
  - Remove code, config, deps, tests after flags burn-in.
  - Update docs to reflect consolidation under O3.
- **Acceptance criteria**
  - No references to deprecated tools in code or tests.
  - O3 parity checks: equivalent or better results on sampled queries.
  - Dependency tree clean (no unused libraries).

### WS7: Testing & Performance
- **Unit tests**
  - Base tool interface contract tests.
  - Registry discovery and validation.
  - Agent scoring, fallback, circuit breaker.
  - Transparency storage and event sequencing.
- **Integration tests**
  - Orchestrator selecting among 2–3 mock tools with varying metadata.
  - O3 end-to-end with streaming events into UI.
- **E2E tests**
  - “User performs literature search; sees live results; exports data” happy path.
  - Failure modes: primary tool fails → fallback triggers; UI shows error.
- **Performance tests**
  - Selection latency: <15ms median, <50ms P95 with ≤10 tools.
  - Streaming throughput: sustain 10 events/sec per session without UI jank.
  - Storage write overhead: <10% increase vs baseline.
- **Acceptance criteria**
  - >85% coverage on core modules (`core/`, `tools/base`).
  - Performance budgets met on CI perf job.

### WS8: Documentation & Change Management
- **Docs**
  - Developer guide: how to add a tool (interface, metadata, tests).
  - Architecture overview: new directories, orchestrator, registry.
  - Transparency & dashboard usage: features, exports, retention.
  - Migration notes: deprecations, import path changes.
- **ADRs**
  - ADR-001: Filesystem & module layout.
  - ADR-002: Tool interface & metadata schema.
  - ADR-003: Recommendation strategy & fallback.
  - ADR-004: Transparency event model & streaming.
- **Acceptance criteria**
  - Docs reviewed by at least two engineers.
  - ADRs linked from `docs/architecture/`.

### WS9: Rollout, Flags, Observability, Security
- **Feature flags**
  - `FF_REGISTRY_ENABLED`: toggle registry discovery.
  - `FF_AGENT_RECOMMENDATION`: toggle recommendation engine use.
  - `FF_TRANSPARENCY_STREAMING`: toggle real-time event streaming.
  - `FF_REMOVE_ARXIV`, `FF_REMOVE_OPENREVIEW`: staged deprecations.
- **Observability**
  - Logging: selection decisions, tool lifecycle, errors (structured).
  - Metrics: selection latency, tool error rates, time-to-first-result, stream drops.
  - Tracing: spans for orchestrator → tool → transparency chain.
  - Dashboards: summary SLOs and per-tool health.
- **Security & privacy**
  - Redaction: configurable fields in stored results (PII scrubbing).
  - RBAC: developer dashboard access controls; avoid exposing raw tokens/keys.
  - Data retention: default 30–90 days; export bypass only for authorized roles.
- **Acceptance criteria**
  - Flags can roll back to legacy behavior instantly.
  - Dashboards live with meaningful baselines.
  - Security review signed off.

## Phase breakdown with tasks, dependencies, exit criteria

### Phase 1 (Weeks 1–2): Foundation
- **Tasks**
  - Create directories and module scaffolding in `tools/`, `core/`, `ui/`.
  - Implement base tool interface and metadata schema (WS2).
  - Build registry with discovery + validation + health checks.
  - Add orchestrator skeleton wired to registry (no recommendation yet).
- **Dependencies**: none.
- **Exit criteria**
  - Legacy tools callable via registry shim.
  - CI green; imports updated where needed.
  - ADR-001, ADR-002 merged.

### Phase 2 (Weeks 2–3): Tool Migration
- **Tasks**
  - Move existing tools into `tools/<name>/` structure; adapt to interface.
  - Implement O3 tool with full metadata and output mapping.
  - Add transparency storage writes (no streaming UI yet).
  - Gate arxiv/openreview via flags; migrate unique bits to O3.
- **Dependencies**: Phase 1 done.
- **Exit criteria**
  - All tool calls via orchestrator+registry.
  - Arxiv/openreview off by default behind flags.
  - ADR-004 draft created (event model).

### Phase 3 (Weeks 3–4): Agent Intelligence
- **Tasks**
  - Implement scoring function and recommendation engine.
  - Add fallback policy, circuit breaker, retry strategy.
  - Log selection decisions and scores to transparency/dev logs.
- **Dependencies**: Phase 2.
- **Exit criteria**
  - `FF_AGENT_RECOMMENDATION` on in staging, off in prod.
  - Integration tests pass for selection and fallbacks.
  - ADR-003 merged.

### Phase 4 (Weeks 4–5): Transparency Features
- **Tasks**
  - Implement in-process event bus; wire tool events to bus.
  - Add WebSocket/SSE endpoint; client subscriptions.
  - Build UI components (results panel, timeline, live indicator).
  - Build developer dashboard (runs explorer, health, decisions).
  - Implement export (JSON/CSV), with size and redaction safeguards.
- **Dependencies**: Phase 3.
- **Exit criteria**
  - `FF_TRANSPARENCY_STREAMING` on in staging.
  - Users see live O3 results and can export.
  - Dashboard shows tool health and selection samples.

### Phase 5 (Weeks 5–6): Testing & Documentation
- **Tasks**
  - Unit/integration/E2E/perf test suites complete.
  - Performance tuning for selection and streaming.
  - Remove arxiv/openreview code and dependencies (flags removed).
  - Finalize docs: developer guide, user guide, architecture.
- **Dependencies**: Phase 4.
- **Exit criteria**
  - Success criteria met (see below).
  - Flags: `FF_REMOVE_*` removed; new defaults stable.
  - Docs and ADRs complete.

## Success criteria (acceptance)
- **Autonomy**: Agent selects tools without hardcoded logic; replays show rationale.
- **Transparency**: O3 results and intermediate steps visible to users and devs; export works.
- **Organization**: New `tools/` structure; clear separation via `core/`, `ui/`.
- **Extensibility**: New tool added by dropping folder + metadata; auto-registered.
- **Quality**: All tests pass; performance budgets met; error budgets respected.

## Risk register and mitigations
- **Regression risk**: Strong integration tests; staged flags; canary rollout.
- **Performance degradation**: Lightweight scoring; cache capability matches; async streaming.
- **UX noise from streaming**: Coalesce partials; user-level filters and pause control.
- **O3 dependency**: Backoff and cached results; graceful degradation messaging.

## Operational runbook
- **Rollout**
  - Enable `FF_REGISTRY_ENABLED` → `FF_AGENT_RECOMMENDATION` → `FF_TRANSPARENCY_STREAMING` in staging, then prod.
  - Monitor selection latency, tool error rates, user time-to-first-result.
- **Rollback**
  - Disable flags to return to legacy flow.
  - Keep deprecation flags available until Phase 5 completes.
- **Monitoring**
  - Alerts on tool failure rate spikes, selection latency P95, streaming disconnects.

## Deliverables checklist
- **Code structure**: `tools/`, `core/`, `ui/` aligned to plan.
- **Contracts**: Tool interface & metadata schema; orchestrator API; event model.
- **Agent**: Recommendation engine + fallback policy with logs.
- **Transparency**: Storage, streaming, export; dev dashboard.
- **Consolidation**: O3 as single literature search tool; others removed.
- **Quality**: Tests, perf checks, dashboards.
- **Docs**: ADRs, developer guide, architecture, user guides.
