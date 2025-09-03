# ADR-003: Orchestrator and Agent Integration (Phased)

## Status
Accepted (Plan for WS3; scaffold present in WS1)

## Context
We need a clear path to transition from the current `runtime.build_agent` + manual `router` to a registry-backed orchestrator with recommendation-driven tool selection.

## Decision
- Keep current CLI + runtime as the source of truth in WS1/WS2.
- Introduce `core.orchestrator.Orchestrator` as a stable facade to coordinate tasks.
- Integrate the orchestrator with the agent in WS3 behind a feature flag (`FF_AGENT_RECOMMENDATION`).
- Maintain a rollback path to legacy behavior until WS5.

## Consequences
- Developers can write integration tests against `Orchestrator` without touching the CLI.
- We avoid breaking changes during early phases and de-risk the rollout.

## Migration Plan (High Level)
1. WS1: Orchestrator scaffold + transparency store (in-memory)
2. WS2: Tool registry/metadata scaffold ready for integration
3. WS3: Agent selection algorithm + orchestrator wiring behind a flag
4. WS4: Transparency streaming + UI integration
5. WS5: Remove legacy routing paths after burn-in
