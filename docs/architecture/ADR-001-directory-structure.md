# ADR-001: Directory Structure and Namespacing

## Status
Accepted

## Context
The project previously had tools and core logic mixed within `src/academic_research_mentor/` and scattered utility modules. We intend to migrate to a clearer separation of concerns to support upcoming workstreams (tool registry, orchestrator, transparency).

## Decision
- Introduce namespaced directories under the package:
  - `src/academic_research_mentor/core/` for orchestrator, transparency, and agent scaffolding
  - `src/academic_research_mentor/tools/` for tool interface and registry; shared `tools/utils/`
- Keep existing entrypoints and imports intact to avoid runtime breakage.
- Add minimal, <200-line scaffolding modules only; defer behavior changes to later workstreams.

## Consequences
- Clear locations for future implementations without immediate refactors.
- No changes to CLI behavior (`uv run academic-research-mentor`) in WS1.
- Developers can begin adding tests and docs around new modules without risk.
