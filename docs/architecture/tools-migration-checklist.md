# Tools Migration Checklist (mentor_tools.py → tools/)

Phased, low-risk migration without changing behavior:

- Inventory current tools in `mentor_tools.py`
  - arxiv_search
  - openreview_fetch
  - venue_guidelines_get
  - math_ground
  - methodology_validate

- Create corresponding package structure under `src/academic_research_mentor/tools/`
  - `o3_search/` (future primary)
  - `heuristics/` (math_ground, methodology_validate)
  - `legacy/` (temporary arxiv/openreview wrappers during deprecation)

- For each tool:
  - Define a `Tool` class implementing `BaseTool` (WS2)
  - Keep function signatures intact; wrap the logic
  - Provide metadata (WS2): capabilities, input/output schema, latency/cost profile
  - Register via registry (manual until auto-discovery lands)

- Update call sites incrementally behind flags
  - Replace imports in `router.py`, `runtime.py`, `literature_review/` with registry lookups
  - Keep fallbacks to legacy functions during burn-in

- Deprecate arxiv/openreview (WS6)
  - Add flags: `FF_REMOVE_ARXIV`, `FF_REMOVE_OPENREVIEW`
  - Migrate any unique behaviors into the O3 tool
  - Remove code, tests, and deps after stabilization
