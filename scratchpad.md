# Scratchpad

Purpose: Track decisions and context while executing WS1 and WS2.

Context:
- Current package layout uses `src/academic_research_mentor` with CLI entrypoint via `pyproject.toml` (script: `academic-research-mentor`).
- We'll avoid breaking the existing CLI (`uv run academic-research-mentor`).
- We will introduce `core/` and `tools/` under the package namespace `src/academic_research_mentor/` instead of top-level modules to preserve imports.
- Keep files concise (<200 LOC each).

WS1 Decisions & Actions (completed):
- Created `src/academic_research_mentor/core/` with `orchestrator.py`, `transparency.py`, `agent.py`, and `__init__.py` exports.
- Created `src/academic_research_mentor/tools/` with `base_tool.py`, `__init__.py` (registry), and `utils/`.
- Added root-level `main.py` shim and updated `pyproject.toml` script to `academic_research_mentor.cli:main`.
- Wrote ADR-001/002/003 and tools migration checklist; updated README with new layout.
- Kept runtime behavior unchanged; CLI sanity checks pass.

WS2 Decisions & Actions (in progress):
- Extended `BaseTool` with lifecycle (`initialize`, `cleanup`), `can_handle`, and `get_metadata`.
- Implemented registry `auto_discover()` with validation; added tests.
- Added `tools/o3_search/tool.py` with metadata and `can_handle`.
- Added `tools/legacy/arxiv/tool.py` wrapper for legacy arXiv search (auto-discovered as fallback).
- Added `core/bootstrap.py`; wired CLI to bootstrap registry behind `FF_REGISTRY_ENABLED`.
- Added CLI flags:
  - `--list-tools`: force discovery and list tools.
  - `--show-candidates "<goal>"`: run orchestrator selection-only and print candidates.
- Orchestrator now returns candidates, prioritizing `o3_search` (score 10) and treating `legacy_*` as fallback (score 0.5); reduces O3 score if client unavailable.
- Test suite expanded (5 tests), all passing under uv/conda.

Upcoming WS3 (not started):
Plan for WS3 (next incremental changes):
- Add recommendation scoring beyond static weights in `core/recommendation.py` (new, <200 LOC).
- Feature flag: `FF_AGENT_RECOMMENDATION` to gate orchestrator using the scorer.
- Scoring signals: prefer `o3_search`, penalize `legacy_*`, consider `can_handle`, metadata cost and reliability, basic domain keyword match.
- Orchestrator: call recommender when flag on; return candidates with rationales (no execution yet).
- CLI: add `--recommend "<goal>"` to print top tool and reasons.
- Tests: validate scoring orders `o3_search` > `legacy_arxiv_search`.

WS3 Decisions & Actions (completed):
- Implemented `core/recommendation.py` and wired orchestrator under `FF_AGENT_RECOMMENDATION`.
- Added CLI `--recommend` to print top tool and rationale.
- Added WS3 tests:
  - Recommender prefers o3_search over legacy.
  - Orchestrator uses recommender when FF_AGENT_RECOMMENDATION=1.
- Updated scratchpad.md with WS3 progress and next tasks.
- All tests pass (7 total).

WS3 Extension - Tool Integration (completed):
- Enhanced O3SearchTool: Implemented real execution combining arXiv + OpenReview searches.
- Extended Orchestrator: Added `execute_task()` method with tool execution + fallback logic.
- Integrated Context Builder: Modified literature search to use orchestrator-based tool selection when `FF_REGISTRY_ENABLED=1`.
- Maintained backward compatibility: Falls back to legacy search when orchestrator unavailable.
- End-to-end testing: CLI queries now use dynamic tool selection (o3_search preferred, score: 2.1).
- Real literature search results: Papers returned from actual searches via orchestrator.

Next: Consider fallback policy improvements, additional tools, or execution optimization. The foundation is ready for further tool ecosystem expansion.
