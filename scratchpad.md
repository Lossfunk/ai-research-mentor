# Scratchpad

Purpose: Track decisions and context while executing WorkStream 1 (directory structure).

Context:
- Current package layout uses `src/academic_research_mentor` with CLI entrypoint via `pyproject.toml` (script: `academic-research-mentor`).
- We'll avoid breaking the existing CLI (`uv run academic-research-mentor`).
- We will introduce `core/` and `tools/` under the package namespace `src/academic_research_mentor/` instead of top-level modules to preserve imports.
- Keep files concise (<200 LOC each).

Decisions:
- Create `src/academic_research_mentor/core/` with `orchestrator.py`, `transparency.py` and `__init__.py`.
- Create `src/academic_research_mentor/tools/` with `base_tool.py`, `__init__.py`, and `utils/`.
- Do not refactor existing imports yet; add scaffolding only.
- No auto-discovery yet; registry exposes manual registration API (auto-discovery can be added later).

Next steps:
1) Remove mistakenly created root-level `core/`, `tools/`, `ui/`.
2) Scaffold namespaced directories and modules.
3) Export orchestrator/transparency in `core/__init__.py`.
4) Update `pyproject.toml` script target to `academic_research_mentor.cli:main`.
5) Commit; verify `uv run academic-research-mentor` and `uv run python main.py` work.
