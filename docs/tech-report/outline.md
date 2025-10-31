# Academic Research Mentor — Technical Report Outline

## 1. Abstract
- Motivation: accelerating research mentorship with tool-grounded guidance.
- Contributions: orchestrated tool routing, guidelines injection, transparent citations, staged evals.
- Key results (placeholder): Stage A baseline scores and examples.

## 2. Introduction
- Problem framing: novices and domain switchers need actionable, evidenced guidance.
- Challenges: hallucinations, ungrounded advice, tool brittleness, eval design.
- Our approach: orchestrator + guidelines engine + citation system + staged eval harness.

## 3. System Overview
- CLI/TUI entrypoints and shared backend services.
- Orchestrator and fallback policy: task selection, degraded modes.
- Guidelines engine: dynamic policy injection into prompts.
- Transparency store and logging.

References: src/academic_research_mentor/core/orchestrator.py:1, src/academic_research_mentor/guidelines_engine/injector.py:1, src/academic_research_mentor/core/transparency.py:200

## 4. Methods
- Prompting strategy and runtime directives for grounding and citations.
- Tool routing and registries; expected tool checks.
- Datasets and stages (A: mentorship; B: novelty/feasibility; C: planning).
- Metrics and judges: scaled and binary metrics, multi-judge aggregation.

References: evaluation/scripts/run_manual_stage.py:21, evaluation/data/evals_single_turn.jsonl:1, evaluation/scripts/judge_metrics.py:15, evaluation/scripts/run_judge_scores.py:1

## 5. Evaluation Protocol
- Artifact generation: responses, tool traces, metadata.
- LLM judges: provider:model specs, averaging, error handling.
- Human annotation: placeholders, IAA plan.
- Reproducibility commands and environment notes.

Run: `uv run python -m evaluation.scripts.run_manual_stage --stage A --force`
Then: `uv run python -m evaluation.scripts.run_judge_scores --stage stage_a --judge openrouter:anthropic/claude-4-sonnet --judge openrouter:google/gemini-2.5-flash --annotator auto --label baseline`

## 6. Results (Milestone 1 — Stage A Baseline)
- Tool routing pass rate, mean Actionability, mean Citation Quality.
- Qualitative examples with inline [P#]/[G#]/[n] citations and tool traces.
- Failure modes and ablations (Guidelines OFF vs DYNAMIC; fallback OFF vs ON).

## 7. Discussion
- Strengths: grounded mentorship, citation policy enforcement.
- Limitations: judge variance, web search dependence, prompt drift.
- Future work: multi-turn, attachments-first synthesis, expanded Stage B/C.

## 8. Related Work
- AI writing assistants and research copilots; tool-augmented LLMs; evaluation with LLM judges.

## 9. Ethics & Safety
- Accurate citations, risk/expectation management, PII handling in convo logs.

## 10. Reproducibility & Config
- Required env: `.env` with `OPENROUTER_API_KEY`, optional `TAVILY_API_KEY`.
- Toggle flags: `FF_AGENT_RECOMMENDATION`, `ARM_GUIDELINES_MODE`.
- Scripted runner: `scripts/eval_baseline.py` (below).

## Appendix A — File Map
- Entry: `src/academic_research_mentor/cli.py`, `src/academic_research_mentor/tui.py`
- Orchestrator: `src/academic_research_mentor/core/orchestrator.py`
- Guidelines: `src/academic_research_mentor/guidelines_engine/`
- Transparency/logging: `src/academic_research_mentor/core/transparency.py`, `src/academic_research_mentor/session_logging.py`
- Evals: `evaluation/scripts/*`, `evaluation/data/evals_single_turn.jsonl`
