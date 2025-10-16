# Evaluation Suite Overview

This folder contains artifacts to run the paper-focused evaluation of the academic research mentor.

Key Files
- `MENTOR_EVAL_PLAN.md` — end-to-end evaluation design (tracks, metrics, judges, matrix).
- `personas/` — persona YAML cards used to condition dialogues and judging context.
- `tasks/` — scenario/task YAMLs with success criteria and expected tools.
- `judges/pairwise_judge_prompt.md` — LLM-as-a-judge pairwise template.
- `judges/aspect_rubrics.yaml` — aspect definitions, anchors, and weights.
- `matrix.yaml` — study matrix (topics × difficulty × personas × systems × seeds × tasks).
- `../evals-for-papers/results/` — raw transcripts/tool traces and judge reports (generated).

Running (prototype)
- Use existing single-turn runner for prompts: `uv run python evaluation/scripts/run_manual_stage.py --stage A`.
- For multi-turn, implement a small orchestrator that:
  - loads a persona + task,
  - runs dialogues up to `turn_budget`,
  - writes transcripts under `../evals-for-papers/results/raw_logs/<scenario_id>/`.
- Then run pairwise judging using `judges/pairwise_judge_prompt.md` and aggregate with `judges/aspect_rubrics.yaml`.

Notes
- Pin models/providers in `.env` and keep seeds fixed.
- Sanitize any sensitive data before sharing artifacts.

