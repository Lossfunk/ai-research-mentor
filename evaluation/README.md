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
- Use the single-turn runner: `uv run python evaluation/scripts/run_manual_stage.py --stage A`.
- Then run pairwise judging using `judges/pairwise_judge_prompt.md` and aggregate with `judges/aspect_rubrics.yaml`.
- Multi-turn stress test (student persona):
  ```bash
  uv run python evaluation/scripts/run_multi_turn_evals.py \
    --scenarios evaluation/multi_turn/scenarios.jsonl \
    --mentors openrouter:anthropic/claude-sonnet-4.5 openrouter:openai/gpt-5 \
    --output-dir reports/multi_turn_runs
  ```
  Add `--mock` for fast local smoke tests or `--killbox-dir` to redirect terminated-dialogue dumps.

Notes
- Pin models/providers in `.env` and keep seeds fixed.
- Sanitize any sensitive data before sharing artifacts.
