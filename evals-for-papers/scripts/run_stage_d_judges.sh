#!/usr/bin/env bash
set -euo pipefail

STAGE=stage_d

echo "== Absolute judging (mentor) :: ${STAGE} =="
uv run python -m evaluation.scripts.run_judge_scores \
  --stage ${STAGE} \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --annotator single_turn_oct21 \
  --label ${STAGE}_mentor_metrics_oct21 \
  --system mentor_manual

echo "== Absolute judging (gpt-5 baseline) :: ${STAGE} =="
uv run python -m evaluation.scripts.run_judge_scores \
  --stage ${STAGE} \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --annotator single_turn_oct21 \
  --label ${STAGE}_gpt5_metrics_oct21 \
  --system openrouter:openai/gpt-5

echo "== Absolute judging (claude baseline) :: ${STAGE} =="
uv run python -m evaluation.scripts.run_judge_scores \
  --stage ${STAGE} \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --annotator single_turn_oct21 \
  --label ${STAGE}_claude_metrics_oct21 \
  --system openrouter:anthropic/claude-sonnet-4.5
