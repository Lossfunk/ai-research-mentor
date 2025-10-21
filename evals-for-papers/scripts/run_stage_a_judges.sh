#!/usr/bin/env bash
set -euo pipefail

STAGE=stage_a

echo "== Absolute judging :: ${STAGE} =="
uv run python -m evaluation.scripts.run_judge_scores \
  --stage ${STAGE} \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --annotator single_turn_oct21 \
  --label ${STAGE}_metrics_oct21

echo "== Pairwise mentor vs GPT-5 :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems mentor_manual openrouter:openai/gpt-5 \
  --skip-generation \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --output-label ${STAGE}_pairwise_mentor_vs_gpt5_oct21 \
  --pairwise-judge openrouter:google/gemini-2.5-flash-lite \
  --pairwise-judge openrouter:deepseek/deepseek-v3.2-exp \
  --pairwise-judge openrouter:x-ai/grok-4-fast \
  --pairwise-repeats 3 \
  --pairwise-seed 42

echo "== Pairwise mentor vs Claude baseline :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems mentor_manual openrouter:anthropic/claude-sonnet-4.5 \
  --skip-generation \
  --judge openrouter:google/gemini-2.5-flash-lite \
  --output-label ${STAGE}_pairwise_mentor_vs_claude_oct21 \
  --pairwise-judge openrouter:google/gemini-2.5-flash-lite \
  --pairwise-judge openrouter:deepseek/deepseek-v3.2-exp \
  --pairwise-judge openrouter:x-ai/grok-4-fast \
  --pairwise-repeats 3 \
  --pairwise-seed 42
