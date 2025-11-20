#!/usr/bin/env bash
set -euo pipefail

STAGE=stage_a

echo "== Pairwise mentor vs GPT-5 :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems mentor_manual openrouter:openai/gpt-5 \
  --skip-generation \
  --pairwise-only \
  --judge openrouter:google/gemini-2.5-pro \
  --output-label ${STAGE}_pairwise_mentor_vs_gpt5_oct21 \
  --pairwise-judge openrouter:google/gemini-2.5-pro \
  --pairwise-judge openrouter:deepseek/deepseek-v3.2-exp \
  --pairwise-judge openrouter:x-ai/grok-4-fast

echo "== Pairwise mentor vs Claude baseline :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems mentor_manual openrouter:anthropic/claude-sonnet-4.5 \
  --skip-generation \
  --pairwise-only \
  --judge openrouter:google/gemini-2.5-pro \
  --output-label ${STAGE}_pairwise_mentor_vs_claude_oct21 \
  --pairwise-judge openrouter:google/gemini-2.5-pro \
  --pairwise-judge openrouter:deepseek/deepseek-v3.2-exp \
  --pairwise-judge openrouter:x-ai/grok-4-fast
