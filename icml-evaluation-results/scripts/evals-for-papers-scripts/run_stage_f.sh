#!/usr/bin/env bash
set -euo pipefail

STAGE=stage_f
ATTACH_DIR="file-upload-test-docs/files-for-evals/stage-f"

echo "== Mentor (manual) :: ${STAGE} =="
uv run python -m evaluation.scripts.run_manual_stage \
  --stage ${STAGE} \
  --force \
  --attachments-dir ${ATTACH_DIR}

echo "== GPT-5 baseline :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems openrouter:openai/gpt-5 \
  --baseline-mode \
  --output-label ${STAGE}_gpt5_baseline_oct21 \
  --skip-judge \
  --judge openrouter:qwen/qwen3-max \
  --attachments-dir ${ATTACH_DIR}

echo "== Claude baseline :: ${STAGE} =="
uv run python -m evaluation.scripts.run_single_turn_batch \
  --input evaluation/data/evals_single_turn.jsonl \
  --stage ${STAGE} \
  --systems openrouter:anthropic/claude-sonnet-4.5 \
  --baseline-mode \
  --output-label ${STAGE}_claude_baseline_oct21 \
  --skip-judge \
  --judge openrouter:qwen/qwen3-max \
  --attachments-dir ${ATTACH_DIR}
