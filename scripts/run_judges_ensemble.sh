#!/usr/bin/env bash
set -euo pipefail

# Run expert and student judges with the upgraded ensemble and produce LOFO summaries.
#
# Usage:
#   scripts/run_judges_ensemble.sh [stage_a|stage_c|stage_f|all] [system_id]
#
# Examples:
#   scripts/run_judges_ensemble.sh all mentor_manual
#   scripts/run_judges_ensemble.sh stage_a
#
# Environment (optional):
#   LABEL_EXPERT=expert_absolute_pro
#   LABEL_STUDENT=student_outcome_judge
#   ANNOTATOR_EXPERT=expert_judge_upgrade
#   ANNOTATOR_STUDENT=student_judge_upgrade

STAGE_ARG="${1:-all}"
SYSTEM_FILTER="${2:-}"

LABEL_EXPERT="${LABEL_EXPERT:-expert_absolute_pro}"
LABEL_STUDENT="${LABEL_STUDENT:-student_outcome_judge}"
ANNOTATOR_EXPERT="${ANNOTATOR_EXPERT:-expert_judge_upgrade}"
ANNOTATOR_STUDENT="${ANNOTATOR_STUDENT:-student_judge_upgrade}"

JUDGES=(
  "openrouter:google/gemini-2.5-pro"
  "openrouter:deepseek/deepseek-v3.2-exp"
  "openrouter:x-ai/grok-4-fast"
)

if [[ "${STAGE_ARG}" == "all" ]]; then
  STAGES=(stage_a stage_c stage_d stage_e stage_f)
else
  STAGES=("${STAGE_ARG}")
fi

run_expert() {
  local stage="$1"
  local cmd=(uv run python -m evaluation.scripts.run_judge_scores \
    --stage "${stage}" \
    --judge "${JUDGES[0]}" \
    --judge "${JUDGES[1]}" \
    --judge "${JUDGES[2]}" \
    --annotator "${ANNOTATOR_EXPERT}" \
    --label "${LABEL_EXPERT}")
  if [[ -n "${SYSTEM_FILTER}" ]]; then
    cmd+=(--system "${SYSTEM_FILTER}")
  fi
  "${cmd[@]}"
}

run_student() {
  local stage="$1"
  local cmd=(uv run python -m evaluation.scripts.run_student_judge_scores \
    --stage "${stage}" \
    --judge "${JUDGES[0]}" \
    --judge "${JUDGES[1]}" \
    --judge "${JUDGES[2]}" \
    --annotator "${ANNOTATOR_STUDENT}" \
    --label "${LABEL_STUDENT}")
  if [[ -n "${SYSTEM_FILTER}" ]]; then
    cmd+=(--system "${SYSTEM_FILTER}")
  fi
  "${cmd[@]}"
}

run_lofo() {
  local stage="$1"
  uv run python -m evaluation.scripts.analyze_lofo \
    --stage "${stage}" \
    --label "${LABEL_STUDENT}"
  uv run python -m evaluation.scripts.analyze_lofo \
    --stage "${stage}" \
    --label "${LABEL_EXPERT}"
}

for STAGE in "${STAGES[@]}"; do
  echo "[ensemble] Running expert judges for ${STAGE}..."
  run_expert "${STAGE}"
  echo "[ensemble] Running student judges for ${STAGE}..."
  run_student "${STAGE}"
  echo "[ensemble] Generating LOFO summaries for ${STAGE}..."
  run_lofo "${STAGE}"
done

echo "[ensemble] Done. Labels: expert=${LABEL_EXPERT}, student=${LABEL_STUDENT}"
