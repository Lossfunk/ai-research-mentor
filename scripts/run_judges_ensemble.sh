#!/usr/bin/env bash
set -euo pipefail

# Run expert and student judges with the upgraded ensemble and produce LOFO summaries.
#
# Usage:
#   scripts/run_judges_ensemble.sh [stage_a|stage_c|stage_d|stage_e|stage_f|all] [system_id] [prompt_ids]
#
# Examples:
#   scripts/run_judges_ensemble.sh all mentor_manual
#   scripts/run_judges_ensemble.sh stage_a mentor_manual "stage_a_01,stage_a_02"
#
# Environment (optional):
#   LABEL_EXPERT=expert_absolute_pro
#   LABEL_STUDENT=student_outcome_judge
#   ANNOTATOR_EXPERT=expert_judge_upgrade
#   ANNOTATOR_STUDENT=student_judge_upgrade
#   SYSTEMS="mentor_manual,openrouter:anthropic/claude-sonnet-4.5,openrouter:openai/gpt-5"

STAGE_ARG="${1:-all}"
SYSTEM_FILTER="${2:-}"
PROMPTS_ARG="${3:-}"

LABEL_EXPERT_BASE="${LABEL_EXPERT:-expert_absolute_pro}"
LABEL_STUDENT_BASE="${LABEL_STUDENT:-student_outcome_judge}"
ANNOTATOR_EXPERT="${ANNOTATOR_EXPERT:-expert_judge_upgrade}"
ANNOTATOR_STUDENT="${ANNOTATOR_STUDENT:-student_judge_upgrade}"

JUDGES=(
  "openrouter:google/gemini-2.5-pro"
  "openrouter:deepseek/deepseek-v3.2-exp"
  "openrouter:x-ai/grok-4-fast"
)

DEFAULT_SYSTEMS=(
  "mentor_manual"
  "openrouter:anthropic/claude-sonnet-4.5"
  "openrouter:openai/gpt-5"
)

sanitize_label_component() {
  local raw="${1:-}"
  local lower
  lower="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  local cleaned="${lower//[^a-z0-9]/_}"
  # collapse multiple underscores
  cleaned="${cleaned//__/_}"
  cleaned="${cleaned//__/_}"
  cleaned="${cleaned//__/_}"
  # trim leading/trailing underscores
  cleaned="${cleaned##_}"
  cleaned="${cleaned%%_}"
  if [[ -z "${cleaned}" ]]; then
    cleaned="default"
  fi
  printf "%s" "${cleaned}"
}

if [[ "${STAGE_ARG}" == "all" ]]; then
  STAGES=(stage_a stage_c stage_d stage_e stage_f)
else
  STAGES=("${STAGE_ARG}")
fi

SYSTEMS_OVERRIDE="${SYSTEMS:-}"
if [[ -n "${SYSTEM_FILTER}" ]]; then
  TARGET_SYSTEMS=("${SYSTEM_FILTER}")
elif [[ -n "${SYSTEMS_OVERRIDE}" ]]; then
  IFS=',' read -r -a TARGET_SYSTEMS <<< "${SYSTEMS_OVERRIDE}"
else
  TARGET_SYSTEMS=("${DEFAULT_SYSTEMS[@]}")
fi

run_expert() {
  local stage="$1"
  local system_id="$2"
  local label="$3"
  local cmd=(uv run python -m evaluation.scripts.run_judge_scores \
    --stage "${stage}" \
    --judge "${JUDGES[0]}" \
    --judge "${JUDGES[1]}" \
    --judge "${JUDGES[2]}" \
    --annotator "${ANNOTATOR_EXPERT}" \
    --label "${label}" \
    --system "${system_id}")
  if [[ -n "${PROMPTS_ARG}" ]]; then
    # Accept comma-separated list; split into args
    IFS=',' read -r -a PIDS <<< "${PROMPTS_ARG}"
    for pid in "${PIDS[@]}"; do
      pid_trimmed="${pid//[[:space:]]/}"
      [[ -n "${pid_trimmed}" ]] && cmd+=(--prompt-id "${pid_trimmed}")
    done
  fi
  "${cmd[@]}"
}

run_student() {
  local stage="$1"
  local system_id="$2"
  local label="$3"
  local cmd=(uv run python -m evaluation.scripts.run_student_judge_scores \
    --stage "${stage}" \
    --judge "${JUDGES[0]}" \
    --judge "${JUDGES[1]}" \
    --judge "${JUDGES[2]}" \
    --annotator "${ANNOTATOR_STUDENT}" \
    --label "${label}" \
    --system "${system_id}")
  if [[ -n "${PROMPTS_ARG}" ]]; then
    IFS=',' read -r -a PIDS <<< "${PROMPTS_ARG}"
    for pid in "${PIDS[@]}"; do
      pid_trimmed="${pid//[[:space:]]/}"
      [[ -n "${pid_trimmed}" ]] && cmd+=(--prompt-id "${pid_trimmed}")
    done
  fi
  "${cmd[@]}"
}

run_lofo() {
  local stage="$1"
  local label="$2"
  uv run python -m evaluation.scripts.analyze_lofo \
    --stage "${stage}" \
    --label "${label}"
}

for STAGE in "${STAGES[@]}"; do
  for SYSTEM_ID in "${TARGET_SYSTEMS[@]}"; do
    # Skip empty entries (possible if SYSTEMS env has trailing commas)
    if [[ -z "${SYSTEM_ID}" ]]; then
      continue
    fi
    sanitized="$(sanitize_label_component "${SYSTEM_ID}")"
    expert_label="${LABEL_EXPERT_BASE}_${sanitized}"
    student_label="${LABEL_STUDENT_BASE}_${sanitized}"

    echo "[ensemble] Running expert judges for ${STAGE} (${SYSTEM_ID}) -> ${expert_label}..."
    run_expert "${STAGE}" "${SYSTEM_ID}" "${expert_label}"

    echo "[ensemble] Running student judges for ${STAGE} (${SYSTEM_ID}) -> ${student_label}..."
    run_student "${STAGE}" "${SYSTEM_ID}" "${student_label}"

    echo "[ensemble] Generating LOFO summaries for ${STAGE} (${SYSTEM_ID})..."
    run_lofo "${STAGE}" "${student_label}"
    run_lofo "${STAGE}" "${expert_label}"
  done
done

echo "[ensemble] Done. Expert base=${LABEL_EXPERT_BASE}, student base=${LABEL_STUDENT_BASE}"
