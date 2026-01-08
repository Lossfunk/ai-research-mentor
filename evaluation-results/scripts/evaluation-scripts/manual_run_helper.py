"""Utility to guide manual evaluation runs for Phase 2."""

from pathlib import Path


STAGES = [
    ("stage_a", "Pre idea"),
    ("stage_b", "Idea"),
    ("stage_c", "Research plan"),
]


def _stage_prompts_path(stage: str) -> Path:
    return Path("evaluation/data/evals_single_turn.jsonl")


def _result_slots(stage: str) -> dict[str, Path]:
    base = Path("evals-for-papers/results")
    return {
        "raw_log_dir": base / "raw_logs" / stage,
        "analysis_dir": base / "analysis_reports" / stage,
        "agreement_dir": base / "inter_annotator_agreement" / stage,
    }


def print_checklist() -> None:
    print("Manual Evaluation Checklist\n")
    print("1. Set required environment variables (e.g., OPENROUTER_API_KEY=sk-or-...).")
    print("2. Launch the mentor CLI: 'uv run academic-research-mentor'.")
    print("3. For each prompt in evaluation/data/evals_single_turn.jsonl:")
    print("   a. Ask the prompt and save the full response text to evals-for-papers/results/raw_logs/<stage>/<prompt_id>.txt.")
    print("   b. Export the tool trace (if available) to evals-for-papers/results/raw_logs/<stage>/<prompt_id>_tools.json.")
    print("   c. Log run metadata (timestamp, model, seed) in evals-for-papers/results/analysis_reports/<stage>/<prompt_id>_meta.json.")
    print("4. After both annotators review a response, record scores in evaluation/annotation/templates/manual_annotation_sheet.csv.")
    print("5. When disagreements exceed 0.5 on rubric scales, record the adjudicated score in evals-for-papers/results/inter_annotator_agreement/<stage>/<prompt_id>_adjudicated.json.")
    print("6. Back up transcripts and completed sheets into a timestamped folder under evals-for-papers/results/analysis_reports.")


if __name__ == "__main__":
    print_checklist()
