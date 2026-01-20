"""Shared utilities for stage handling - no heavy dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


class StageRunError(RuntimeError):
    """Error during stage execution."""
    pass


def normalize_stage(stage: str) -> Tuple[str, str]:
    """Convert stage input to (letter, folder) tuple."""
    value = stage.strip().lower()
    if value in {"a", "stage_a"}:
        return "A", "stage_a"
    if value in {"b", "stage_b"}:
        return "B", "stage_b"
    if value in {"c", "stage_c"}:
        return "C", "stage_c"
    if value in {"d", "stage_d"}:
        return "D", "stage_d"
    if value in {"e", "stage_e"}:
        return "E", "stage_e"
    if value in {"f", "stage_f"}:
        return "F", "stage_f"
    raise StageRunError(f"Unknown stage: {stage}")


def ensure_stage_directories(stage_folder: str) -> Tuple[Path, Path, Path]:
    """Create and return (raw_dir, analysis_dir, iaa_dir) for a stage."""
    base = Path("evals-for-papers/results")
    raw_dir = base / "raw_logs" / stage_folder
    analysis_dir = base / "analysis_reports" / stage_folder
    iaa_dir = base / "inter_annotator_agreement" / stage_folder
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    iaa_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, analysis_dir, iaa_dir
