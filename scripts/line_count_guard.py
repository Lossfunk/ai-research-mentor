#!/usr/bin/env python3
"""
Line count guard for pre-commit.

- Checks only Python files under src/**
- Skips __init__.py files
- Fails if any file exceeds MAX allowed lines (default 200)

Usage (pre-commit passes filenames):
  python3 scripts/line_count_guard.py <files...>

Override max via environment variable LOC_MAX, e.g. LOC_MAX=220
"""

import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Resolve repository root (one directory above this script)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
MAX_LOC: int = int(os.environ.get("LOC_MAX", "200"))


def is_under_src_python_file(file_path: Path) -> bool:
    """Return True if file is a Python file under src/** and not __init__.py."""
    try:
        rel = file_path.resolve().relative_to(REPO_ROOT)
    except Exception:
        return False

    if not str(rel).startswith("src/"):
        return False
    if file_path.name == "__init__.py":
        return False
    return file_path.suffix == ".py"


def count_lines(file_path: Path) -> int:
    """Count total lines in the file (including blank lines)."""
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def filter_target_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path_obj = Path(p)
        if is_under_src_python_file(path_obj):
            files.append(path_obj)
    return files


def validate_line_counts(files: Iterable[Path]) -> List[Tuple[Path, int]]:
    """Return a list of (path, loc) that exceed the MAX_LOC threshold."""
    offenders: List[Tuple[Path, int]] = []
    for path in files:
        loc = count_lines(path)
        if loc > MAX_LOC:
            offenders.append((path, loc))
    return offenders


def main(argv: List[str]) -> int:
    targets = filter_target_files(argv[1:])
    offenders = validate_line_counts(targets)

    if offenders:
        print("The following files exceed the maximum allowed LOC:")
        for path, loc in sorted(offenders, key=lambda t: str(t[0])):
            rel = path.resolve().relative_to(REPO_ROOT)
            print(f"- {rel}: {loc} lines (max {MAX_LOC})")
        print(
            "\nPlease split these files so each stays at or below the limit."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
