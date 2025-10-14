from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from academic_research_mentor.rich_formatter import print_info, print_error

try:  # pragma: no cover - import convenience for direct execution
    from .run_manual_stage import run_stage, StageRunError
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from evaluation.scripts.run_manual_stage import run_stage, StageRunError  # type: ignore


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all evaluation stages sequentially")
    parser.add_argument("--force", action="store_true", help="Overwrite artifacts for all prompts")
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "stage_f", "A", "B", "C", "D", "E", "F"],
        help="Restrict execution to selected stage(s); repeatable",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected = args.stages or ["stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "stage_f"]

    summaries = []
    for stage in selected:
        try:
            summary = run_stage(stage, force=args.force)
            summaries.append(summary)
        except StageRunError as exc:
            print_error(f"Stage {stage}: {exc}")
            return 1

    combined = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stages": summaries,
        "total_prompts": sum(s.get("total_prompts", 0) for s in summaries),
    }

    reports_dir = Path("reports/evals")
    reports_dir.mkdir(parents=True, exist_ok=True)
    latest_path = reports_dir / "latest_run.json"
    latest_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print_info(f"Wrote combined results to {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
