#!/usr/bin/env python3
"""Convert multi-turn evaluation JSONL outputs into per-scenario folders.

Usage:
    uv run python -m scripts.convert_multi_turn_results \
        --input evaluation/results/multi_turn_example_2/openrouter_openai_gpt-5.jsonl \
        --output evals-for-papers/results/raw_logs/multi_turn

The script mirrors the organization used for single-turn evals by creating one
directory per scenario and writing:
    * <scenario>__<system>.json           # full result record (pretty JSON)
    * <scenario>__<system>_tools.json     # tool runs only
    * <scenario>__<system>_transcript.txt # human-readable transcript
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    records: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _format_transcript(transcript: Sequence[Mapping[str, object]]) -> str:
    lines: List[str] = []
    for turn in transcript:
        role = str(turn.get("role", "unknown")).upper()
        turn_idx = turn.get("turn")
        prefix = f"Turn {turn_idx}" if turn_idx not in (None, "") else "System"
        content = str(turn.get("content", "")).strip()
        lines.append(f"[{prefix}] {role}:\n{content}\n")
    return "\n".join(lines).strip() + "\n"


def _write_record(record: Mapping[str, object], out_root: Path) -> None:
    scenario_id = str(record.get("scenario_id", "unknown_scenario"))
    system_id = _sanitize(str(record.get("system_id", "unknown_system")))

    scenario_dir = out_root / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    base = f"{scenario_id}__{system_id}"

    # Full pretty JSON record
    (scenario_dir / f"{base}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Tool runs only
    tool_runs = record.get("tool_runs", [])
    (scenario_dir / f"{base}_tools.json").write_text(
        json.dumps(tool_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Human-readable transcript
    transcript = record.get("transcript", [])
    text = _format_transcript(transcript if isinstance(transcript, Iterable) else [])
    (scenario_dir / f"{base}_transcript.txt").write_text(text, encoding="utf-8")


def convert(input_path: Path, output_dir: Path) -> int:
    records = _read_jsonl(input_path)
    if not records:
        print(f"No records found in {input_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        _write_record(record, output_dir)

    print(f"Exported {len(records)} record(s) to {output_dir}")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert multi-turn JSONL into per-scenario folders")
    parser.add_argument("--input", required=True, help="Path to JSONL file produced by multi-turn evals")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write organized outputs (one subdirectory per scenario)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    convert(input_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
