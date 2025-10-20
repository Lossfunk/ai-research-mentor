#!/usr/bin/env python3
"""Pairwise judge runner for multi-turn transcripts.

This script reuses the pairwise judging flow from the single-turn pipeline but
operates directly on aggregated multi-turn transcript folders. It expects the
directory structure created by `scripts.convert_multi_turn_results`.

Example:
    uv run python -m scripts.judge_multi_turn_pairwise \
        --mentor evals-for-papers/results/raw_logs/multi_turn_mentor/research_methodology_consultation/research_methodology_consultation__mentor.txt \
        --baseline evals-for-papers/results/raw_logs/multi_turn_gpt5/research_methodology_consultation/research_methodology_consultation__openrouter_openai_gpt-5_transcript.txt \
        --label multi_turn_gpt5 \
        --judge openrouter:anthropic/claude-3.5-sonnet

In practice you should point the script at two directories, each containing
per-scenario transcript files, so the CLI provides a directory-based interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from academic_research_mentor.rich_formatter import print_error, print_info, print_success

from evaluation.scripts.judge_utils import build_judge_clients, save_judge_payload
from evaluation.scripts.run_single_turn_batch import parse_pairwise_output

PAIRWISE_PROMPT_PATH = Path("evaluation/judges/pairwise_judge_prompt.md")


def _find_transcript_files(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for transcript in root.rglob("*_transcript.txt"):
        scenario = transcript.parent.name
        mapping[scenario] = transcript
    return mapping


def _digests(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _load_transcript(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _paired_scenarios(mentor_dir: Path, baseline_dir: Path) -> List[str]:
    mentor_map = _find_transcript_files(mentor_dir)
    baseline_map = _find_transcript_files(baseline_dir)
    shared = sorted(set(mentor_map).intersection(baseline_map))
    missing_mentor = sorted(set(baseline_map) - set(mentor_map))
    missing_baseline = sorted(set(mentor_map) - set(baseline_map))
    if missing_mentor:
        print_error(f"Mentor transcripts missing for scenarios: {missing_mentor}")
    if missing_baseline:
        print_error(f"Baseline transcripts missing for scenarios: {missing_baseline}")
    pairs: List[str] = []
    for scenario in shared:
        mentor_path = mentor_map[scenario]
        baseline_path = baseline_map[scenario]
        if mentor_path.stat().st_size == 0 or baseline_path.stat().st_size == 0:
            print_error(f"Skipping {scenario}: empty transcript")
            continue
        pairs.append(scenario)
    return pairs


def run_pairwise_judges(
    scenarios: Sequence[str],
    mentor_dir: Path,
    baseline_dir: Path,
    *,
    judge_specs: Sequence[str],
    label: str,
    repeats: int,
    seed: Optional[int],
) -> None:
    if not PAIRWISE_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Pairwise judge prompt missing: {PAIRWISE_PROMPT_PATH}")
    judge_clients = build_judge_clients(judge_specs)
    if not judge_clients:
        raise RuntimeError("No pairwise judge clients could be initialized")

    prompt_template = PAIRWISE_PROMPT_PATH.read_text(encoding="utf-8")
    digest = _digests(PAIRWISE_PROMPT_PATH)
    output_dir = Path("evals-for-papers/results/analysis/multi_turn_pairwise") / label
    output_dir.mkdir(parents=True, exist_ok=True)

    mentor_map = _find_transcript_files(mentor_dir)
    baseline_map = _find_transcript_files(baseline_dir)

    rng = random.Random(seed)

    for scenario in scenarios:
        mentor_text = _load_transcript(mentor_map[scenario])
        baseline_text = _load_transcript(baseline_map[scenario])

        judge_outputs: List[Dict[str, object]] = []
        votes = {"mentor": 0, "baseline": 0, "tie": 0}

        for name, client in judge_clients:
            for run_idx in range(repeats):
                judge_label = f"{name}#{run_idx + 1}" if repeats > 1 else name
                mentor_as_a = rng.random() < 0.5
                if mentor_as_a:
                    prompt_text = prompt_template.replace("{{SYSTEM_A}}", mentor_text).replace("{{SYSTEM_B}}", baseline_text)
                    assignment = {"A": "mentor", "B": "baseline"}
                else:
                    prompt_text = prompt_template.replace("{{SYSTEM_A}}", baseline_text).replace("{{SYSTEM_B}}", mentor_text)
                    assignment = {"A": "baseline", "B": "mentor"}
                try:
                    raw = client.invoke(prompt_text)
                    response = raw if isinstance(raw, str) else getattr(raw, "content", "") or str(raw)
                except Exception as exc:  # noqa: BLE001
                    judge_outputs.append({"judge": judge_label, "raw": str(exc), "parsed": {"winner": "Tie"}, "error": True})
                    votes["tie"] += 1
                    continue

                parsed = parse_pairwise_output(response) or {"winner": "Tie"}
                winner = parsed.get("winner")
                if winner not in {"A", "B", "Tie"}:
                    winner = "Tie"
                if winner == "Tie":
                    votes["tie"] += 1
                    winner_actual = "tie"
                else:
                    actual = assignment[winner]
                    votes[actual] += 1
                    winner_actual = actual
                judge_outputs.append(
                    {
                        "judge": judge_label,
                        "raw": response,
                        "parsed": parsed,
                        "mapping": assignment,
                        "winner_actual": winner_actual,
                    }
                )

        max_votes = max(votes.values()) if votes else 0
        if list(votes.values()).count(max_votes) > 1 or max_votes == 0:
            majority = "tie"
        else:
            majority = max(votes, key=votes.get)
        payload = {
            "scenario": scenario,
            "label": label,
            "judges": [
                f"{name}#{run_idx + 1}" if repeats > 1 else name
                for name, _ in judge_clients
                for run_idx in range(repeats)
            ],
            "judge_outputs": judge_outputs,
            "winner": majority,
            "votes": votes,
            "pairwise_prompt_digest": digest,
            "mentor_transcript": str(mentor_map[scenario]),
            "baseline_transcript": str(baseline_map[scenario]),
        }
        save_judge_payload(output_dir / f"{scenario}_pairwise.json", payload)
        print_success(f"Judged {scenario}: winner={majority} votes={votes}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise judge on multi-turn transcripts")
    parser.add_argument("--mentor-dir", required=True, help="Directory containing mentor transcripts (_transcript.txt files)")
    parser.add_argument("--baseline-dir", required=True, help="Directory containing baseline transcripts")
    parser.add_argument("--judge", dest="judges", action="append", required=True, help="Judge model spec (repeatable)")
    parser.add_argument("--label", required=True, help="Label for output directory")
    parser.add_argument("--prompt-ids", nargs="*", help="Optional subset of scenarios to score")
    parser.add_argument("--repeats", type=int, default=1, help="Number of passes per judge (default: 1)")
    parser.add_argument("--seed", type=int, help="Optional random seed for judge ordering")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mentor_dir = Path(args.mentor_dir).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()
    if not mentor_dir.exists() or not mentor_dir.is_dir():
        raise FileNotFoundError(f"Mentor directory not found: {mentor_dir}")
    if not baseline_dir.exists() or not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    scenarios = args.prompt_ids or _paired_scenarios(mentor_dir, baseline_dir)
    if not scenarios:
        raise RuntimeError("No overlapping scenarios found to judge")

    print_info(
        f"Pairwise judging {len(scenarios)} scenario(s) with judges: {args.judges} and repeats={args.repeats}"
    )
    run_pairwise_judges(
        scenarios,
        mentor_dir,
        baseline_dir,
        judge_specs=args.judges,
        label=args.label,
        repeats=max(1, args.repeats),
        seed=args.seed,
    )
    print_success(f"Completed pairwise judging for label={args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
