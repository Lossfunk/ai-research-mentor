#!/usr/bin/env python3
"""
Batch runner for single-turn evaluation with automated analysis.

Usage:
  uv run python -m evaluation.scripts.run_single_turn_batch \
      --input evaluation/data/evals_single_turn.jsonl \
      --systems openai/gpt-4o claude/sonnet-4 \
      --judge claude/sonnet-4 \
      --output-label quick_test
"""

import argparse
import os
import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.rich_formatter import print_error, print_info, print_success

from .config_loader import compute_file_digest, load_metrics_config
from .judge_utils import build_judge_clients
from .run_judge_scores import run_judges
from .run_manual_stage import ensure_stage_directories, normalize_stage
from .single_turn_orchestrator import SingleTurnOrchestrator

try:
    # Attachments API (optional; runner works without attachments)
    from academic_research_mentor.attachments import attach_pdfs as _attach_pdfs
except Exception:  # pragma: no cover - optional import
    _attach_pdfs = None  # type: ignore


PAIRWISE_PROMPT_PATH = Path("evaluation/judges/pairwise_judge_prompt.md")


def load_test_data(data_path: Path) -> List[Dict]:
    """Load single-turn test prompts from JSONL file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")
    
    prompts = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def run_system_responses(
    prompts: List[Dict[str, Any]],
    orchestrator: SingleTurnOrchestrator,
    *,
    stage_letter: str,
    raw_dir: Path,
    analysis_dir: Path,
) -> Dict[str, List[str]]:
    """Generate responses from all target systems for each prompt."""

    system_results: Dict[str, List[str]] = {system: [] for system in orchestrator.systems_to_test}
    total_prompts = len(prompts)

    print_info(
        f"Generating responses for {total_prompts} prompts across {len(orchestrator.systems_to_test)} systems..."
    )
    
    for index, prompt_data in enumerate(prompts, start=1):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["prompt"]
        metadata = dict(prompt_data.get("metadata") or {})
        expected_tools = list(metadata.get("expected_tools", []))
        expected_checks = list(prompt_data.get("expected_checks", []))

        print_info(f"[{index}/{total_prompts}] Processing {prompt_id}...")

        system_responses = orchestrator.process_single_prompt(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            expected_tools=expected_tools,
            expected_checks=expected_checks,
            stage=stage_letter,
            metadata=metadata,
        )

        for system_id, response_data in system_responses.items():
            meta_payload = dict(response_data.get("meta") or {})
            alias = meta_payload.get("system_alias") or system_id.replace("/", "_")

            response_success = response_data.get("success", False)
            response_text = response_data.get("response", "")

            if response_success and response_text:
                response_path = raw_dir / f"{prompt_id}_{alias}.txt"
                response_path.write_text(response_text, encoding="utf-8")

                tool_trace = response_data.get("tool_trace") or []
                tool_trace_path: Optional[Path] = None
                if tool_trace:
                    tool_trace_path = raw_dir / f"{prompt_id}_{alias}_tools.json"
                    tool_trace_path.write_text(
                        json.dumps(tool_trace, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                meta_payload.update(
                    {
                        "prompt_id": prompt_id,
                        "stage": stage_letter,
                        "prompt": prompt_text,
                        "expected_checks": expected_checks,
                        "metadata": metadata,
                        "system": system_id,
                        "response_path": str(response_path),
                        "tool_trace_path": str(tool_trace_path) if tool_trace_path else None,
                        "run_timestamp": response_data.get("timestamp") or meta_payload.get("generated_at"),
                        "elapsed_seconds": response_data.get("elapsed"),
                        "tool_runs_count": len(tool_trace),
                        "success": True,
                        "error": None,
                    }
                )

                meta_path = analysis_dir / f"{prompt_id}_{alias}_meta.json"
                meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                system_results.setdefault(system_id, []).append(prompt_id)
                print_success(f"  ✓ {system_id}: Response saved")
            else:
                print_error(f"  ✗ {system_id}: Generation failed ({response_data.get('error') or 'unknown error'})")
    
    return system_results


def load_meta_index(analysis_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    meta_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for meta_file in analysis_dir.glob("*_meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print_error(f"Failed to load meta file {meta_file}: {exc}")
            continue
        prompt_id = meta.get("prompt_id")
        system_id = meta.get("system_id") or meta.get("system")
        if not prompt_id or not system_id:
            continue
        meta_index.setdefault(prompt_id, {})[system_id] = meta
    return meta_index


def parse_pairwise_output(raw: str) -> Optional[Dict[str, Any]]:
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1]
        candidate = candidate.strip("`\n ")
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    winner = parsed.get("winner")
    if winner not in {"A", "B", "Tie"}:
        parsed["winner"] = "Tie"
    aspect_votes = parsed.get("aspect_votes")
    if not isinstance(aspect_votes, dict):
        parsed["aspect_votes"] = {}
    return parsed


def run_pairwise_comparisons(
    meta_index: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    systems: Sequence[str],
    judge_specs: Sequence[str],
    analysis_dir: Path,
    output_label: str,
    random_seed: Optional[int] = None,
) -> None:
    if not judge_specs:
        return

    if not meta_index:
        print_error("Pairwise comparisons skipped: no metadata found")
        return

    if not PAIRWISE_PROMPT_PATH.exists():
        print_error(f"Pairwise judge prompt missing: {PAIRWISE_PROMPT_PATH}")
        return

    try:
        pairwise_template = PAIRWISE_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print_error(f"Unable to read pairwise judge prompt: {exc}")
        return

    judge_clients = build_judge_clients(judge_specs)
    if not judge_clients:
        print_error("No pairwise judge clients could be initialized")
        return

    metrics_config = load_metrics_config()
    pairwise_aspects = (metrics_config.get("pairwise", {}) or {}).get("aspects", {})
    pairwise_digest = compute_file_digest(PAIRWISE_PROMPT_PATH)

    rng = random.Random(random_seed or 0)
    overall_wins: Counter[str] = Counter()
    pair_stats: Dict[tuple[str, str], Dict[str, Any]] = {}

    pairwise_root = analysis_dir / output_label / "pairwise"
    pairwise_root.mkdir(parents=True, exist_ok=True)

    for prompt_id in sorted(meta_index.keys()):
        system_map = meta_index[prompt_id]
        available_systems = [system for system in systems if system in system_map]
        for sys_a, sys_b in itertools.combinations(available_systems, 2):
            order_map = {"A": sys_a, "B": sys_b}
            if rng.random() < 0.5:
                order_map = {"A": sys_b, "B": sys_a}

            try:
                meta_a = system_map[order_map["A"]]
                meta_b = system_map[order_map["B"]]
            except KeyError:
                continue

            response_a = Path(meta_a.get("response_path", ""))
            response_b = Path(meta_b.get("response_path", ""))
            if not response_a.exists() or not response_b.exists():
                print_error(f"Missing response files for pairwise comparison on {prompt_id}")
                continue

            try:
                text_a = response_a.read_text(encoding="utf-8")
                text_b = response_b.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                print_error(f"Failed reading responses for {prompt_id}: {exc}")
                continue

            persona_blob = meta_a.get("metadata") or {}
            human_sections = [
                "### Persona & Metadata",
                json.dumps(persona_blob, ensure_ascii=False, indent=2),
                "\n### User Prompt",
                meta_a.get("prompt", ""),
                "\n### System A Response",
                text_a,
                "\n### System B Response",
                text_b,
            ]
            human_message = "\n".join(human_sections)

            judge_outputs: List[Dict[str, Any]] = []
            for name, client in judge_clients:
                try:
                    result = client.invoke(
                        [
                            SystemMessage(content=pairwise_template),
                            HumanMessage(content=human_message),
                        ]
                    )
                    raw = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
                except Exception as exc:  # noqa: BLE001
                    raw = json.dumps({"winner": "Tie", "error": str(exc)})
                parsed = parse_pairwise_output(raw) or {"winner": "Tie"}
                judge_outputs.append({"judge": name, "raw": raw, "parsed": parsed})

            vote_counts = Counter()
            for output in judge_outputs:
                winner = (output.get("parsed") or {}).get("winner", "Tie")
                if winner in {"A", "B", "Tie"}:
                    vote_counts[winner] += 1

            if vote_counts:
                majority_vote, majority_count = vote_counts.most_common(1)[0]
                if majority_count == vote_counts["Tie"] and majority_vote != "Tie":
                    majority_vote = "Tie"
            else:
                majority_vote = "Tie"

            winning_system: Optional[str]
            if majority_vote == "A":
                winning_system = order_map["A"]
            elif majority_vote == "B":
                winning_system = order_map["B"]
            else:
                winning_system = None

            pair_key = tuple(sorted((sys_a, sys_b)))
            pair_entry = pair_stats.setdefault(
                pair_key,
                {
                    "comparisons": [],
                    "wins": Counter(),
                    "ties": 0,
                },
            )

            if winning_system:
                overall_wins[winning_system] += 1
                pair_entry["wins"][winning_system] += 1
            else:
                pair_entry["ties"] += 1

            safe_pair = tuple(s.replace("/", "_") for s in pair_key)
            pair_dir = pairwise_root / f"{safe_pair[0]}__vs__{safe_pair[1]}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            comparison_payload = {
                "prompt_id": prompt_id,
                "order": order_map,
                "winner": majority_vote,
                "winner_system_id": winning_system,
                "judge_outputs": judge_outputs,
                "pairwise_prompt_digest": pairwise_digest,
            }
            comparison_path = pair_dir / f"{prompt_id}.json"
            comparison_path.write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            pair_entry["comparisons"].append(
                {
                    "prompt_id": prompt_id,
                    "winner": majority_vote,
                    "winner_system_id": winning_system,
                }
            )

    for pair_key, stats in pair_stats.items():
        safe_pair = tuple(s.replace("/", "_") for s in pair_key)
        pair_dir = pairwise_root / f"{safe_pair[0]}__vs__{safe_pair[1]}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "systems": list(pair_key),
            "total_comparisons": len(stats["comparisons"]),
            "wins": {k: v for k, v in stats["wins"].items()},
            "ties": stats["ties"],
            "judges": list(judge_specs),
            "pairwise_prompt_digest": pairwise_digest,
            "pairwise_aspects": pairwise_aspects,
        }
        summary_path = pair_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if pair_stats:
        overall_path = pairwise_root / "overall_summary.json"
        overall_payload = {
            "wins": {k: v for k, v in overall_wins.items()},
            "judges": list(judge_specs),
            "pairwise_prompt_digest": pairwise_digest,
            "pairwise_aspects": pairwise_aspects,
            "seed": random_seed,
        }
        overall_path.write_text(json.dumps(overall_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print_success(
            f"Pairwise comparisons completed across {len(pair_stats)} system pair(s); summary -> {overall_path}"
        )
    else:
        print_info("No eligible system pairs found for pairwise comparisons")


def generate_summary_report(
    system_results: Dict[str, List[str]], 
    total_prompts: int,
    stage: str,
    output_label: str,
    output_dir: Path
) -> None:
    """Generate summary report for the batch run."""
    summary_dir = output_dir / output_label
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_type": "single_turn_batch",
        "stage": stage,
        "output_label": output_label,
        "total_prompts": total_prompts,
        "systems": {
            system: {
                "processed_count": len(prompt_ids),
                "success_rate": len(prompt_ids) / total_prompts,
                "prompt_ids": prompt_ids
            }
            for system, prompt_ids in system_results.items()
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    summary_path = summary_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_success(f"Batch summary saved: {summary_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Batch runner for single-turn evaluation")
    parser.add_argument("--input", required=True, help="Path to single-turn JSONL test data")
    parser.add_argument("--systems", nargs="+", required=True, help="Systems to test (provider/model format)")
    parser.add_argument("--stage", default="stage_a", help="Stage to evaluate (stage_a, stage_b, stage_c)")
    parser.add_argument("--judge", required=True, help="Judge model for scoring")
    parser.add_argument("--output-label", default="batch_run", help="Label for organizing outputs")
    parser.add_argument("--skip-generation", action="store_true", help="Skip system response generation, only run scoring")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM generation temperature")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum tokens per response")
    parser.add_argument("--seed", type=int, help="Random seed for deterministic runs")
    parser.add_argument(
        "--pairwise-judge",
        dest="pairwise_judges",
        action="append",
        help="Optional judge spec for pairwise preference scoring (repeatable)",
    )
    parser.add_argument(
        "--attach-pdf",
        dest="attach_pdfs",
        action="append",
        help="Attach one or more PDF files to ground answers (repeatable)",
    )
    parser.add_argument(
        "--attachments-dir",
        help="Directory containing PDFs to attach (all *.pdf files will be loaded)",
    )
    args = parser.parse_args(argv)
    
    # Setup directories
    stage_letter, stage_folder = normalize_stage(args.stage)
    raw_dir, analysis_dir, _ = ensure_stage_directories(stage_folder)

    # Load attachments, if provided
    try:
        pdfs: List[str] = []
        if getattr(args, "attach_pdfs", None):
            pdfs.extend([str(Path(p)) for p in args.attach_pdfs if p])
        if getattr(args, "attachments_dir", None):
            base = Path(args.attachments_dir)
            if base.exists() and base.is_dir():
                for p in base.glob("*.pdf"):
                    pdfs.append(str(p))
        if pdfs and _attach_pdfs is not None:
            print_info(f"Attaching PDFs ({len(pdfs)}): first={os.path.basename(pdfs[0])}")
            _attach_pdfs(pdfs)
        elif pdfs and _attach_pdfs is None:
            print_error("Attachments support not available; skipping PDF attach")
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to attach PDFs: {exc}")
    
    # Load test data
    prompts = load_test_data(Path(args.input))
    print_info(f"Loaded {len(prompts)} test prompts from {args.input}")
    
    system_results: Dict[str, List[str]] = {system: [] for system in args.systems}
    orchestrator: Optional[SingleTurnOrchestrator] = None

    # Phase 1: Generate system responses (unless skipped)
    if not args.skip_generation:
        orchestrator = SingleTurnOrchestrator(
            systems_to_test=args.systems,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
        )
        system_results = run_system_responses(
            prompts,
            orchestrator,
            stage_letter=stage_letter,
            raw_dir=raw_dir,
            analysis_dir=analysis_dir,
        )
        generate_summary_report(system_results, len(prompts), args.stage, args.output_label, analysis_dir)
    else:
        print_info("Skipping response generation (--skip-generation)")
    
    # Phase 2: Run judge scoring
    print_info(f"Running judge scoring with model: {args.judge}")
    
    # Find all meta files for the stage (including from multiple systems)
    meta_files = list(analysis_dir.glob("*_meta.json"))

    if not meta_files:
        print_error("No meta files found. Run with --skip-generation=false first.")
        return 1

    print_info(f"Found {len(meta_files)} meta files for scoring")
    
    # Ensure root output directory exists
    (analysis_dir / args.output_label).mkdir(parents=True, exist_ok=True)

    # Score each system separately
    for system in args.systems:
        system_safe = system.replace("/", "_")
        print_info(f"Scoring {system}...")

        try:
            summary = run_judges(
                stage=args.stage,
                prompt_ids=None,
                judge_specs=[args.judge],
                annotator=f"batch_{system_safe}",
                force=False,
                output_label=f"{args.output_label}/absolute/{system_safe}",
                system_filter=system,
            )
            print_success(
                f"Completed scoring for {system} — processed {summary.get('processed', 0)} prompt(s)"
            )
        except Exception as exc:
            print_error(f"Scoring failed for {system}: {exc}")
    
    if args.pairwise_judges:
        meta_index = load_meta_index(analysis_dir)
        run_pairwise_comparisons(
            meta_index,
            systems=args.systems,
            judge_specs=args.pairwise_judges,
            analysis_dir=analysis_dir,
            output_label=args.output_label,
            random_seed=args.seed,
        )
        print_info(f"Pairwise outputs stored under: {analysis_dir / args.output_label / 'pairwise'}")

    print_success("Batch evaluation completed!")
    print_info(f"Results available in: {analysis_dir / args.output_label}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
