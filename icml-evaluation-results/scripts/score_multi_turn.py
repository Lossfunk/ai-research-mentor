"""Score multi-turn mentor conversations with the student judge rubric."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.rich_formatter import print_error, print_info, print_success

from .judge_utils import build_judge_clients, truncate_text
from .run_student_judge_scores import (
    _aggregate_student,
    _call_student_judge,
    _parse_student_json,
    _render_student_prompt,
)


DEFAULT_INPUT_ROOT = Path("reports/multi_turn_eval_all5")
DEFAULT_TEMPLATE = Path("evaluation/judges/student_outcome_judge.md")
# Threshold anchored to rubric: 1.5 = "Good" (clear, specific, actionable guidance)
# Scores above 1.5 indicate the mentor consistently provides guidance that students
# can execute with their available resources. See evaluation/judges/student_outcome_judge.md
DEFAULT_THRESHOLD = 1.5
DEFAULT_STAGE = "C"
DEFAULT_JUDGES: Tuple[str, ...] = (
    "openrouter:qwen/qwen3-max",  # Alibaba - independent from evaluated systems
    "openrouter:deepseek/deepseek-v3.2-exp",  # DeepSeek - independent
    "openrouter:x-ai/grok-4-fast",  # xAI - independent
    # Note: Deliberately excluding Google/Anthropic/OpenAI judges to avoid
    # family bias when evaluating Gemini/Claude/GPT systems
)

# Stop reason classification for success determination
# Based on empirical analysis of multi-turn conversation outcomes
POSITIVE_STOP_REASONS = frozenset({
    "goal_reached",      # Student explicitly got what they needed
    "mentored_enough",   # Student satisfied, has concrete plan to proceed
})

NEGATIVE_STOP_REASONS = frozenset({
    "blocked",           # Student gave up due to mentor failures or inability to help
    "not_helpful",       # Mentor responses weren't useful
    "error",             # Technical error during conversation
    "invalid_student_json",  # Student model failed to produce valid output
    "empty_mentor_reply",    # Mentor returned nothing
})

AMBIGUOUS_STOP_REASONS = frozenset({
    "student_terminated",    # Fallback - no explicit reason given
    "max_turns_reached",     # Conversation hit limit, may or may not have succeeded
    "time_constraint",       # Student had to leave, not necessarily satisfied
})


def classify_stop_reason(stop_reason: str) -> str:
    """Classify a stop_reason into positive/negative/ambiguous."""
    reason = stop_reason.lower().strip() if stop_reason else ""
    if reason in POSITIVE_STOP_REASONS or reason in {r.lower() for r in POSITIVE_STOP_REASONS}:
        return "positive"
    if reason in NEGATIVE_STOP_REASONS or reason in {r.lower() for r in NEGATIVE_STOP_REASONS}:
        return "negative"
    return "ambiguous"


def compute_success_with_stop_reason(
    final_score: Optional[float],
    stop_reason: str,
    threshold: float,
    ambiguous_threshold_bump: float = 0.1,
) -> Tuple[bool, str]:
    """
    Determine success incorporating both score and stop_reason semantics.

    Returns (is_success, success_type) where success_type is one of:
    - "score_and_positive_stop": High score + positive stop reason
    - "score_only": High score but ambiguous/missing stop reason
    - "positive_stop_only": Positive stop reason but score below threshold
    - "failed_negative_stop": Negative stop reason (automatic failure)
    - "failed_low_score": Score below threshold with ambiguous stop
    - "failed_no_data": Missing score data
    """
    classification = classify_stop_reason(stop_reason)

    # Negative stops are automatic failures regardless of score
    if classification == "negative":
        return False, "failed_negative_stop"

    if final_score is None:
        return False, "failed_no_data"

    # Positive stops: use standard threshold
    if classification == "positive":
        if final_score >= threshold:
            return True, "score_and_positive_stop"
        # Positive stop but low score - still count as success but flag it
        # Student said they were satisfied, trust them
        return True, "positive_stop_only"

    # Ambiguous stops: require higher threshold since we're less certain
    adjusted_threshold = threshold + ambiguous_threshold_bump
    if final_score >= adjusted_threshold:
        return True, "score_only"

    return False, "failed_low_score"


@dataclass
class ConversationTurn:
    agent_label: str
    system_id: str
    scenario_id: str
    turn_index: int
    mentor_reply: str
    student_message: str
    scores: Dict[str, Optional[float]]
    overall_score: Optional[float]
    cumulative_avg: Optional[float]
    delta: Optional[float]
    success_at_turn: bool
    judge_outputs: List[Dict[str, Any]]


@dataclass
class ConversationSummary:
    agent_label: str
    system_id: str
    scenario_id: str
    total_turns: int
    elapsed_seconds: float
    final_score: Optional[float]
    net_gain: Optional[float]
    success_turn: Optional[int]
    success_elapsed_seconds: Optional[float]
    positive_delta_share: Optional[float]
    stop_reason: str
    stopped_by_student: bool
    # New fields for stop_reason-aware success
    stop_reason_class: str = ""  # "positive", "negative", "ambiguous"
    is_success: bool = False
    success_type: str = ""  # detailed success/failure classification


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multi-turn mentor transcripts with student judge rubric")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing multi-turn runs (default: reports/multi_turn_eval_all5)",
    )
    parser.add_argument(
        "--judge",
        dest="judges",
        action="append",
        help="Judge LLM spec provider:model (repeatable). Defaults to the 3-model ensemble (Gemini-2.5-flash, DeepSeek v3.2-exp, Grok-4-fast)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Student judge prompt template",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Success threshold for overall score (default 1.6)",
    )
    parser.add_argument(
        "--stage",
        default=DEFAULT_STAGE,
        help="Stage code to inject into judge prompt (default C)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory; defaults to <input-root>/scoring",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score conversations even if cached scores exist",
    )
    return parser.parse_args(argv)


def iter_transcript_files(root: Path) -> Iterable[Tuple[str, Path]]:
    """
    Iterate over transcript files in either of two directory structures:

    Structure A (nested runs):
        root/run_name/transcripts/system_id/*.json
        -> yields (run_name, path)

    Structure B (flat, single run):
        root/transcripts/system_id/*.json
        -> yields (root.name, path)
    """
    # Check for Structure B: flat transcripts dir directly under root
    flat_transcripts = root / "transcripts"
    if flat_transcripts.exists() and flat_transcripts.is_dir():
        # Check if it contains system directories (not JSON files directly)
        subdirs = [p for p in flat_transcripts.iterdir() if p.is_dir()]
        if subdirs:
            # Structure B: flat layout
            run_label = root.name
            for system_dir in sorted(subdirs):
                for json_file in sorted(system_dir.glob("*.json")):
                    yield run_label, json_file
            return  # Don't also check for Structure A

    # Structure A: nested runs
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        transcripts_dir = run_dir / "transcripts"
        if not transcripts_dir.exists():
            continue
        for system_dir in sorted(p for p in transcripts_dir.iterdir() if p.is_dir()):
            for json_file in sorted(system_dir.glob("*.json")):
                yield run_dir.name, json_file


def load_transcript(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_persona_card(scenario: Dict[str, Any]) -> str:
    persona = scenario.get("persona") or "Research student"
    constraints = scenario.get("constraints") or "(no explicit constraints)"
    topic = scenario.get("topic") or "general research"
    lines = [str(persona).strip()]
    if constraints:
        lines.append(f"constraints: {constraints}")
    lines.append(f"topic_focus: {topic}")
    return "\n".join(lines)


def build_task_card(initial_user_message: str, latest_user_message: str, scenario: Dict[str, Any]) -> str:
    topic = scenario.get("topic") or "general research"
    initial_block = initial_user_message.strip()
    latest_block = latest_user_message.strip()
    if not latest_block:
        latest_block = initial_block
    content = [f"Research topic: {topic}."]
    content.append("Initial question from student:\n" + initial_block)
    if latest_block and latest_block != initial_block:
        content.append("Latest student follow-up:\n" + latest_block)
    return "\n\n".join(content)


def score_turn(
    mentor_reply: str,
    persona_card: str,
    task_card: str,
    stage: str,
    judge_template: str,
    judge_clients: Sequence[Tuple[str, Any]],
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Any]]]:
    system_msg, user_msg = _render_student_prompt(
        judge_template,
        persona_card,
        task_card,
        stage,
        truncate_text(mentor_reply, limit=12000),
    )

    judge_outputs: List[Dict[str, Any]] = []
    for name, client in judge_clients:
        try:
            raw = _call_student_judge(client, system_msg, user_msg)
            parsed = _parse_student_json(raw) or {}
            judge_outputs.append({"judge": name, "raw": raw, "parsed": parsed})
        except Exception as exc:  # noqa: BLE001
            judge_outputs.append({"judge": name, "error": str(exc)})

    parsed_payloads = [out.get("parsed") or {} for out in judge_outputs if isinstance(out.get("parsed"), dict)]
    aggregated = _aggregate_student(parsed_payloads)
    return aggregated, judge_outputs


def ensure_output_dirs(base: Path) -> Tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    scores_dir = base / "scores"
    plots_dir = base / "plots"
    scores_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    return scores_dir, plots_dir


def process_conversation(
    run_label: str,
    path: Path,
    judge_template: str,
    judge_clients: Sequence[Tuple[str, Any]],
    stage: str,
    threshold: float,
) -> Tuple[List[ConversationTurn], ConversationSummary]:
    payload = load_transcript(path)
    scenario = payload.get("scenario") or {}
    persona_card = build_persona_card(scenario)
    transcript = payload.get("transcript") or []
    if not transcript:
        raise ValueError(f"Transcript empty: {path}")

    initial_user = next((entry for entry in transcript if entry.get("role") == "user"), None)
    initial_message = initial_user.get("content", "") if initial_user else ""

    system_id = str(payload.get("system_id") or "unknown")
    scenario_id = str(scenario.get("scenario_id") or path.stem)
    elapsed_seconds = float(payload.get("elapsed_seconds") or 0.0)
    stop_reason = str(payload.get("stop_reason") or "")
    stopped_by_student = bool(payload.get("stopped_by_student"))

    convo_pairs: List[Tuple[Dict[str, Any], str]] = []
    latest_student_message = initial_message
    for entry in transcript:
        role = entry.get("role")
        if role == "user":
            latest_student_message = str(entry.get("content") or "")
        elif role == "assistant":
            convo_pairs.append((entry, latest_student_message))

    if not convo_pairs:
        raise ValueError(f"No assistant turns found in {path}")

    turn_records: List[ConversationTurn] = []
    overall_scores: List[float] = []
    cumulative_avg: List[float] = []
    deltas: List[float] = []
    success_turn: Optional[int] = None

    total_turns = len(convo_pairs)
    for idx, (turn, student_message) in enumerate(convo_pairs, start=1):
        print_info(f"  Turn {idx}/{total_turns} - calling {len(judge_clients)} judges...")
        mentor_reply = str(turn.get("content") or "").strip()
        task_card = build_task_card(initial_message, student_message, scenario)
        scores, judge_outputs = score_turn(
            mentor_reply,
            persona_card,
            task_card,
            stage,
            judge_template,
            judge_clients,
        )
        overall = scores.get("student_outcome_score")
        print_info(f"  Turn {idx}/{total_turns} - score: {overall}")
        if overall is not None:
            overall_scores.append(float(overall))
            cumulative_avg.append(sum(overall_scores) / len(overall_scores))
        else:
            cumulative_avg.append(cumulative_avg[-1] if cumulative_avg else None)

        if len(overall_scores) >= 2:
            deltas.append(overall_scores[-1] - overall_scores[-2])
        else:
            deltas.append(None)

        if success_turn is None and overall is not None and overall >= threshold:
            success_turn = idx

        turn_records.append(
            ConversationTurn(
                agent_label=run_label,
                system_id=system_id,
                scenario_id=scenario_id,
                turn_index=idx,
                mentor_reply=mentor_reply,
                student_message=student_message,
                scores=scores,
                overall_score=overall,
                cumulative_avg=cumulative_avg[-1],
                delta=deltas[-1],
                success_at_turn=success_turn == idx,
                judge_outputs=judge_outputs,
            )
        )

    total_turns = len(convo_pairs)
    final_score = overall_scores[-1] if overall_scores else None
    net_gain = None
    if len(overall_scores) >= 2:
        net_gain = overall_scores[-1] - overall_scores[0]

    positive_delta_share: Optional[float] = None
    real_deltas = [d for d in deltas[1:] if d is not None]
    if real_deltas:
        positive_delta_share = sum(1 for d in real_deltas if d > 0) / len(real_deltas)

    success_elapsed = None
    if success_turn is not None and total_turns > 0 and elapsed_seconds:
        success_elapsed = elapsed_seconds * (success_turn / total_turns)

    # Compute stop_reason-aware success
    stop_reason_class = classify_stop_reason(stop_reason)
    is_success, success_type = compute_success_with_stop_reason(
        final_score, stop_reason, threshold
    )

    summary = ConversationSummary(
        agent_label=run_label,
        system_id=system_id,
        scenario_id=scenario_id,
        total_turns=total_turns,
        elapsed_seconds=elapsed_seconds,
        final_score=final_score,
        net_gain=net_gain,
        success_turn=success_turn,
        success_elapsed_seconds=success_elapsed,
        positive_delta_share=positive_delta_share,
        stop_reason=stop_reason,
        stopped_by_student=stopped_by_student,
        stop_reason_class=stop_reason_class,
        is_success=is_success,
        success_type=success_type,
    )

    return turn_records, summary


def write_conversation_scores(scores_dir: Path, turns: List[ConversationTurn]) -> None:
    if not turns:
        return
    sample = turns[0]
    out_path = scores_dir / f"{sample.agent_label}__{sample.scenario_id}.json"
    data = {
        "agent_label": sample.agent_label,
        "system_id": sample.system_id,
        "scenario_id": sample.scenario_id,
        "turns": [
            {
                "turn_index": t.turn_index,
                "mentor_reply": t.mentor_reply,
                "student_message": t.student_message,
                "scores": t.scores,
                "overall_score": t.overall_score,
                "cumulative_avg": t.cumulative_avg,
                "delta": t.delta,
                "success_at_turn": t.success_at_turn,
                "judge_outputs": t.judge_outputs,
            }
            for t in turns
        ],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, summaries: List[ConversationSummary]) -> None:
    fieldnames = [
        "agent_label",
        "system_id",
        "scenario_id",
        "total_turns",
        "elapsed_seconds",
        "final_score",
        "net_gain",
        "success_turn",
        "success_elapsed_seconds",
        "positive_delta_share",
        "stop_reason",
        "stop_reason_class",
        "stopped_by_student",
        "is_success",
        "success_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "agent_label": row.agent_label,
                "system_id": row.system_id,
                "scenario_id": row.scenario_id,
                "total_turns": row.total_turns,
                "elapsed_seconds": row.elapsed_seconds,
                "final_score": row.final_score,
                "net_gain": row.net_gain,
                "success_turn": row.success_turn,
                "success_elapsed_seconds": row.success_elapsed_seconds,
                "positive_delta_share": row.positive_delta_share,
                "stop_reason": row.stop_reason,
                "stop_reason_class": row.stop_reason_class,
                "stopped_by_student": row.stopped_by_student,
                "is_success": row.is_success,
                "success_type": row.success_type,
            })


def write_agent_summary(path: Path, summaries: List[ConversationSummary]) -> None:
    agg: Dict[str, Dict[str, List[float]]] = {}
    for summary in summaries:
        key = summary.agent_label
        bucket = agg.setdefault(key, {
            "final_scores": [],
            "net_gains": [],
            "turns": [],
            "success_flags": [],
            "success_flags_new": [],  # Using is_success (stop_reason-aware)
            "success_turns": [],
            "positive_stops": [],
            "negative_stops": [],
        })
        if summary.final_score is not None:
            bucket["final_scores"].append(summary.final_score)
        if summary.net_gain is not None:
            bucket["net_gains"].append(summary.net_gain)
        bucket["turns"].append(float(summary.total_turns))
        # Old success metric (score threshold only)
        bucket["success_flags"].append(1.0 if summary.success_turn is not None else 0.0)
        # New success metric (stop_reason-aware)
        bucket["success_flags_new"].append(1.0 if summary.is_success else 0.0)
        if summary.success_turn is not None:
            bucket["success_turns"].append(float(summary.success_turn))
        # Track stop reason distribution
        bucket["positive_stops"].append(1.0 if summary.stop_reason_class == "positive" else 0.0)
        bucket["negative_stops"].append(1.0 if summary.stop_reason_class == "negative" else 0.0)

    fieldnames = [
        "agent_label",
        "conversations",
        "avg_final_score",
        "avg_net_gain",
        "avg_turns",
        "success_rate_score_only",
        "success_rate_with_stop_reason",
        "positive_stop_rate",
        "negative_stop_rate",
        "avg_success_turn",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for agent, stats in sorted(agg.items()):
            conversations = len(stats["turns"])
            writer.writerow({
                "agent_label": agent,
                "conversations": conversations,
                "avg_final_score": mean(stats["final_scores"]) if stats["final_scores"] else None,
                "avg_net_gain": mean(stats["net_gains"]) if stats["net_gains"] else None,
                "avg_turns": mean(stats["turns"]) if stats["turns"] else None,
                "success_rate_score_only": (sum(stats["success_flags"]) / conversations) if conversations else None,
                "success_rate_with_stop_reason": (sum(stats["success_flags_new"]) / conversations) if conversations else None,
                "positive_stop_rate": (sum(stats["positive_stops"]) / conversations) if conversations else None,
                "negative_stop_rate": (sum(stats["negative_stops"]) / conversations) if conversations else None,
                "avg_success_turn": mean(stats["success_turns"]) if stats["success_turns"] else None,
            })


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    load_env_file()

    judge_specs = args.judges or list(DEFAULT_JUDGES)
    judge_clients = build_judge_clients(judge_specs)
    judge_models = ", ".join(spec for spec, _ in judge_clients)
    judge_template = args.template.read_text(encoding="utf-8")

    output_dir = args.output_dir or (args.input_root / "scoring")
    scores_dir, plots_dir = ensure_output_dirs(output_dir)
    _ = plots_dir  # reserved for downstream plotting

    turn_records: List[ConversationTurn] = []
    summaries: List[ConversationSummary] = []

    for run_label, transcript_path in iter_transcript_files(args.input_root):
        print_info(f"Scoring {transcript_path.name} (run={run_label})")
        try:
            turns, summary = process_conversation(
                run_label,
                transcript_path,
                judge_template,
                judge_clients,
                args.stage,
                args.threshold,
            )
        except Exception as exc:  # noqa: BLE001
            print_error(f"Failed to score {transcript_path}: {exc}")
            continue

        write_conversation_scores(scores_dir, turns)
        turn_records.extend(turns)
        summaries.append(summary)
        print_success(
            f"Scored {transcript_path.name} ({run_label}) with {len(turns)} turns — final score {summary.final_score!r}"
        )

    if not summaries:
        print_error("No conversations processed")
        return 1

    write_summary_csv(output_dir / "summary_conversations.csv", summaries)
    write_agent_summary(output_dir / "summary_by_agent.csv", summaries)
    print_success(
        f"Wrote {len(summaries)} conversation summaries using judges [{judge_models}] (threshold={args.threshold})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
