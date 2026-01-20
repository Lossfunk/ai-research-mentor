"""Score multi-turn mentor conversations holistically - one score per conversation."""

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
from .run_student_judge_scores import _call_student_judge, _parse_student_json


DEFAULT_INPUT_ROOT = Path("reports/multi_turn_eval")
DEFAULT_TEMPLATE = Path("evaluation/judges/holistic_conversation_judge.md")
DEFAULT_THRESHOLD = 1.5
DEFAULT_JUDGES: Tuple[str, ...] = (
    "openrouter:qwen/qwen3-max",
    "openrouter:deepseek/deepseek-v3.2-exp",
    "openrouter:x-ai/grok-4-fast",
)

# Stop reason classification (same as score_multi_turn.py)
POSITIVE_STOP_REASONS = frozenset({"goal_reached", "mentored_enough"})
NEGATIVE_STOP_REASONS = frozenset({"blocked", "not_helpful", "error", "invalid_student_json", "empty_mentor_reply"})


def classify_stop_reason(stop_reason: str) -> str:
    """Classify a stop_reason into positive/negative/ambiguous."""
    reason = stop_reason.lower().strip() if stop_reason else ""
    if reason in POSITIVE_STOP_REASONS or reason in {r.lower() for r in POSITIVE_STOP_REASONS}:
        return "positive"
    if reason in NEGATIVE_STOP_REASONS or reason in {r.lower() for r in NEGATIVE_STOP_REASONS}:
        return "negative"
    return "ambiguous"


@dataclass
class HolisticResult:
    """Result of holistic scoring for one conversation."""
    agent_label: str
    system_id: str
    scenario_id: str
    total_turns: int
    elapsed_seconds: float
    stop_reason: str
    stop_reason_class: str
    stopped_by_student: bool
    # Holistic scores
    overall_helpfulness: Optional[float]
    student_progress: Optional[float]
    mentor_effectiveness: Optional[float]
    conversation_efficiency: Optional[float]
    holistic_score: Optional[float]  # Composite
    # Success determination
    is_success: bool
    success_type: str
    # Weaknesses identified by judges
    weaknesses: List[str]
    # Raw judge outputs for debugging
    judge_outputs: List[Dict[str, Any]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multi-turn conversations holistically")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing transcripts",
    )
    parser.add_argument(
        "--judge",
        dest="judges",
        action="append",
        help="Judge LLM spec (repeatable). Defaults to Qwen3-max, DeepSeek, Grok ensemble",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Holistic judge prompt template",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Success threshold (default 1.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <input-root>/holistic_scoring)",
    )
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=60000,
        help="Max characters for transcript (truncate middle if exceeded)",
    )
    return parser.parse_args(argv)


def iter_transcript_files(root: Path) -> Iterable[Tuple[str, Path]]:
    """Iterate over transcript files, supporting both flat and nested structures."""
    flat_transcripts = root / "transcripts"
    if flat_transcripts.exists() and flat_transcripts.is_dir():
        subdirs = [p for p in flat_transcripts.iterdir() if p.is_dir()]
        if subdirs:
            run_label = root.name
            for system_dir in sorted(subdirs):
                for json_file in sorted(system_dir.glob("*.json")):
                    yield run_label, json_file
            return

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        transcripts_dir = run_dir / "transcripts"
        if not transcripts_dir.exists():
            continue
        for system_dir in sorted(p for p in transcripts_dir.iterdir() if p.is_dir()):
            for json_file in sorted(system_dir.glob("*.json")):
                yield run_dir.name, json_file


def load_transcript(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_transcript_for_judge(transcript: List[Dict[str, Any]], max_chars: int = 60000) -> str:
    """Format transcript as readable conversation for the judge."""
    lines = []
    for entry in transcript:
        role = entry.get("role", "unknown")
        content = entry.get("content", "").strip()

        # Clean up thinking tags for readability
        if "<thinking>" in content and "</thinking>" in content:
            # Keep thinking but mark it
            content = content.replace("<thinking>", "[MENTOR THINKING: ").replace("</thinking>", "]")

        role_label = "STUDENT" if role == "user" else "MENTOR"
        lines.append(f"**{role_label}:**\n{content}\n")

    full_text = "\n---\n".join(lines)

    # Truncate from middle if too long (keep beginning and end which are most important)
    if len(full_text) > max_chars:
        keep_each = (max_chars - 100) // 2
        full_text = (
            full_text[:keep_each] +
            "\n\n[... MIDDLE OF CONVERSATION TRUNCATED FOR LENGTH ...]\n\n" +
            full_text[-keep_each:]
        )

    return full_text


def build_scenario_context(scenario: Dict[str, Any]) -> str:
    """Build context string from scenario metadata."""
    parts = []
    if scenario.get("topic"):
        parts.append(f"**Topic:** {scenario['topic']}")
    if scenario.get("persona"):
        parts.append(f"**Student:** {scenario['persona']}")
    if scenario.get("constraints"):
        parts.append(f"**Constraints:** {scenario['constraints']}")
    return "\n".join(parts) if parts else "General research mentoring conversation"


def render_holistic_prompt(
    template: str,
    scenario_context: str,
    transcript_text: str,
    stop_reason: str,
    turn_count: int,
) -> Tuple[str, str]:
    """Render the holistic judge prompt."""
    system_msg = "You are evaluating the overall quality of a research mentoring conversation. Output JSON only."

    user_msg = template.format(
        scenario_context=scenario_context,
        transcript=transcript_text,
        stop_reason=stop_reason or "unknown",
        turn_count=turn_count,
    )

    return system_msg, user_msg


def parse_holistic_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse holistic judge JSON output."""
    txt = (raw or "").strip()
    if not txt:
        return None

    # Strip code fences
    if txt.startswith("```"):
        lines = txt.split("\n")
        txt = "\n".join(lines[1:])
        if "```" in txt:
            txt = txt[:txt.rfind("```")]

    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    import re
    out: Dict[str, Any] = {}

    for key in ["overall_helpfulness", "student_progress", "mentor_effectiveness", "conversation_efficiency", "holistic_score"]:
        match = re.search(rf'"{key}"\s*:\s*([0-9]+\.?[0-9]*)', txt)
        if match:
            try:
                out[key] = float(match.group(1))
            except ValueError:
                pass

    # Try to extract weaknesses array via regex as fallback
    weaknesses_match = re.search(r'"weaknesses_identified"\s*:\s*\[(.*?)\]', txt, re.DOTALL)
    if weaknesses_match:
        try:
            # Parse the array contents
            weaknesses_str = "[" + weaknesses_match.group(1) + "]"
            out["weaknesses_identified"] = json.loads(weaknesses_str)
        except json.JSONDecodeError:
            pass

    return out if out else None


def aggregate_holistic_scores(judge_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scores from multiple judges."""
    keys = ["overall_helpfulness", "student_progress", "mentor_effectiveness", "conversation_efficiency"]

    scores: Dict[str, List[float]] = {k: [] for k in keys}
    all_weaknesses: List[str] = []

    for output in judge_outputs:
        parsed = output.get("parsed") or {}
        for key in keys:
            val = parsed.get(key)
            if val is not None:
                try:
                    scores[key].append(float(val))
                except (ValueError, TypeError):
                    pass

        # Collect weaknesses from each judge
        weaknesses = parsed.get("weaknesses_identified") or []
        if isinstance(weaknesses, list):
            all_weaknesses.extend(str(w) for w in weaknesses if w)

    result: Dict[str, Any] = {}
    for key in keys:
        result[key] = mean(scores[key]) if scores[key] else None

    # Compute holistic composite: weighted average
    # Helpfulness and progress are most important
    weights = {
        "overall_helpfulness": 0.35,
        "student_progress": 0.30,
        "mentor_effectiveness": 0.20,
        "conversation_efficiency": 0.15,
    }

    if all(result[k] is not None for k in keys):
        result["holistic_score"] = sum(weights[k] * result[k] for k in keys)
    else:
        result["holistic_score"] = None

    # Deduplicate weaknesses (keep unique ones)
    result["weaknesses"] = list(dict.fromkeys(all_weaknesses))

    return result


def compute_success(
    holistic_score: Optional[float],
    stop_reason: str,
    threshold: float,
) -> Tuple[bool, str]:
    """Determine success based on holistic score and stop reason."""
    classification = classify_stop_reason(stop_reason)

    if classification == "negative":
        return False, "failed_negative_stop"

    if holistic_score is None:
        return False, "failed_no_score"

    if classification == "positive":
        if holistic_score >= threshold:
            return True, "score_and_positive_stop"
        return True, "positive_stop_only"  # Trust student satisfaction

    # Ambiguous: require threshold
    if holistic_score >= threshold:
        return True, "score_only"

    return False, "failed_low_score"


def score_conversation(
    run_label: str,
    path: Path,
    judge_template: str,
    judge_clients: Sequence[Tuple[str, Any]],
    threshold: float,
    max_transcript_chars: int,
) -> HolisticResult:
    """Score a single conversation holistically."""
    payload = load_transcript(path)

    scenario = payload.get("scenario") or {}
    transcript = payload.get("transcript") or []
    system_id = str(payload.get("system_id") or "unknown")
    scenario_id = str(scenario.get("scenario_id") or path.stem)
    elapsed_seconds = float(payload.get("elapsed_seconds") or 0.0)
    stop_reason = str(payload.get("stop_reason") or "")
    stopped_by_student = bool(payload.get("stopped_by_student"))
    turn_count = int(payload.get("turn_count") or len([t for t in transcript if t.get("role") == "assistant"]))

    # Format for judge
    scenario_context = build_scenario_context(scenario)
    transcript_text = format_transcript_for_judge(transcript, max_transcript_chars)

    system_msg, user_msg = render_holistic_prompt(
        judge_template,
        scenario_context,
        transcript_text,
        stop_reason,
        turn_count,
    )

    # Call judges
    judge_outputs: List[Dict[str, Any]] = []
    for name, client in judge_clients:
        print_info(f"    Calling judge: {name}")
        try:
            raw = _call_student_judge(client, system_msg, user_msg)
            parsed = parse_holistic_json(raw)
            judge_outputs.append({"judge": name, "raw": raw, "parsed": parsed or {}})
        except Exception as exc:
            print_error(f"    Judge {name} failed: {exc}")
            judge_outputs.append({"judge": name, "error": str(exc)})

    # Aggregate
    scores = aggregate_holistic_scores(judge_outputs)
    stop_reason_class = classify_stop_reason(stop_reason)
    is_success, success_type = compute_success(scores["holistic_score"], stop_reason, threshold)

    return HolisticResult(
        agent_label=run_label,
        system_id=system_id,
        scenario_id=scenario_id,
        total_turns=turn_count,
        elapsed_seconds=elapsed_seconds,
        stop_reason=stop_reason,
        stop_reason_class=stop_reason_class,
        stopped_by_student=stopped_by_student,
        overall_helpfulness=scores["overall_helpfulness"],
        student_progress=scores["student_progress"],
        mentor_effectiveness=scores["mentor_effectiveness"],
        conversation_efficiency=scores["conversation_efficiency"],
        holistic_score=scores["holistic_score"],
        is_success=is_success,
        success_type=success_type,
        weaknesses=scores.get("weaknesses", []),
        judge_outputs=judge_outputs,
    )


def write_results_csv(path: Path, results: List[HolisticResult]) -> None:
    """Write results to CSV."""
    fieldnames = [
        "agent_label",
        "system_id",
        "scenario_id",
        "total_turns",
        "elapsed_seconds",
        "stop_reason",
        "stop_reason_class",
        "stopped_by_student",
        "overall_helpfulness",
        "student_progress",
        "mentor_effectiveness",
        "conversation_efficiency",
        "holistic_score",
        "is_success",
        "success_type",
        "weakness_count",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "agent_label": r.agent_label,
                "system_id": r.system_id,
                "scenario_id": r.scenario_id,
                "total_turns": r.total_turns,
                "elapsed_seconds": f"{r.elapsed_seconds:.2f}",
                "stop_reason": r.stop_reason,
                "stop_reason_class": r.stop_reason_class,
                "stopped_by_student": r.stopped_by_student,
                "overall_helpfulness": f"{r.overall_helpfulness:.3f}" if r.overall_helpfulness else "",
                "student_progress": f"{r.student_progress:.3f}" if r.student_progress else "",
                "mentor_effectiveness": f"{r.mentor_effectiveness:.3f}" if r.mentor_effectiveness else "",
                "conversation_efficiency": f"{r.conversation_efficiency:.3f}" if r.conversation_efficiency else "",
                "holistic_score": f"{r.holistic_score:.3f}" if r.holistic_score else "",
                "is_success": r.is_success,
                "success_type": r.success_type,
                "weakness_count": len(r.weaknesses),
            })


def write_agent_summary(path: Path, results: List[HolisticResult]) -> None:
    """Write per-agent summary."""
    agg: Dict[str, Dict[str, List]] = {}

    for r in results:
        key = r.system_id
        bucket = agg.setdefault(key, {
            "scores": [],
            "turns": [],
            "successes": [],
            "positive_stops": [],
        })
        if r.holistic_score is not None:
            bucket["scores"].append(r.holistic_score)
        bucket["turns"].append(r.total_turns)
        bucket["successes"].append(1 if r.is_success else 0)
        bucket["positive_stops"].append(1 if r.stop_reason_class == "positive" else 0)

    fieldnames = [
        "system_id",
        "conversations",
        "avg_holistic_score",
        "std_holistic_score",
        "avg_turns",
        "success_rate",
        "positive_stop_rate",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for system_id, stats in sorted(agg.items()):
            n = len(stats["turns"])
            scores = stats["scores"]
            avg_score = mean(scores) if scores else None

            # Compute std
            std_score = None
            if len(scores) >= 2:
                avg = mean(scores)
                variance = sum((x - avg) ** 2 for x in scores) / len(scores)
                std_score = variance ** 0.5

            writer.writerow({
                "system_id": system_id,
                "conversations": n,
                "avg_holistic_score": f"{avg_score:.3f}" if avg_score else "",
                "std_holistic_score": f"{std_score:.3f}" if std_score else "",
                "avg_turns": f"{mean(stats['turns']):.1f}",
                "success_rate": f"{sum(stats['successes']) / n:.2%}" if n else "",
                "positive_stop_rate": f"{sum(stats['positive_stops']) / n:.2%}" if n else "",
            })


def write_detailed_json(path: Path, results: List[HolisticResult]) -> None:
    """Write detailed results with judge outputs."""
    data = []
    for r in results:
        data.append({
            "agent_label": r.agent_label,
            "system_id": r.system_id,
            "scenario_id": r.scenario_id,
            "total_turns": r.total_turns,
            "stop_reason": r.stop_reason,
            "stop_reason_class": r.stop_reason_class,
            "scores": {
                "overall_helpfulness": r.overall_helpfulness,
                "student_progress": r.student_progress,
                "mentor_effectiveness": r.mentor_effectiveness,
                "conversation_efficiency": r.conversation_efficiency,
                "holistic_score": r.holistic_score,
            },
            "weaknesses_identified": r.weaknesses,
            "is_success": r.is_success,
            "success_type": r.success_type,
            "judge_outputs": r.judge_outputs,
        })

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    load_env_file()

    # Load judge template
    if not args.template.exists():
        print_error(f"Judge template not found: {args.template}")
        print_info("Creating default holistic judge template...")
        args.template.parent.mkdir(parents=True, exist_ok=True)
        args.template.write_text(DEFAULT_HOLISTIC_TEMPLATE, encoding="utf-8")

    judge_template = args.template.read_text(encoding="utf-8")

    # Build judge clients
    judge_specs = args.judges or list(DEFAULT_JUDGES)
    judge_clients = build_judge_clients(judge_specs)
    judge_names = ", ".join(spec for spec, _ in judge_clients)
    print_info(f"Using judges: {judge_names}")

    # Setup output
    output_dir = args.output_dir or (args.input_root / "holistic_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process transcripts
    results: List[HolisticResult] = []
    transcript_files = list(iter_transcript_files(args.input_root))
    total = len(transcript_files)

    for idx, (run_label, path) in enumerate(transcript_files, start=1):
        print_info(f"[{idx}/{total}] Scoring {path.name} ({run_label})")

        try:
            result = score_conversation(
                run_label,
                path,
                judge_template,
                judge_clients,
                args.threshold,
                args.max_transcript_chars,
            )
            results.append(result)

            score_str = f"{result.holistic_score:.2f}" if result.holistic_score else "N/A"
            print_success(f"  → Score: {score_str}, Success: {result.is_success} ({result.success_type})")

        except Exception as exc:
            print_error(f"  Failed: {exc}")
            continue

    if not results:
        print_error("No conversations scored")
        return 1

    # Write outputs
    write_results_csv(output_dir / "holistic_results.csv", results)
    write_agent_summary(output_dir / "agent_summary.csv", results)
    write_detailed_json(output_dir / "detailed_results.json", results)

    # Print summary
    print_success(f"\nScored {len(results)} conversations")
    print_info(f"Results written to: {output_dir}")

    # Quick stats
    scores = [r.holistic_score for r in results if r.holistic_score is not None]
    if scores:
        print_info(f"Average holistic score: {mean(scores):.2f}")

    success_count = sum(1 for r in results if r.is_success)
    print_info(f"Success rate: {success_count}/{len(results)} ({success_count/len(results):.1%})")

    return 0


# Default template (created if file doesn't exist)
DEFAULT_HOLISTIC_TEMPLATE = '''# Holistic Conversation Evaluation

You are a critical evaluator assessing research mentoring conversations. Your role is to identify both strengths AND weaknesses. Be rigorous and avoid score inflation.

## Scenario Context
{scenario_context}

## Conversation ({turn_count} turns, ended with: {stop_reason})

{transcript}

---

## CRITICAL: Flaw Identification (REQUIRED)

Before scoring, you MUST identify weaknesses in the mentoring. Even good conversations have room for improvement. Consider:

- Did the mentor miss any important aspects of the student's problem?
- Were there unnecessary tangents or repetitive exchanges?
- Could the mentor have been more concise or direct?
- Did the mentor fully address the student's constraints (time, resources, expertise level)?
- Were there missed opportunities to provide more specific guidance?
- Did the conversation take longer than necessary to reach resolution?

**You must identify at least 2 weaknesses or missed opportunities**, even for conversations that seem good overall.

---

## Evaluation Criteria

Rate each dimension from 0.0 to 2.0:

### 1. Overall Helpfulness (overall_helpfulness)
Did the mentor actually help the student make progress on their research problem?
- 0.0: Not helpful at all, student is no better off
- 0.5: Minimally helpful, gave generic advice that ignored specifics
- 1.0: Somewhat helpful, student has some direction but gaps remain
- 1.5: Helpful, student has a solid path forward with minor uncertainties
- 2.0: Exceptionally helpful, student has complete clarity and actionable next steps

### 2. Student Progress (student_progress)
By the end of the conversation, has the student moved forward?
- 0.0: No progress, still stuck or more confused
- 0.5: Minimal progress, student has vague ideas but no concrete plan
- 1.0: Some progress, has partial clarity on what to do
- 1.5: Good progress, student knows the path but may have lingering questions
- 2.0: Excellent progress, student is fully equipped and confident

### 3. Mentor Effectiveness (mentor_effectiveness)
How well did the mentor communicate and adapt to the student?
- 0.0: Poor communication, ignored student's level/constraints entirely
- 0.5: Weak communication, responses were generic or mismatched to student
- 1.0: Adequate communication, generally appropriate but not tailored
- 1.5: Good communication, well-adapted with minor misses
- 2.0: Excellent communication, perfectly calibrated to student's needs

### 4. Conversation Efficiency (conversation_efficiency)
Was the conversation efficient, or did it waste time?
- 0.0: Very inefficient, many wasted turns, went in circles
- 0.5: Inefficient, notable redundancy or unnecessary back-and-forth
- 1.0: Reasonably efficient, some minor redundancy
- 1.5: Efficient, most turns added value
- 2.0: Highly efficient, every single turn was necessary and valuable

---

## Calibration Anchors (YOU MUST USE THESE)

| Score | Meaning | Example Scenario |
|-------|---------|------------------|
| 0.5 | Poor | Mentor gave generic textbook advice, ignored student's specific constraints, student still confused |
| 1.0 | Adequate | Mentor provided relevant guidance but missed nuances, student has rough direction |
| 1.25 | Decent | Mentor addressed main points but conversation had inefficiencies or gaps |
| 1.5 | Good | Mentor gave clear, tailored advice; student knows what to do with minor uncertainties |
| 1.75 | Very Good | Strong mentoring with minimal flaws, student is well-prepared |
| 2.0 | Exceptional | RARE - Mentor demonstrated outstanding insight, perfect adaptation, zero wasted effort |

### Score Distribution Expectations

- **Scores of 2.0 should be rare (<10% of conversations)**
- **Most conversations should fall between 1.0-1.5**
- **A score of 1.5 represents genuinely good mentoring**
- **Only give 1.8+ for truly outstanding conversations with no significant flaws**

⚠️ **WARNING**: If you find yourself giving scores of 1.9-2.0 frequently, you are miscalibrated. Re-read the anchors above.

---

## Output Format

Return ONLY valid JSON with no additional text:

```json
{{
  "weaknesses_identified": [
    "<weakness 1 - be specific>",
    "<weakness 2 - be specific>"
  ],
  "overall_helpfulness": <float 0.0-2.0>,
  "student_progress": <float 0.0-2.0>,
  "mentor_effectiveness": <float 0.0-2.0>,
  "conversation_efficiency": <float 0.0-2.0>,
  "rationale": "<2-3 sentence assessment that references the weaknesses you identified>"
}}
```
'''


if __name__ == "__main__":
    raise SystemExit(main())
