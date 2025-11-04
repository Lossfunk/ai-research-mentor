#!/usr/bin/env python3
"""Adaptive multi-turn evaluation runner with a kill-switch student persona."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.rich_formatter import print_error, print_info, print_success
from academic_research_mentor.runtime.context import prepare_agent
from academic_research_mentor.core.transparency import get_transparency_store

from .multi_turn_orchestrator import (
    _json_safe,  # re-use helpers to avoid duplication
    _override_env,
    _resolve_openrouter_student_llm,
)


BASE_DIR = Path(__file__).resolve().parents[1]
SCENARIO_PATH_DEFAULT = BASE_DIR / "multi_turn" / "scenarios.jsonl"
USER_TEMPLATE_PATH = BASE_DIR / "multi_turn" / "student_user_prompt.md"


@dataclass
class ScenarioSpec:
    scenario_id: str
    topic: str
    persona: str
    constraints: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def initial_prompt(self) -> str:
        pieces = [
            "Hey, I'm {persona}.",
            "I want to get started in research in {topic}.",
        ]
        if self.constraints:
            pieces.append(f"Constraints: {self.constraints}.")
        pieces.append("How should I start?")
        return " ".join(pieces).format(persona=self.persona, topic=self.topic)


@dataclass
class MentorReply:
    text: str
    raw: Any
    tool_runs: List[Dict[str, Any]]


@dataclass
class UserDecision:
    continue_conversation: bool
    message: str
    stop_reason: Optional[str]
    notes: Optional[str]
    raw: Any


@dataclass
class ConversationResult:
    scenario: ScenarioSpec
    system_id: str
    transcript: List[Dict[str, Any]]
    stop_reason: str
    stopped_by_student: bool
    error: Optional[str]
    turn_count: int
    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return self.error is None


def load_scenarios(path: Path) -> List[ScenarioSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    records: List[ScenarioSpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            scenario_id = str(payload.get("scenario_id") or payload.get("id") or len(records))
            topic = str(payload.get("topic") or "general research" )
            persona = str(payload.get("persona") or "student")
            constraints = str(payload.get("constraints") or "").strip()
            extra = {k: v for k, v in payload.items() if k not in {"scenario_id", "id", "topic", "persona", "constraints"}}
            records.append(ScenarioSpec(scenario_id, topic, persona, constraints, extra))
    return records


def _safe_system_id(system_spec: str) -> str:
    return system_spec.replace(":", "_").replace("/", "_")


def _summarize_history(history: Sequence[Dict[str, str]], *, max_pairs: int = 3) -> str:
    pairs: List[str] = []
    recent = history[-(max_pairs * 2) :]
    for idx in range(0, len(recent), 2):
        block = recent[idx : idx + 2]
        summary_bits: List[str] = []
        for item in block:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if len(content) > 160:
                content = content[:157] + "..."
            summary_bits.append(f"{role}: {content}")
        if summary_bits:
            pairs.append(" | ".join(summary_bits))
    return "\n".join(pairs[-max_pairs:]) or "(no prior turns)"


def _extract_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate.strip()
    text = getattr(candidate, "content", None) or getattr(candidate, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(text, list):
        joined = "".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in text
        )
        if joined.strip():
            return joined.strip()
    meta = getattr(candidate, "additional_kwargs", None)
    if isinstance(meta, dict):
        completion = meta.get("completion") or meta.get("message") or meta.get("text")
        if isinstance(completion, str) and completion.strip():
            return completion.strip()
    response_meta = getattr(candidate, "response_metadata", None)
    if isinstance(response_meta, dict):
        text = response_meta.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


class MentorAdapter:
    def __init__(
        self,
        system_spec: str,
        *,
        baseline_mode: bool = False,
        tool_whitelist: Optional[Sequence[str]] = None,
    ) -> None:
        self.system_spec = system_spec
        self._baseline_mode = baseline_mode
        self._tool_whitelist = list(tool_whitelist) if tool_whitelist else None
        self._agent = self._prepare_agent(system_spec)

    def _prepare_agent(self, spec: str):
        provider, sep, model = spec.partition(":")
        env_overrides: Dict[str, str] = {}
        if not sep:
            env_overrides["OPENROUTER_MODEL"] = provider.strip()
        else:
            if provider.strip().lower() != "openrouter":
                raise ValueError(f"Unsupported mentor provider '{provider}' in spec '{spec}'")
            env_overrides["OPENROUTER_MODEL"] = model.strip()
        if self._baseline_mode:
            env_overrides["ARM_BASELINE_MODE"] = "1"
            env_overrides.setdefault("ARM_GUIDELINES_MODE", "off")
        if self._tool_whitelist:
            env_overrides["ARM_TOOL_WHITELIST"] = ",".join(self._tool_whitelist)

        load_env_file()
        with _override_env(env_overrides):
            prep = prepare_agent(prompt_arg=None, ascii_override=None)
        if prep.agent is None:
            raise RuntimeError(f"Failed to initialise mentor agent for spec '{spec}': {prep.offline_reason}")
        agent = prep.agent
        if hasattr(agent, "reset_history"):
            try:
                agent.reset_history()
            except Exception:
                pass
        return agent

    def reset(self) -> None:
        if hasattr(self._agent, "reset_history"):
            try:
                self._agent.reset_history()
            except Exception:
                pass

    def respond(self, user_message: str) -> MentorReply:
        store = get_transparency_store()
        try:
            store.clear_runs()
        except Exception:
            pass

        reply_obj = self._agent.run(user_message)
        reply_text = _extract_text(reply_obj) if not isinstance(reply_obj, str) else reply_obj.strip()

        if not reply_text:
            raise RuntimeError("mentor_returned_empty_reply")

        runs_snapshot = []
        for run in store.list_runs():
            runs_snapshot.append(
                {
                    "tool_name": getattr(run, "tool_name", None),
                    "status": getattr(run, "status", None),
                    "duration_seconds": getattr(run, "duration_seconds", None),
                    "metadata": _json_safe(getattr(run, "metadata", {}) or {}),
                }
            )
        try:
            store.clear_runs()
        except Exception:
            pass

        return MentorReply(text=reply_text, raw=_json_safe(reply_obj), tool_runs=runs_snapshot)


class MockMentorAdapter(MentorAdapter):
    def __init__(self, system_spec: str) -> None:
        self.system_spec = system_spec
        self._turn = 0

    def reset(self) -> None:
        self._turn = 0

    def respond(self, user_message: str) -> MentorReply:
        self._turn += 1
        text = (
            f"Turn {self._turn}: let's break this into steps. "
            f"You mentioned '{user_message[:60]}'. First, outline subtopics, then pick one manageable angle."
        )
        return MentorReply(text=text, raw=text, tool_runs=[])


class UserSimulator:
    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 0.5,
        max_tokens: int = 320,
        template_path: Path = USER_TEMPLATE_PATH,
    ) -> None:
        if not template_path.exists():
            raise FileNotFoundError(f"User prompt template not found: {template_path}")
        self._template = template_path.read_text(encoding="utf-8")
        self._client = _resolve_openrouter_student_llm(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_prompt(
        self,
        scenario: ScenarioSpec,
        history: Sequence[Dict[str, str]],
        mentor_reply: str,
    ) -> str:
        summary = _summarize_history(history)
        return self._template.format(
            topic=scenario.topic,
            persona=scenario.persona,
            constraints=scenario.constraints or "(none stated)",
            history_summary=summary,
            mentor_reply=mentor_reply,
        )

    def generate(self, scenario: ScenarioSpec, history: Sequence[Dict[str, str]], mentor_reply: str) -> UserDecision:
        prompt = self._build_prompt(scenario, history, mentor_reply)
        messages = [
            {"role": "system", "content": "You must reply with JSON only."},
            {"role": "user", "content": prompt},
        ]
        raw_payload: Optional[str] = None
        try:
            result = self._client.invoke(messages)
            text = _extract_text(result)
        except Exception as exc:  # noqa: BLE001
            raw_payload = " ".join(str(arg) for arg in exc.args if isinstance(arg, str)) or str(exc)
            print_info(f"[warn] Student model invoke error: {exc!r}")
            text = raw_payload
            result = {"error": repr(exc)}

        print_info(f"[debug] Student model raw output: {repr(text)[:200]}")
        decision = _parse_user_json(text)
        if decision is None:
            stop_reason = "invalid_student_json"
            fallback_message = ""
            if text:
                fallback_message = text if len(text) <= 160 else text[:160] + "..."
            print_info(f"[warn] Falling back due to unparsable student JSON (reason={stop_reason})")
            return UserDecision(
                continue_conversation=False,
                message=fallback_message,
                stop_reason=stop_reason,
                notes=raw_payload,
                raw=_json_safe({"raw_text": text, "error": raw_payload}),
            )
        return UserDecision(
            continue_conversation=bool(decision.get("continue", False)),
            message=str(decision.get("message", "")).strip(),
            stop_reason=str(decision.get("stop_reason")) if decision.get("stop_reason") else None,
            notes=str(decision.get("notes")) if decision.get("notes") else None,
            raw=_json_safe(result),
        )


class MockUserSimulator(UserSimulator):
    def __init__(self) -> None:
        self._turn = 0

    def generate(self, scenario: ScenarioSpec, history: Sequence[Dict[str, str]], mentor_reply: str) -> UserDecision:
        self._turn += 1
        if self._turn >= 2:
            return UserDecision(
                continue_conversation=False,
                message="Thanks, that gives me a concrete plan to follow. I'll stop here.",
                stop_reason="plan_identified",
                notes=None,
                raw={"mock": True, "turn": self._turn},
            )
        return UserDecision(
            continue_conversation=True,
            message=(
                "Got it. I can start by reading survey papers. Could you suggest one or two good keywords "
                "for finding a beginner-friendly dataset?"
            ),
            stop_reason=None,
            notes=None,
            raw={"mock": True, "turn": self._turn},
        )


def _escape_newlines_in_strings(payload: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False
    for char in payload:
        if escape:
            out.append(char)
            escape = False
            continue
        if char == "\\":
            escape = True
            out.append(char)
            continue
        if char == '"':
            in_string = not in_string
            out.append(char)
            continue
        if in_string and char == "\n":
            out.append("\\n")
            continue
        out.append(char)
    return "".join(out)


def _parse_user_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```") and "```" in cleaned[3:]:
        cleaned = cleaned.strip("`\n ")

    # Some models prefix the fence with `json` or similar tokens.
    cleaned = cleaned.lstrip()
    cleaned = re.sub(r"^(json|JSON)\b[:\s]*", "", cleaned)

    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")

    candidates: List[str] = [cleaned, _escape_newlines_in_strings(cleaned)]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = cleaned[start : end + 1]
        candidates.extend([fragment, _escape_newlines_in_strings(fragment)])

    # Try parsing each candidate as JSON, falling back to YAML for lenient syntax.
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    try:  # pragma: no cover - optional dependency path
        import yaml

        for candidate in candidates:
            try:
                data = yaml.safe_load(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    decision = _heuristic_partial_parse(cleaned)
    if decision is not None:
        print_info("[warn] Heuristic parse recovered partial student JSON.")
        return decision

    if last_error:
        print_info(f"[warn] Failed to parse student JSON: {last_error}")
        preview = cleaned if len(cleaned) < 200 else cleaned[:200] + "..."
        print_info(f"[warn] Raw student payload preview: {preview}")
    return None


def _heuristic_partial_parse(text: str) -> Optional[Dict[str, Any]]:
    continue_match = re.search(r'"continue"\s*:\s*(true|false)', text, re.IGNORECASE)
    message_match = re.search(r'"message"\s*:\s*"(.*)', text, re.DOTALL)
    if not message_match and not continue_match:
        return None

    decision: Dict[str, Any] = {}
    if continue_match:
        decision["continue"] = continue_match.group(1).lower() == "true"

    if message_match:
        fragment = message_match.group(1)
        split_match = re.search(r'"\s*,\s*"(?:stop_reason|notes|continue|message)\s*":', fragment)
        if split_match:
            fragment = fragment[: split_match.start()]
        brace_match = re.search(r'"\s*}', fragment)
        if brace_match:
            fragment = fragment[: brace_match.start()]
        fragment = fragment.rstrip('"\n\r\t ,}')
        fragment = fragment.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        decision["message"] = fragment.strip()

    stop_reason_match = re.search(r'"stop_reason"\s*:\s*"(.*?)"', text, re.DOTALL)
    if stop_reason_match:
        decision["stop_reason"] = stop_reason_match.group(1).strip()

    notes_match = re.search(r'"notes"\s*:\s*"(.*?)"', text, re.DOTALL)
    if notes_match:
        decision["notes"] = notes_match.group(1).strip()

    return decision if decision else None


def _build_transcript_entry(role: str, content: str, turn: int) -> Dict[str, Any]:
    return {"role": role, "content": content, "turn": turn}


def _latest_user_message(history: Sequence[Dict[str, str]]) -> str:
    for entry in reversed(history):
        if entry.get("role") == "user":
            return entry.get("content", "")
    return ""


def run_conversation(
    scenario: ScenarioSpec,
    mentor: MentorAdapter,
    user_sim: UserSimulator,
    *,
    max_turns: int,
    killbox_dir: Path,
) -> ConversationResult:
    mentor.reset()
    transcript: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []

    initial_user = scenario.initial_prompt()
    transcript.append(_build_transcript_entry("user", initial_user, 0))
    history.append({"role": "user", "content": initial_user})

    start_time = time.time()
    stop_reason = ""
    stopped_by_student = False
    error: Optional[str] = None

    try:
        for turn_index in range(1, max_turns + 1):
            last_user = _latest_user_message(history)
            if not last_user:
                raise RuntimeError("missing_student_message_before_mentor_call")
            mentor_reply = mentor.respond(last_user)
            transcript.append(_build_transcript_entry("assistant", mentor_reply.text, turn_index))
            history.append({"role": "assistant", "content": mentor_reply.text})

            decision = user_sim.generate(scenario, history, mentor_reply.text)
            if decision.continue_conversation:
                if not decision.message:
                    raise RuntimeError("student_missing_message_despite_continue")
                transcript.append(_build_transcript_entry("user", decision.message, turn_index))
                history.append({"role": "user", "content": decision.message})
                continue

            stopped_by_student = True
            stop_reason = decision.stop_reason or "student_terminated"
            if decision.message:
                transcript.append(_build_transcript_entry("user", decision.message, turn_index))
                history.append({"role": "user", "content": decision.message})
            break
        else:
            stop_reason = "max_turns_reached"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        stop_reason = stop_reason or "error"
        print_info(f"[warn] Conversation exception: {exc!r}")
        import traceback

        traceback.print_exc()

    elapsed = time.time() - start_time

    if stopped_by_student and stop_reason != "":
        killbox_dir.mkdir(parents=True, exist_ok=True)
        safe_id = _safe_system_id(mentor.system_spec)
        kill_path = killbox_dir / f"{safe_id}__{scenario.scenario_id}.json"
        kill_payload = {
            "scenario_id": scenario.scenario_id,
            "system_id": mentor.system_spec,
            "stop_reason": stop_reason,
            "transcript_tail": transcript[-4:],
        }
        kill_path.write_text(json.dumps(kill_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    turn_count = sum(1 for entry in transcript if entry["role"] == "assistant")

    return ConversationResult(
        scenario=scenario,
        system_id=mentor.system_spec,
        transcript=transcript,
        stop_reason=stop_reason,
        stopped_by_student=stopped_by_student,
        error=error,
        turn_count=turn_count,
        elapsed_seconds=elapsed,
    )


def write_transcripts(results: Iterable[ConversationResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        safe_id = _safe_system_id(result.system_id)
        system_dir = output_dir / safe_id
        system_dir.mkdir(parents=True, exist_ok=True)
        path = system_dir / f"{result.scenario.scenario_id}.json"
        payload = {
            "scenario": {
                "scenario_id": result.scenario.scenario_id,
                "topic": result.scenario.topic,
                "persona": result.scenario.persona,
                "constraints": result.scenario.constraints,
                "extra": result.scenario.extra,
            },
            "system_id": result.system_id,
            "stop_reason": result.stop_reason,
            "stopped_by_student": result.stopped_by_student,
            "error": result.error,
            "turn_count": result.turn_count,
            "elapsed_seconds": result.elapsed_seconds,
            "transcript": result.transcript,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(results: Iterable[ConversationResult], output_dir: Path) -> Path:
    records = list(results)
    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario_id",
            "system_id",
            "turn_count",
            "stop_reason",
            "stopped_by_student",
            "error",
            "elapsed_seconds",
        ])
        for item in records:
            writer.writerow([
                item.scenario.scenario_id,
                item.system_id,
                item.turn_count,
                item.stop_reason,
                "yes" if item.stopped_by_student else "no",
                item.error or "",
                f"{item.elapsed_seconds:.2f}",
            ])
    return summary_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive multi-turn mentor evaluations")
    parser.add_argument("--scenarios", default=str(SCENARIO_PATH_DEFAULT), help="Path to scenario JSONL file")
    parser.add_argument("--mentors", nargs="+", required=True, help="Mentor system specs (e.g. openrouter:anthropic/claude-sonnet-4.5)")
    parser.add_argument("--user-model", default="openrouter:moonshot/kimi-k2", help="OpenRouter model id for the student persona")
    parser.add_argument("--max-turns", type=int, default=12, help="Maximum mentor turns before forcing stop")
    parser.add_argument("--output-dir", required=True, help="Directory to store transcripts and summary")
    parser.add_argument("--killbox-dir", help="Optional directory for aborted conversations", default=None)
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for the student model")
    parser.add_argument("--user-max-output", type=int, default=240, help="Max tokens for the student model")
    parser.add_argument("--baseline-mode", action="store_true", help="Run mentors with baseline prompt/tools")
    parser.add_argument("--tool-whitelist", help="Comma-separated tools to allow for mentors")
    parser.add_argument("--dry-run", action="store_true", help="Load scenarios and exit without running")
    parser.add_argument("--mock", action="store_true", help="Use mock mentors and students (for tests)")
    parser.add_argument("--sample", type=int, help="Optional limit on number of scenarios")
    return parser.parse_args(argv)


def build_mentor_adapters(args: argparse.Namespace) -> List[MentorAdapter]:
    mentors: List[MentorAdapter] = []
    for spec in args.mentors:
        if args.mock:
            mentors.append(MockMentorAdapter(spec))
        else:
            whitelist = [tool.strip() for tool in args.tool_whitelist.split(",")] if args.tool_whitelist else None
            mentors.append(MentorAdapter(spec, baseline_mode=args.baseline_mode, tool_whitelist=whitelist))
    return mentors


def build_user_simulator(args: argparse.Namespace) -> UserSimulator:
    if args.mock:
        return MockUserSimulator()
    provider, sep, model = args.user_model.partition(":")
    if not sep:
        model_id = provider
    else:
        if provider.strip().lower() != "openrouter":
            raise ValueError("Currently only openrouter:* user models are supported")
        model_id = model.strip()
    return UserSimulator(
        model_id=model_id,
        temperature=args.temperature,
        max_tokens=args.user_max_output,
        template_path=USER_TEMPLATE_PATH,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        scenario_path = Path(args.scenarios)
        scenarios = load_scenarios(scenario_path)
    except Exception as exc:  # noqa: BLE001
        print_error(str(exc))
        return 1

    if args.sample:
        scenarios = scenarios[: args.sample]

    if not scenarios:
        print_error("No scenarios found; aborting")
        return 1

    if args.dry_run:
        print_info(f"Loaded {len(scenarios)} scenario(s); dry-run requested, exiting")
        return 0

    try:
        mentors = build_mentor_adapters(args)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to prepare mentor agents: {exc}")
        return 1

    try:
        user_sim = build_user_simulator(args)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to prepare user simulator: {exc}")
        return 1

    output_dir = Path(args.output_dir)
    killbox_dir = Path(args.killbox_dir) if args.killbox_dir else (output_dir / "killbox")

    print_info(
        f"Running {len(scenarios)} scenario(s) across {len(mentors)} mentor(s) "
        f"with max {args.max_turns} turns (mock={args.mock})"
    )

    all_results: List[ConversationResult] = []
    for scenario in scenarios:
        for mentor in mentors:
            result = run_conversation(
                scenario,
                mentor,
                user_sim,
                max_turns=args.max_turns,
                killbox_dir=killbox_dir,
            )
            all_results.append(result)
            if result.error:
                print_info(
                    f"[warn] Scenario {scenario.scenario_id} with {mentor.system_spec} errored: {result.error}"
                )

    write_transcripts(all_results, output_dir / "transcripts")
    summary_path = write_summary(all_results, output_dir)

    successes = sum(1 for item in all_results if item.success)
    failures = len(all_results) - successes
    print_success(
        f"Completed multi-turn evals: {len(all_results)} conversations ({successes} ok, {failures} failed)."
    )
    print_success(f"Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
