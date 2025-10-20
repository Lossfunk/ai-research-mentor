"""Orchestrator for multi-turn conversations between a mentor agent and a synthetic student."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.runtime.context import prepare_agent
from academic_research_mentor.core.transparency import get_transparency_store


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return str(value)


def _truncate_words(text: str, max_words: int = 40) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    words = stripped.split()
    if len(words) <= max_words:
        return stripped
    return " ".join(words[:max_words]) + "..."


def _resolve_openrouter_student_llm(
    model_id: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 256,
) -> Any:
    """Return a ChatOpenAI configured for the synthetic student model."""

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set to run multi-turn evals")

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )


@dataclass
class StudentReply:
    turn_index: int
    content: str
    raw: Any


class UserLMStudent:
    """Wraps an OpenRouter model to act as the student in multi-turn conversations."""

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        followup_instruction: str = (
            "You are the student seeking help. Speak in first person about your uncertainties,"
            " ask short clarifying questions under 80 words, end with a question mark,"
            " and never provide advice or explanations."
        ),
    ) -> None:
        self._model_id = model_id
        self._llm = _resolve_openrouter_student_llm(model_id, temperature=temperature, max_tokens=max_tokens)
        self._followup_instruction = followup_instruction
        self._nudge_instructions = [
            (
                "Reminder: stay in the student role. Ask a single follow-up question under 60 words,"
                " end with exactly one question mark, and do not give explanations or advice."
            ),
            (
                "Stay curious and brief—pose one clarifying question under 55 words, end with one question mark,"
                " and avoid giving any suggestions or instructions."
            ),
            (
                "You're the student. Ask one specific follow-up question (≤50 words) ending with a question mark;"
                " don't provide explanations, lists, or action steps."
            ),
        ]

    def _convert_messages(self, messages: Sequence[Dict[str, str]]) -> List[Any]:
        converted: List[Any] = []
        for entry in messages:
            role = entry.get("role", "user").strip().lower()
            content = entry.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def _extract_text(self, result: Any) -> str:
        candidate = getattr(result, "content", None) or getattr(result, "text", None)
        if callable(candidate):
            try:
                candidate = candidate()
            except Exception:  # noqa: BLE001
                candidate = None
        if isinstance(candidate, list):
            candidate = "".join(
                [chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in candidate]
            )
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

        meta = getattr(result, "additional_kwargs", None)
        if isinstance(meta, dict):
            candidates = meta.get("candidates")
            if isinstance(candidates, list) and candidates:
                parts = candidates[0].get("content")
                if isinstance(parts, str) and parts.strip():
                    return parts.strip()

        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            text = response_meta.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

        return ""

    def _is_valid_student_message(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if "?" not in stripped:
            return False
        if len(stripped.split()) > 80:
            return False
        lowered = stripped.lower()
        banned_phrases = [
            "let's",
            "here's",
            "i recommend",
            "you should",
            "we should",
            "welcome",
            "i'll explain",
            "break this down",
        ]
        if any(phrase in lowered for phrase in banned_phrases):
            return False
        if stripped.count("?") != 1:
            return False
        return True

    def generate(
        self,
        system_prompt: str,
        stage_instruction: str,
        history: Sequence[Dict[str, str]],
        turn_index: int,
        *,
        retries: int = 1,
    ) -> StudentReply:
        base_messages: List[Any] = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=self._followup_instruction),
            SystemMessage(content=stage_instruction),
        ]

        conversation = [entry for entry in history if entry.get("role") != "system"]
        converted_history = self._convert_messages(conversation)

        last_result: Any = None
        for attempt in range(retries + 1):
            nudge_text = self._nudge_instructions[attempt % len(self._nudge_instructions)]
            payload = [*base_messages, *converted_history, HumanMessage(content=nudge_text)]
            result = self._llm.invoke(payload)
            last_result = result
            text = self._extract_text(result)
            if text and self._is_valid_student_message(text):
                return StudentReply(turn_index=turn_index, content=text, raw=_json_safe(result))
            converted_history = [*converted_history, HumanMessage(content=nudge_text)]

        return StudentReply(turn_index=turn_index, content="", raw=_json_safe(last_result))


@dataclass
class TurnRecord:
    role: str
    turn: int
    content: str


@dataclass
class ScenarioResult:
    scenario_id: str
    system_id: str
    transcript: List[TurnRecord]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str]
    elapsed_seconds: float
    tool_runs: List[Dict[str, Any]]


class MultiTurnOrchestrator:
    """Coordinates mentor agents with synthetic student loops for multi-turn evals."""

    def __init__(
        self,
        mentor_specs: Sequence[str],
        student_model: str = "moonshotai/kimi-k2-0905",
        *,
        max_turns: int = 3,
        student_history_limit: int = 12,
        student_followup_instruction: str = (
            "You are the student. Respond with a short, curious follow-up question or clarification."
            " Stay within 80 words, never provide advice or instructions, and remain in character."
        ),
        baseline_mode: bool = False,
        tool_whitelist: Optional[Sequence[str]] = None,
    ) -> None:
        if not mentor_specs:
            raise ValueError("mentor_specs cannot be empty")
        self._mentor_specs = list(mentor_specs)
        self._student_model = student_model
        self._max_turns = max_turns
        self._student_history_limit = max(4, student_history_limit)
        self._student_followup_instruction = student_followup_instruction
        self._baseline_mode = bool(baseline_mode)
        self._tool_whitelist = list(tool_whitelist) if tool_whitelist else None
        self._stage_prompts = {
            1: "Stage A: Pre-idea exploration. You feel lost and need big-picture help.",
            2: "Stage B: Problem framing. You're clarifying scope and baseline setup.",
            3: "Stage C: Experiment planning. You're digging into design and logistics.",
            4: "Stage D: Analysis and interpretation. You're examining results.",
            5: "Stage E: Writing and polish. You're refining narrative and presentation.",
        }
        self._mentors: Dict[str, Any] = {}

    def _spec_to_env(self, system_spec: str) -> Dict[str, str]:
        provider, sep, model = system_spec.partition(":")
        if not sep:
            return {"OPENROUTER_MODEL": provider.strip()}
        provider_key = provider.strip().lower()
        model_id = model.strip()
        if provider_key != "openrouter":
            raise ValueError(f"Unsupported provider in system spec '{system_spec}'")
        if not model_id:
            raise ValueError(f"Missing model identifier in system spec '{system_spec}'")
        return {"OPENROUTER_MODEL": model_id}

    def _ensure_mentors(self) -> None:
        if self._mentors:
            return
        load_env_file()
        for spec in self._mentor_specs:
            env_overrides = self._spec_to_env(spec)
            if self._baseline_mode:
                env_overrides["ARM_BASELINE_MODE"] = "1"
                env_overrides["ARM_GUIDELINES_MODE"] = "off"
                if self._tool_whitelist:
                    env_overrides["ARM_TOOL_WHITELIST"] = ",".join(self._tool_whitelist)
                else:
                    env_overrides.setdefault("ARM_TOOL_WHITELIST", "attachments_search,web_search")
                baseline_prompt_path = Path("baseline_prompt.md")
                if baseline_prompt_path.exists():
                    env_overrides["ARM_PROMPT_FILE"] = str(baseline_prompt_path.resolve())
                    env_overrides["ARM_PROMPT"] = "baseline"
                else:
                    print(
                        "Baseline mode warning: baseline_prompt.md not found; falling back to default prompt",
                    )
            elif self._tool_whitelist:
                env_overrides["ARM_TOOL_WHITELIST"] = ",".join(self._tool_whitelist)
            with _override_env(env_overrides):
                prep = prepare_agent(prompt_arg=None, ascii_override=None)
            if prep.agent is None:
                raise RuntimeError(f"Failed to init mentor agent '{spec}': {prep.offline_reason}")
            if hasattr(prep.agent, "reset_history"):
                try:
                    prep.agent.reset_history()
                except Exception:
                    pass
            self._mentors[spec] = prep.agent

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, ScenarioResult]:
        self._ensure_mentors()

        scenario_id = str(scenario.get("scenario_id"))
        system_prompt = str(scenario.get("system_prompt", ""))
        requested_turns = int(scenario.get("num_turns", self._max_turns))
        total_turns = min(requested_turns, self._max_turns)

        student = UserLMStudent(self._student_model, followup_instruction=self._student_followup_instruction)

        results: Dict[str, ScenarioResult] = {}
        for mentor_id, agent in self._mentors.items():
            transcript: List[TurnRecord] = [TurnRecord(role="system", turn=0, content=system_prompt)]
            history: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
            ]
            turn_contexts: List[str] = []

            start = time.time()
            success = True
            error: Optional[str] = None

            total_tool_runs: List[Dict[str, Any]] = []

            try:
                if hasattr(agent, "reset_history"):
                    agent.reset_history()

                for turn_index in range(1, total_turns + 1):
                    trimmed_history = history[-self._student_history_limit :]
                    history_for_student = list(trimmed_history)
                    stage_instruction = scenario.get("stage_instructions", {}).get(str(turn_index))
                    if not stage_instruction:
                        stage_instruction = self._stage_prompts.get(turn_index, "")
                    recent_context = " ".join(turn_contexts[-2:])
                    stage_instruction_with_summary = stage_instruction
                    if recent_context:
                        stage_instruction_with_summary = f"{stage_instruction}\nRecent context: {recent_context}"
                    student_reply = student.generate(
                        system_prompt,
                        stage_instruction_with_summary,
                        history_for_student,
                        turn_index,
                        retries=2,
                    )
                    user_text = student_reply.content.strip()
                    if (
                        not user_text
                        or user_text.lower().startswith("assistant")
                        or stage_instruction and "stage" in stage_instruction.lower() and "?" not in user_text
                    ):
                        success = False
                        error = "student_generation_failed"
                        break
                    history.append({"role": "user", "content": user_text})
                    transcript.append(TurnRecord(role="user", turn=turn_index, content=user_text))

                    store = get_transparency_store()
                    try:
                        store.clear_runs()
                    except Exception:
                        pass

                    mentor_text = ""
                    mentor_reply_obj: Any = None
                    mentor_attempt_error: Optional[str] = None
                    for mentor_attempt in range(2):
                        if mentor_attempt > 0:
                            try:
                                store.clear_runs()
                            except Exception:
                                pass
                        mentor_reply_obj = agent.run(user_text)
                        if isinstance(mentor_reply_obj, str):
                            candidate_text = mentor_reply_obj
                        else:
                            candidate_text = getattr(mentor_reply_obj, "content", "") or getattr(mentor_reply_obj, "text", "") or ""
                        mentor_text = (candidate_text or "").strip()
                        if mentor_text:
                            break
                        mentor_attempt_error = "empty_mentor_reply"
                    if not mentor_text:
                        success = False
                        error = mentor_attempt_error or "empty_mentor_reply"
                        break
                    transcript.append(TurnRecord(role="assistant", turn=turn_index, content=mentor_text))
                    history.append({"role": "assistant", "content": mentor_text})
                    student_snip = _truncate_words(user_text, max_words=30)
                    mentor_snip = _truncate_words(mentor_text, max_words=40)
                    turn_contexts.append(
                        f"Turn {turn_index}: Student asked: {student_snip}. Mentor replied: {mentor_snip}."
                    )

                    runs_snapshot = [
                        {
                            "turn": turn_index,
                            "tool_name": getattr(run, "tool_name", None),
                            "status": getattr(run, "status", None),
                            "success": getattr(run, "status", "").lower() == "success" if hasattr(run, "status") else getattr(run, "success", None),
                            "duration_seconds": getattr(run, "duration_seconds", None),
                            "metadata": _json_safe(getattr(run, "metadata", {}) or {}),
                        }
                        for run in store.list_runs()
                    ]
                    total_tool_runs.extend(runs_snapshot)

            except Exception as exc:  # noqa: BLE001
                success = False
                error = str(exc)
            finally:
                try:
                    store.clear_runs()
                except Exception:
                    pass

            user_turns = sum(1 for turn in transcript if turn.role == "user")
            mentor_turns = sum(1 for turn in transcript if turn.role == "assistant")
            if user_turns < total_turns or mentor_turns < total_turns:
                success = False
                if error is None:
                    error = "incomplete_conversation"

            elapsed = time.time() - start
            results[mentor_id] = ScenarioResult(
                scenario_id=scenario_id,
                system_id=mentor_id,
                transcript=transcript,
                metadata=dict(scenario.get("metadata") or {}),
                success=success,
                error=error,
                elapsed_seconds=elapsed,
                tool_runs=total_tool_runs,
            )

        return results

    def run_batch(self, scenarios: Iterable[Dict[str, Any]]) -> Dict[str, List[ScenarioResult]]:
        output: Dict[str, List[ScenarioResult]] = {spec: [] for spec in self._mentor_specs}
        for scenario in scenarios:
            result_map = self.run_scenario(scenario)
            for spec, outcome in result_map.items():
                output.setdefault(spec, []).append(outcome)
        return output


def load_scenarios(jsonl_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def export_transcripts(results: Dict[str, List[ScenarioResult]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for system_id, records in results.items():
        safe_id = system_id.replace(":", "_").replace("/", "_")
        file_path = output_dir / f"{safe_id}.jsonl"
        with file_path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "scenario_id": record.scenario_id,
                    "system_id": record.system_id,
                    "metadata": record.metadata,
                    "success": record.success,
                    "error": record.error,
                    "elapsed_seconds": record.elapsed_seconds,
                    "tool_runs": record.tool_runs,
                    "transcript": [
                        {
                            "role": turn.role,
                            "turn": turn.turn,
                            "content": turn.content,
                        }
                        for turn in record.transcript
                    ],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@contextmanager
def _override_env(env_overrides: Dict[str, str]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in env_overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
