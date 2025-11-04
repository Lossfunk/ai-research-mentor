from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.scripts.run_multi_turn_evals import (  # noqa: E402
    MockMentorAdapter,
    MockUserSimulator,
    ScenarioSpec,
    ScriptedUserSimulator,
    UserDecision,
    USER_TEMPLATE_PATH,
    _parse_user_json,
    run_conversation,
)


def test_parse_user_json_handles_code_fence():
    payload = """```json\n{\n  \"continue\": false,\n  \"message\": \"done\",\n  \"stop_reason\": \"complete\"\n}\n```"""
    parsed = _parse_user_json(payload)
    assert parsed == {"continue": False, "message": "done", "stop_reason": "complete"}


def test_run_conversation_with_mocks(tmp_path):
    scenario = ScenarioSpec(
        scenario_id="unit_test",
        topic="test automation",
        persona="graduate student",
        constraints="has 5 hours per week",
    )
    mentor = MockMentorAdapter("mock:mentor")
    student = MockUserSimulator()

    killbox_dir = tmp_path / "kill"
    result = run_conversation(
        scenario,
        mentor,
        student,
        max_turns=6,
        killbox_dir=killbox_dir,
    )

    assert result.stopped_by_student is True
    assert result.stop_reason == "plan_identified"
    assert result.error is None
    # Expect at least one assistant and one user reply recorded beyond the opener
    assistant_turns = [entry for entry in result.transcript if entry["role"] == "assistant"]
    user_turns = [entry for entry in result.transcript if entry["role"] == "user" and entry["turn"] > 0]
    assert assistant_turns, "mentor replies should be present"
    assert user_turns, "student follow-ups should be present"

    killbox_files = list(killbox_dir.glob("*.json"))
    assert killbox_files, "killbox log should be written when student terminates"
    content = killbox_files[0].read_text(encoding="utf-8")
    assert "plan_identified" in content


def test_user_stops_when_not_helpful(tmp_path):
    scenario = ScenarioSpec(
        scenario_id="unit_test_not_helpful",
        topic="graph ml",
        persona="undergrad",
        constraints="no gpu",
    )
    mentor = MockMentorAdapter("mock:mentor")

    class NotHelpfulUser:
        def __init__(self) -> None:
            self._turn = 0

        def generate(self, scenario, history, mentor_reply):  # noqa: D401
            self._turn += 1
            if self._turn == 1:
                return UserDecision(
                    continue_conversation=True,
                    message="That helps a bit, do you have anything more concrete?",
                    stop_reason=None,
                    notes=None,
                    raw={"mock": True, "turn": self._turn},
                )
            return UserDecision(
                continue_conversation=False,
                message="This isn't helpful anymore, let's stop.",
                stop_reason="not_helpful",
                notes=None,
                raw={"mock": True, "turn": self._turn},
            )

    student = NotHelpfulUser()

    killbox_dir = tmp_path / "kill"
    result = run_conversation(
        scenario,
        mentor,
        student,
        max_turns=4,
        killbox_dir=killbox_dir,
    )

    assert result.stopped_by_student is True
    assert result.stop_reason == "not_helpful"
    assert any(
        entry["role"] == "user" and "isn't helpful anymore" in entry["content"]
        for entry in result.transcript
    )

    killbox_files = list(killbox_dir.glob("*.json"))
    assert killbox_files, "killbox log should exist for student-initiated stop"
    killbox_payload = killbox_files[0].read_text(encoding="utf-8")
    assert "not_helpful" in killbox_payload


def test_scripted_user_simulator_terminates(tmp_path):
    scenario = ScenarioSpec(
        scenario_id="scripted",
        topic="gnn",
        persona="student",
        constraints="",
    )
    mentor = MockMentorAdapter("mock:mentor")
    script = [
        {"message": "Thanks, I'll try that.", "continue": True},
        {"message": "Here are the results: accuracy 0.82.", "continue": False, "stop_reason": "results_ready"},
    ]
    student = ScriptedUserSimulator(
        model_id="openrouter:stub/student",
        script=script,
        temperature=0.0,
        max_tokens=128,
        template_path=USER_TEMPLATE_PATH,
    )

    killbox_dir = tmp_path / "kill"
    result = run_conversation(
        scenario,
        mentor,
        student,
        max_turns=5,
        killbox_dir=killbox_dir,
    )

    assert result.stop_reason == "results_ready"
    assert result.stopped_by_student is True
    assert any(
        entry["role"] == "user" and "accuracy" in entry["content"]
        for entry in result.transcript
    )
