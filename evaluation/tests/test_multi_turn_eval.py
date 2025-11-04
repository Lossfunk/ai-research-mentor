from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.scripts.run_multi_turn_evals import (  # noqa: E402
    MockMentorAdapter,
    MockUserSimulator,
    ScenarioSpec,
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
