from __future__ import annotations

import os

import pytest

from academic_research_mentor.cli.openrouter_setup import maybe_run_openrouter_setup


def test_force_setup_applies_env_and_persists(monkeypatch: pytest.MonkeyPatch, tmp_path):
    inputs = iter(["2", "y"])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    def fake_getpass(_prompt: str) -> str:
        return "sk-openrouter"

    monkeypatch.setenv("ARM_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    applied = maybe_run_openrouter_setup(force=True, input_fn=fake_input, getpass_fn=fake_getpass)

    assert applied is True
    assert os.environ["OPENROUTER_API_KEY"] == "sk-openrouter"
    assert os.environ["OPENROUTER_MODEL"] == "openai/gpt-5"

    config_path = tmp_path / "academic-research-mentor" / ".env"
    assert config_path.exists()
    content = config_path.read_text()
    assert "OPENROUTER_API_KEY=sk-openrouter" in content
    assert "OPENROUTER_MODEL=openai/gpt-5" in content


def test_auto_setup_skipped_when_other_provider(monkeypatch: pytest.MonkeyPatch):
    inputs_called = False

    def fake_input(_prompt: str) -> str:
        nonlocal inputs_called
        inputs_called = True
        return ""

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("ARM_SKIP_INTERACTIVE_SETUP", "")

    from academic_research_mentor.cli import openrouter_setup

    monkeypatch.setattr(openrouter_setup, "_is_interactive_terminal", lambda: True)

    applied = maybe_run_openrouter_setup(force=False, input_fn=fake_input)

    assert applied is False
    assert not inputs_called

    monkeypatch.delenv("OPENAI_API_KEY")


def test_persist_defaults_to_local_config(monkeypatch: pytest.MonkeyPatch, tmp_path):
    inputs = iter(["", "y"])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    def fake_getpass(_prompt: str) -> str:
        return "sk-openrouter-local"

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARM_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    applied = maybe_run_openrouter_setup(force=True, input_fn=fake_input, getpass_fn=fake_getpass)

    assert applied is True
    config_path = tmp_path / ".config" / "academic-research-mentor" / ".env"
    assert config_path.exists()
    content = config_path.read_text()
    assert "OPENROUTER_API_KEY=sk-openrouter-local" in content
