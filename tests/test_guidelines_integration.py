from __future__ import annotations

import os


def test_orchestrator_emits_citation_policy(monkeypatch):
    monkeypatch.setenv("FF_AGENT_RECOMMENDATION", "1")
    from academic_research_mentor.core.orchestrator import Orchestrator

    orch = Orchestrator()
    out = orch.run_task("mentorship", {"goal": "need research methodology guidance"})
    policy = out.get("policy", {})
    assert policy.get("must_include_citations") is True


def test_dynamic_banner_nonzero_total(monkeypatch):
    # Ensure dynamic mode
    monkeypatch.setenv("ARM_GUIDELINES_MODE", "dynamic")
    from academic_research_mentor.guidelines_engine import create_guidelines_injector
    inj = create_guidelines_injector()
    stats = inj.get_stats()
    gs = stats.get("guidelines_stats", {})
    assert gs.get("total_guidelines", 0) >= 1

