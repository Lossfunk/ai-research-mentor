from __future__ import annotations

import os


def test_orchestrator_sets_citation_policy_for_mentorship(monkeypatch):
    monkeypatch.setenv("FF_AGENT_RECOMMENDATION", "1")
    monkeypatch.setenv("ARM_GUIDELINES_MODE", "dynamic")
    from academic_research_mentor.core.orchestrator import Orchestrator

    orch = Orchestrator()
    out = orch.run_task("mentor_guidance", context={"goal": "research methodology and problem selection advice"})
    policy = out.get("policy", {})
    assert policy.get("must_include_citations") is True
