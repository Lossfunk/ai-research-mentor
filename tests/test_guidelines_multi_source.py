from __future__ import annotations

from collections import Counter


def test_multi_source_coverage_and_caps(monkeypatch):
    # Enable v2
    monkeypatch.setenv("FF_GUIDELINES_V2", "1")

    from academic_research_mentor.tools.guidelines.tool import GuidelinesTool

    tool = GuidelinesTool()
    tool.initialize()

    # Keep runtime fast
    monkeypatch.setenv("GUIDELINES_GLOBAL_BUDGET_SECS", "4.0")
    monkeypatch.setenv("GUIDELINES_PER_DOMAIN_BUDGET_SECS", "0.7")

    out = tool.execute({
        "query": "mentorship",
        "topic": "research methodology",
        "mode": "fast",
        "max_per_source": 1,
        "response_format": "concise",
        "page_size": 50,
    })

    assert "evidence" in out
    evidence = out["evidence"]
    # Should not exceed global cap
    assert len(evidence) <= tool.config.RESULT_CAP

    # Multi-source coverage: target at least a few domains when budgets allow
    sources = out.get("sources_covered", [])
    assert isinstance(sources, list)
    assert len(sources) >= 1

    # Per-domain cap enforced (max 1 per domain)
    counts = Counter(item.get("domain") for item in evidence)
    assert all(c <= 1 for c in counts.values())


