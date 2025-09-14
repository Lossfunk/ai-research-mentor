from __future__ import annotations

import os


def test_end_to_end_flow_minimal() -> None:
    from academic_research_mentor.attachments import attach_pdfs, search
    from academic_research_mentor.runtime.telemetry import get_usage, get_metrics, record_tool_usage, record_metric

    # Load test doc
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf = os.path.join(repo, "file-upload-test-docs", "consistency_confound_paper_draft_0.pdf")
    if not os.path.exists(pdf):
        import pytest
        pytest.skip("missing test pdf")

    attach_pdfs([pdf])
    res = search("novelty research", k=4)
    assert res and isinstance(res, list)
    assert any(r.get("page") for r in res)

    # Fake telemetry updates
    record_tool_usage("attachments_search")
    record_metric("tool_success")
    usage = get_usage()
    metrics = get_metrics()
    assert usage.get("attachments_search") == 1
    assert metrics.get("tool_success") == 1
