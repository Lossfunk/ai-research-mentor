from __future__ import annotations

import os
import re


def test_attachments_retrieval_basic() -> None:
    from academic_research_mentor.attachments import attach_pdfs, search, get_summary

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, "consistency_confound_paper_draft_0.pdf"),
        os.path.join(repo_root, "file-upload-test-docs", "consistency_confound_paper_draft_0.pdf"),
    ]
    pdf_path = next((p for p in candidates if os.path.exists(p)), None)
    if not pdf_path:
        import pytest
        pytest.skip("Test PDF not found; skipping attachments integration test")

    summ = attach_pdfs([pdf_path])
    assert summ.get("files") == 1
    assert summ.get("pages", 0) > 0

    results = search("novelty research questions", k=5)
    assert isinstance(results, list)
    assert len(results) > 0

    # Ensure each result has expected metadata
    file_pages = []
    for r in results:
        assert r.get("text")
        assert r.get("file")
        assert isinstance(r.get("page"), int)
        file_pages.append(f"[{r.get('file')}:{r.get('page')}]")

    # At least one [file:page] citation-like tag can be constructed
    assert any(re.match(r"^\[.+:\d+\]$", fp) for fp in file_pages)


