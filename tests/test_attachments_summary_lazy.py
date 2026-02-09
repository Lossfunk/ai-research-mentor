from __future__ import annotations

from types import SimpleNamespace

from academic_research_mentor.attachments import ingest


def test_document_summary_is_generated_lazily(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    docs = [
        SimpleNamespace(page_content="alpha", metadata={"source": "/tmp/a.pdf", "file_name": "a.pdf", "page": 1}),
        SimpleNamespace(page_content="beta", metadata={"source": "/tmp/a.pdf", "file_name": "a.pdf", "page": 2}),
    ]

    monkeypatch.setattr(ingest, "_load_pdfs", lambda _paths: (docs, {"skipped_large": 0, "truncated": 0}))
    monkeypatch.setattr(ingest, "_split_documents", lambda d: d)
    monkeypatch.setattr(ingest, "_try_build_vector_retriever", lambda _chunks: ("keyword", None))

    calls = {"n": 0}

    def _fake_summary(_docs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return "lazy-summary"

    monkeypatch.setattr(ingest, "generate_document_summary", _fake_summary)

    summary = ingest.attach_pdfs(["/tmp/a.pdf"])
    assert summary["files"] == 1
    assert calls["n"] == 0

    first = ingest.get_document_summary()
    second = ingest.get_document_summary()
    assert first == "lazy-summary"
    assert second == "lazy-summary"
    assert calls["n"] == 1
