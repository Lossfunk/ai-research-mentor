from __future__ import annotations

"""Minimal PDF attachments ingestion for session-scoped retrieval."""

from typing import Any, Tuple
import os

from .summarizer import generate_document_summary
from .pdf_loader import load_pdfs

# ---- Module state (session-scoped) ----
_chunk_texts: list[str] = []
_chunk_meta: list[dict] = []
_retriever: Any | None = None
_summary: dict[str, Any] = {"files": 0, "pages": 0, "chunks": 0}
_doc_summary: str = ""
_LIMITS = {"max_mb": 50, "max_pages": 500}
_DEFAULT_K = 12


def _try_build_vector_retriever(chunks: list[Any]) -> tuple[str, Any | None]:
    """Attempt to construct a FAISS retriever with available embeddings.
    Falls back to None if embeddings/backends not available.
    """
    try:
        # Embeddings preference: FastEmbed (no key) -> OpenAI -> None
        embeddings = None
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore

            embeddings = FastEmbedEmbeddings()
            backend = "fastembed"
        except Exception:
            embeddings = None
            backend = "none"

        if embeddings is None:
            try:
                # Requires langchain-openai package and API key
                from langchain_openai import OpenAIEmbeddings  # type: ignore

                if os.environ.get("OPENAI_API_KEY"):
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    backend = "openai"
            except Exception:
                pass

        if embeddings is None:
            return "keyword", None

        from langchain_community.vectorstores import FAISS  # type: ignore

        index = FAISS.from_documents(chunks, embeddings)
        retriever = index.as_retriever(search_type="mmr", search_kwargs={"k": _DEFAULT_K, "fetch_k": 48})
        return backend, retriever
    except Exception:
        return "keyword", None


def _split_documents(docs: list[Any]) -> list[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _load_pdfs(paths: list[str]) -> tuple[list[Any], dict[str, int]]:
    """Load PDFs using PyMuPDF for better extraction. Delegates to pdf_loader module."""
    return load_pdfs(paths, _LIMITS)


def attach_pdfs(paths: list[str]) -> dict[str, Any]:
    """Attach PDFs for current session and build a retriever.

    Returns summary with counts. Safe to call multiple times; rebuilds index.
    """
    global _retriever, _chunk_texts, _chunk_meta, _summary, _doc_summary

    docs, stats = _load_pdfs(paths)
    if not docs:
        _retriever = None
        _chunk_texts = []
        _chunk_meta = []
        _doc_summary = ""
        _summary = {"files": 0, "pages": 0, "chunks": 0, "backend": "none", "skipped_large": 0, "truncated": 0}
        return _summary

    pages_by_file: dict[str, int] = {}
    for d in docs:
        fn = (d.metadata or {}).get("file_name") or "file.pdf"
        pages_by_file[fn] = pages_by_file.get(fn, 0) + 1

    chunks = _split_documents(docs)
    _chunk_texts = [c.page_content for c in chunks]
    _chunk_meta = [c.metadata or {} for c in chunks]

    backend_name, retr = _try_build_vector_retriever(chunks)
    _retriever = retr

    # Generate document summary for context awareness
    _doc_summary = generate_document_summary(docs)

    _summary = {
        "files": len({(d.metadata or {}).get("source") for d in docs}),
        "pages": sum(pages_by_file.values()),
        "chunks": len(chunks),
        "backend": backend_name,
        "skipped_large": int(stats.get("skipped_large", 0)),
        "truncated": int(stats.get("truncated", 0)),
    }
    return _summary


def has_attachments() -> bool:
    return bool(_chunk_texts)


def _keyword_rank(query: str, k: int = 12) -> list[Tuple[int, float]]:
    """Very simple keyword ranker when vectors are unavailable."""
    if not query:
        return []
    terms = [t.lower() for t in query.split() if len(t) > 2]
    scored: list[Tuple[int, float]] = []
    for idx, text in enumerate(_chunk_texts):
        t = text.lower()
        score = sum(t.count(term) for term in terms)
        if score > 0:
            scored.append((idx, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _make_snippet(text: str, query: str, max_len: int = 240) -> str:
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    # Center around first occurrence of any query term
    terms = [w.lower() for w in (query or "").split() if len(w) > 2]
    idx = -1
    lower = t.lower()
    for term in terms:
        idx = lower.find(term)
        if idx != -1:
            break
    if idx == -1:
        return t[: max_len - 1] + "…"
    start = max(0, idx - max_len // 2)
    end = min(len(t), start + max_len)
    snippet = t[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(t):
        snippet = snippet + "…"
    return snippet


def search(query: str, k: int = 12) -> list[dict[str, Any]]:
    """Retrieve top-k chunks with metadata and snippet text."""
    if not has_attachments():
        return []

    try:
        if _retriever is not None:
            docs = _retriever.invoke(query) if hasattr(_retriever, "invoke") else _retriever.get_relevant_documents(query)
            results: list[dict[str, Any]] = []
            for d in docs[:k]:
                meta = d.metadata or {}
                page = meta.get("page") or meta.get("page_number") or 1
                file_name = meta.get("file_name") or os.path.basename(meta.get("source", "")) or "file.pdf"
                results.append(
                    {
                        "text": d.page_content,
                        "snippet": _make_snippet(d.page_content or "", query),
                        "file": file_name,
                        "page": int(page) if isinstance(page, int) else 1,
                        "source": meta.get("source", ""),
                        "anchor": f"{file_name}#page={int(page) if isinstance(page, int) else 1}",
                    }
                )
            return results
    except Exception:
        pass

    # Fallback keyword search
    ranked = _keyword_rank(query, k=k)
    results_k: list[dict[str, Any]] = []
    for idx, _s in ranked:
        meta = _chunk_meta[idx] if idx < len(_chunk_meta) else {}
        page = meta.get("page") or meta.get("page_number") or 1
        file_name = meta.get("file_name") or os.path.basename(meta.get("source", "")) or "file.pdf"
        results_k.append(
            {
                "text": _chunk_texts[idx],
                "snippet": _make_snippet(_chunk_texts[idx] or "", query),
                "file": file_name,
                "page": int(page) if isinstance(page, int) else 1,
                "source": meta.get("source", ""),
                "anchor": f"{file_name}#page={int(page) if isinstance(page, int) else 1}",
            }
        )
    return results_k


def get_summary() -> dict[str, Any]:
    return dict(_summary)


def get_document_summary() -> str:
    """Get the LLM-generated summary of attached documents.
    
    Returns structured summary with experiments, findings, methods, and questions.
    Empty string if no attachments or summary generation failed.
    """
    return _doc_summary