from __future__ import annotations

"""PDF loading utilities using PyMuPDF for better text extraction."""

from typing import Any
import os


def load_pdfs(paths: list[str], limits: dict[str, int]) -> tuple[list[Any], dict[str, int]]:
    """Load PDFs using PyMuPDF for better text extraction quality.
    
    Args:
        paths: List of PDF file paths to load
        limits: Dictionary with "max_mb" and "max_pages" constraints
        
    Returns:
        Tuple of (documents, stats) where:
        - documents: List of Document objects with page_content and metadata
        - stats: Dictionary with "skipped_large" and "truncated" counts
    """
    import fitz  # PyMuPDF

    documents: list[Any] = []
    stats = {"skipped_large": 0, "truncated": 0}
    
    for p in paths:
        try:
            abs_path = os.path.abspath(os.path.expanduser(p))
            if not os.path.exists(abs_path):
                continue
                
            # Skip overly large files (>50MB default)
            try:
                if os.path.getsize(abs_path) > limits["max_mb"] * 1024 * 1024:
                    stats["skipped_large"] += 1
                    continue
            except Exception:
                pass

            # Use PyMuPDF for better text extraction
            doc = fitz.open(abs_path)
            page_count = min(len(doc), limits["max_pages"])
            if len(doc) > limits["max_pages"]:
                stats["truncated"] += 1
            
            for page_num in range(page_count):
                page = doc[page_num]
                # Extract text with better layout preservation
                text = page.get_text("text", sort=True)
                
                # Create Document-like object for compatibility
                from langchain_core.documents import Document
                doc_obj = Document(
                    page_content=text,
                    metadata={
                        "source": abs_path,
                        "file_name": os.path.basename(abs_path),
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                )
                documents.append(doc_obj)
            
            doc.close()
        except Exception:
            # Skip unreadable file; keep MVP resilient
            continue
            
    return documents, stats
