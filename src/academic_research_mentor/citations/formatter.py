from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .models import Citation


class CitationFormatter:
    """Format citations consistently across tools.

    Provides a minimal API; can be extended with more styles as needed.
    """

    def __init__(self, style: str = "academic") -> None:
        self.style = style

    def format_inline(self, c: Citation) -> str:
        authors = ", ".join(c.authors[:2]) + (" et al." if len(c.authors) > 2 else "") if c.authors else "Unknown"
        year = str(c.year) if c.year else "n.d."
        title = c.title.strip().rstrip(".")
        return f"{authors} ({year}). {title}. {c.venue or c.source}. {c.url}"

    def format_list(self, citations: Iterable[Citation]) -> List[str]:
        return [self.format_inline(c) for c in citations]

    def to_output_block(self, citations: Iterable[Citation]) -> Dict[str, Any]:
        items = [c.to_dict() for c in citations]
        return {"citations": items, "count": len(items), "style": self.style}


