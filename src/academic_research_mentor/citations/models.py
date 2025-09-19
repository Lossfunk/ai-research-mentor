from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    """Lightweight citation record used across tools.

    Not meant to be exhaustive; captures key fields for grounding and display.
    """

    id: str
    title: str
    url: str
    source: str = "unknown"
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            **({"extra": self.extra} if self.extra else {}),
        }


