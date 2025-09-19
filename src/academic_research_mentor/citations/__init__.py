from __future__ import annotations

"""
Unified citation utilities for the Academic Research Mentor.

Provides lightweight models and formatting helpers to standardize
citations across tools without introducing heavy dependencies.
"""

from .models import Citation
from .formatter import CitationFormatter

__all__ = [
    "Citation",
    "CitationFormatter",
]


