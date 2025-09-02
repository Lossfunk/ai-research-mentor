"""Guidelines Engine for Academic Research Mentor.

This module provides functionality to load, format, and inject research mentorship
guidelines into the agent's context for improved mentoring responses.
"""

from .loader import GuidelinesLoader
from .formatter import GuidelinesFormatter
from .injector import GuidelinesInjector, create_guidelines_injector
from .config import GuidelinesConfig

__all__ = [
    "GuidelinesLoader",
    "GuidelinesFormatter", 
    "GuidelinesInjector",
    "GuidelinesConfig",
    "create_guidelines_injector"
]