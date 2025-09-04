from __future__ import annotations

"""
Bootstrap utilities for optional registry initialization (WS2).

Keeps side effects contained and returns discovered tool names for callers to log.
"""

from typing import List
import os


def bootstrap_registry_if_enabled() -> List[str]:
    enabled = os.getenv("FF_REGISTRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    if not enabled:
        return []
    try:
        from ..tools import auto_discover, list_tools
    except Exception:
        return []
    try:
        auto_discover()
        names = sorted(list(list_tools().keys()))
        return names
    except Exception:
        return []
