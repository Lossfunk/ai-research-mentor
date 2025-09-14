from __future__ import annotations

from typing import Any, Dict

_usage: Dict[str, int] = {}
_metrics: Dict[str, int] = {}


def record_tool_usage(tool_name: str) -> None:
    try:
        _usage[tool_name] = _usage.get(tool_name, 0) + 1
    except Exception:
        pass


def get_usage() -> Dict[str, int]:
    return dict(_usage)


def record_metric(name: str, inc: int = 1) -> None:
    try:
        _metrics[name] = _metrics.get(name, 0) + int(inc)
    except Exception:
        pass


def get_metrics() -> Dict[str, int]:
    return dict(_metrics)
