from __future__ import annotations

from typing import Optional

from .formatter import get_formatter


def print_formatted_response(content: str, title: Optional[str] = None) -> None:
    get_formatter().print_response(content, title)


def print_streaming_chunk(chunk: str) -> None:
    get_formatter().print_streaming_chunk(chunk)


def start_streaming_response(title: str = "Mentor") -> None:
    get_formatter().start_streaming_response(title)


def end_streaming_response() -> None:
    get_formatter().end_streaming_response()


def print_error(message: str) -> None:
    get_formatter().print_error(message)


def print_info(message: str) -> None:
    get_formatter().print_info(message)


def print_success(message: str) -> None:
    get_formatter().print_success(message)


def print_agent_reasoning(content: str) -> None:
    get_formatter().print_section(content, "Agent's reasoning", border_style="magenta")


def print_user_input(content: str) -> None:
    get_formatter().print_section(content, "You", border_style="cyan")


def print_stage_badge(stage_code: str, stage_name: str, confidence: float) -> None:
    title = f"Stage {stage_code} — {stage_name} (conf {confidence:.2f})"
    # Use a minimal visible glyph to ensure the panel renders even without content
    get_formatter().print_section("—", title, border_style="yellow")
