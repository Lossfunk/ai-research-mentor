"""Rich formatting utilities for enhanced console output."""

from __future__ import annotations

# Compatibility wrapper for legacy imports
from .rich_ui.formatter import RichFormatter, get_formatter  # noqa: F401
from .rich_ui.io_helpers import (  # noqa: F401
    print_formatted_response,
    print_streaming_chunk,
    start_streaming_response,
    end_streaming_response,
    print_error,
    print_info,
    print_success,
    print_agent_reasoning,
    print_user_input,
)
