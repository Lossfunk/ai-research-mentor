from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Academic Research Mentor - AI-powered research assistance with O3-powered literature review",
        epilog="Environment variables are automatically loaded from .env file. Use --env-help for configuration details.",
    )
    parser.add_argument(
        "--attach-pdf",
        action="append",
        default=None,
        help=(
            "Attach one or more PDF files for this session (repeat flag for multiple). "
            "The mentor will retrieve from these documents to ground answers."
        ),
    )
    parser.add_argument(
        "--prompt",
        choices=["mentor", "system"],
        default=None,
        help=(
            "Select prompt variant: 'mentor' for conversational guidance, 'system' for technical assistance "
            "(default: from ARM_PROMPT env var or 'mentor')"
        ),
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Normalize prompt symbols to ASCII-friendly characters for better terminal compatibility",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Verify and display current environment configuration including API keys and agent settings, then exit",
    )
    parser.add_argument(
        "--env-help",
        action="store_true",
        help="Show comprehensive help about environment variables, .env file usage, and configuration options, then exit",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Discover and list available tools (ignores FF_REGISTRY_ENABLED), then exit",
    )
    parser.add_argument(
        "--show-candidates",
        type=str,
        default=None,
        help="Show tool candidates for a goal (uses orchestrator selection), then exit",
    )
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        help="Recommend the best tool for a goal (uses recommender), then exit",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Show recent tool runs from transparency store (in-memory)",
    )
    return parser
