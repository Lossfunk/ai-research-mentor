from __future__ import annotations

import os
import signal
import sys
from typing import Any

from dotenv import load_dotenv

from ..rich_formatter import print_info
from ..chat_logger import ChatLogger
from ..session_logging import SessionLogManager, set_active_session_logger


def load_env_file() -> None:
    debug_env = os.environ.get("ARM_DEBUG_ENV", "").lower() in ("1", "true", "yes")

    try:
        if os.path.exists(".env"):
            load_dotenv(".env", verbose=False, override=False)
            if debug_env:
                print(f"Debug: Loaded .env from current directory: {os.path.abspath('.env')}")
            return

        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            env_path = os.path.join(current_dir, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, verbose=False, override=False)
                if debug_env:
                    print(f"Debug: Loaded .env from: {env_path}")
                return
            current_dir = os.path.dirname(current_dir)

        if debug_env:
            print("Debug: No .env file found, using system environment variables only")

    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")


def cleanup_and_save_session(chat_logger: ChatLogger, exit_command: str = "exit", session_logger: SessionLogManager | None = None) -> None:
    saved = False
    if chat_logger.has_user_turns():
        chat_logger.add_exit_turn(exit_command)
        log_file = chat_logger.save_session()
        if log_file:
            summary = chat_logger.get_session_summary()
            print_info(f"Chat session saved to: {log_file}")
            print_info(f"Session summary: {summary['total_turns']} turns, {summary['session_start'][:10]}")
            saved = True
    else:
        print_info("No conversation turns recorded; skipping chat log save.")

    if session_logger:
        if not saved:
            session_logger.log_event("session_discarded", {"reason": "no_turns_recorded", "exit_command": exit_command})
        session_logger.finalize(exit_command)
        set_active_session_logger(None)


def signal_handler(signum, frame):
    signal_name = "Ctrl+C" if signum == signal.SIGINT else f"Signal {signum}"
    print_info(f"\n🛑 {signal_name} received. Saving chat session...")
    sys.exit(0)
