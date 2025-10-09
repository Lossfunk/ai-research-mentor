from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..rich_ui.io_helpers import print_stage_badge
from ..rich_formatter import print_info, print_user_input
from ..session_logging import SessionLogManager
from ..chat_logger import ChatLogger
from ..cli.repl_helpers import (
    CommandOutcome,
    build_react_enhanced_input,
    create_session_stack,
    handle_system_command,
    run_agent_turn,
    safe_detect_stage,
)
from ..cli.session import cleanup_and_save_session


@dataclass
class ConversationOutcome:
    exit_command: Optional[str] = None
    handled: bool = False


class TUISessionManager:
    """Session orchestration for the Textual front-end."""

    def __init__(self, agent: Any, loaded_variant: str) -> None:
        self._agent = agent
        self._loaded_variant = loaded_variant
        metadata = {"loaded_prompt_variant": loaded_variant}
        self._session_logger, self._chat_logger = create_session_stack(metadata)
        if hasattr(agent, "set_chat_logger"):
            agent.set_chat_logger(self._chat_logger)
        if hasattr(agent, "set_session_logger"):
            try:
                agent.set_session_logger(self._session_logger)
            except Exception:  # pragma: no cover - defensive fallback
                pass
        print_info(f"Loaded prompt variant: {loaded_variant}")
        print_info("Type 'exit' to quit")
        self._closed = False

    @property
    def session_logger(self) -> SessionLogManager:
        return self._session_logger

    @property
    def chat_logger(self) -> ChatLogger:
        return self._chat_logger

    def handle_user_message(self, user_text: str) -> ConversationOutcome:
        user = user_text.strip()
        self._session_logger.log_event("raw_input", {"text": user})

        outcome = handle_system_command(user, self._agent, self._session_logger, allow_resume=True)
        if outcome.exit_command:
            self.close(outcome.exit_command)
            return ConversationOutcome(exit_command=outcome.exit_command, handled=True)
        if outcome.handled:
            return ConversationOutcome(handled=True)

        turn_number = self._chat_logger.next_turn_number()
        self._session_logger.start_turn(turn_number, user)

        stage = safe_detect_stage(user, self._chat_logger, self._session_logger)
        if stage:
            stage_code = str(stage.get("code", "")).upper() or "A"
            stage_name = str(stage.get("name", "")).strip() or "Pre idea"
            confidence = float(stage.get("confidence", 0.0))
            print_stage_badge(stage_code, stage_name, confidence)

        print_user_input(user)

        enhanced_input = build_react_enhanced_input(user, self._session_logger)

        run_agent_turn(
            self._agent,
            user,
            enhanced_input,
            use_manual_routing=False,
            chat_logger=self._chat_logger,
            session_logger=self._session_logger,
            turn_number=turn_number,
        )
        return ConversationOutcome()

    def close(self, exit_command: str = "exit") -> None:
        if self._closed:
            return
        cleanup_and_save_session(self._chat_logger, exit_command, self._session_logger)
        self._closed = True
