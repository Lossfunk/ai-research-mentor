"""Chat logging functionality for Academic Research Mentor."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .session_logging import SessionLogManager


class ChatLogger:
    """Logs chat conversations in JSON format similar to the provided examples."""
    
    def __init__(self, log_dir: str = "convo-logs", session_logger: Optional[SessionLogManager] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_logger = session_logger
        if session_logger:
            self.session_id = session_logger.session_id
            self.session_start_time = datetime.fromtimestamp(session_logger.started_ms / 1000)
        else:
            now = datetime.now()
            self.session_start_time = now
            self.session_id = f"chat_{now.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.log_dir / self.session_id
        if not self.session_dir.exists():
            self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = []
        self._exit_handler_registered = False
        self._pending_stage: Optional[Dict[str, Any]] = None
        self._real_turns: int = 0

    def _log_path(self) -> Path:
        """Return the path for the primary chat log file."""
        return self.session_dir / f"{self.session_id}.json"
        
    def set_pending_stage(self, stage: Dict[str, Any]) -> None:
        """Set a stage dict to be attached to the next added turn."""
        try:
            self._pending_stage = dict(stage) if isinstance(stage, dict) else None
        except Exception:
            self._pending_stage = None

    def add_turn(self, 
                 user_prompt: str, 
                 tool_calls: List[Dict[str, Any]], 
                 ai_response: Optional[str] = None,
                 stage: Optional[Dict[str, Any]] = None) -> None:
        """Add a conversation turn to the current session."""
        turn_number = len(self.current_session) + 1
        turn_data: Dict[str, Any] = {
            "turn": turn_number,
            "user_prompt": user_prompt,
            "tool_calls": tool_calls,
            "ai_response": ai_response
        }
        stage_payload = stage if stage is not None else self._pending_stage
        if stage_payload:
            turn_data["stage"] = stage_payload
        # Clear pending stage after consumption
        self._pending_stage = None
        self.current_session.append(turn_data)
        self._real_turns += 1
        if self._session_logger:
            self._session_logger.finalize_turn(turn_number, {
                "user_prompt": user_prompt,
                "tool_calls": tool_calls,
                "ai_response": ai_response,
                "stage": stage_payload,
            })
        
    def add_exit_turn(self, exit_command: str = "exit") -> None:
        """Add an exit turn to the current session."""
        turn_number = len(self.current_session) + 1
        turn_data = {
            "turn": turn_number,
            "user_prompt": exit_command,
            "tool_calls": [],
            "ai_response": None
        }
        self.current_session.append(turn_data)
        if self._session_logger:
            self._session_logger.log_event("exit_recorded", {"exit_command": exit_command, "turn": turn_number})
            self._session_logger.finalize_turn(turn_number, {
                "user_prompt": exit_command,
                "tool_calls": [],
                "ai_response": None,
                "exit": True,
            })
        
    def save_session(self) -> str:
        """Save the current session to a JSON file."""
        if not self.current_session:
            return ""
            
        log_file = self._log_path()
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)

        if self._session_logger:
            summary = self.get_session_summary()
            self._session_logger.attach_metadata("chat_log_path", str(log_file))
            self._session_logger.attach_metadata("total_turns", summary["total_turns"])
            self._session_logger.attach_metadata("chat_log_dir", str(self.session_dir))
            
        return str(log_file)

    def next_turn_number(self) -> int:
        return len(self.current_session) + 1

    def has_user_turns(self) -> bool:
        return self._real_turns > 0
        
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "total_turns": len(self.current_session),
            "session_start": self.session_start_time.isoformat(),
            "log_file": self._log_path().name,
            "log_dir": str(self.session_dir),
            "has_ai_responses": any(turn.get("ai_response") for turn in self.current_session)
        }