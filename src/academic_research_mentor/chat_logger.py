"""Chat logging functionality for Academic Research Mentor."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ChatLogger:
    """Logs chat conversations in JSON format similar to the provided examples."""
    
    def __init__(self, log_dir: str = "convo-logs"):
        self.log_dir = Path(log_dir)
        # Create directory if it doesn't exist (with parents)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = []
        self.session_start_time = datetime.now()
        self._exit_handler_registered = False
        self._pending_stage: Optional[Dict[str, Any]] = None
        
    def _generate_log_filename(self) -> str:
        """Generate a unique log filename based on timestamp."""
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        return f"chat_{timestamp}.json"
        
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
        turn_data: Dict[str, Any] = {
            "turn": len(self.current_session) + 1,
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
        
    def add_exit_turn(self, exit_command: str = "exit") -> None:
        """Add an exit turn to the current session."""
        turn_data = {
            "turn": len(self.current_session) + 1,
            "user_prompt": exit_command,
            "tool_calls": [],
            "ai_response": None
        }
        self.current_session.append(turn_data)
        
    def save_session(self) -> str:
        """Save the current session to a JSON file."""
        if not self.current_session:
            return ""
            
        log_file = self.log_dir / self._generate_log_filename()
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)
            
        return str(log_file)
        
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "total_turns": len(self.current_session),
            "session_start": self.session_start_time.isoformat(),
            "log_file": self._generate_log_filename(),
            "has_ai_responses": any(turn.get("ai_response") for turn in self.current_session)
        }