from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict


def should_debug_log() -> bool:
    return os.getenv("ARM_DEBUG_LITERATURE", "false").lower() in ("true", "1", "yes")


def init_debug_logging(user_input: str) -> Dict[str, Any]:
    return {
        "session_start": datetime.now().isoformat(),
        "user_input": user_input,
        "process_id": os.getpid(),
        "steps": {},
    }


def save_debug_log(debug_log: Dict[str, Any], stage: str) -> None:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"literature_debug_{timestamp}_{stage}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(debug_log, f, indent=2, ensure_ascii=False)
        print(f"🔍 Debug log saved: {filename}")
    except Exception as e:
        print(f"Warning: Failed to save debug log: {e}")
