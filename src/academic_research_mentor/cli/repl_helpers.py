from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..session_logging import SessionLogManager, set_active_session_logger
from ..chat_logger import ChatLogger
from ..core.stage_detector import detect_stage
from ..literature_review import build_research_context
from ..router import route_and_maybe_run_tool
from ..rich_formatter import print_info, print_error, print_formatted_response
from .resume import handle_resume_command


@dataclass
class CommandOutcome:
    handled: bool
    exit_command: Optional[str] = None


@dataclass
class ManualRoutingResult:
    consumed: bool
    enhanced_input: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


def create_session_stack(metadata: Optional[Dict[str, Any]] = None) -> Tuple[SessionLogManager, ChatLogger]:
    logger = SessionLogManager()
    set_active_session_logger(logger)
    for key, value in (metadata or {}).items():
        logger.attach_metadata(key, value)
    chat_logger = ChatLogger(session_logger=logger)
    return logger, chat_logger


def safe_detect_stage(user: str, chat_logger: ChatLogger, session_logger: SessionLogManager) -> Optional[Dict[str, Any]]:
    try:
        stage = detect_stage(user)
        if hasattr(chat_logger, "set_pending_stage"):
            chat_logger.set_pending_stage(stage)
        session_logger.record_stage(stage)
        return stage
    except Exception as exc:
        session_logger.log_event("stage_detection_error", {"error": str(exc)})
        return None


def handle_system_command(user: str, agent: Any, session_logger: SessionLogManager, *, allow_resume: bool) -> CommandOutcome:
    lower_user = user.lower()

    if not lower_user:
        session_logger.log_event("input_ignored", {"reason": "empty"})
        return CommandOutcome(handled=True)

    if lower_user in {"/reset", "reset"}:
        session_logger.log_event("system_command", {"command": lower_user})
        if hasattr(agent, "reset_history"):
            try:
                agent.reset_history()  # type: ignore[attr-defined]
                print_info("Conversation memory cleared (in this session).")
                session_logger.log_event("system_command_result", {"command": lower_user, "status": "cleared"})
            except Exception as exc:  # pragma: no cover - defensive
                print_info("Failed to clear conversation memory.")
                session_logger.log_event("system_command_result", {"command": lower_user, "status": "error", "error": str(exc)})
        else:
            print_info("This agent does not support clearing memory in this mode.")
            session_logger.log_event("system_command_result", {"command": lower_user, "status": "unsupported"})
        return CommandOutcome(handled=True)

    if allow_resume and lower_user.startswith("/resume"):
        session_logger.log_event("system_command", {"command": "resume", "payload": user})
        handle_resume_command(agent, user)
        session_logger.log_event("system_command_result", {"command": "resume", "status": "handled"})
        return CommandOutcome(handled=True)

    if lower_user in {"exit", "quit", "/exit", "/quit"}:
        session_logger.log_event("system_command", {"command": lower_user})
        return CommandOutcome(handled=True, exit_command=lower_user)

    return CommandOutcome(handled=False)


def process_manual_turn(user: str, session_logger: SessionLogManager, *, enable_research_context: bool) -> ManualRoutingResult:
    session_logger.log_event("manual_routing", {})

    research_context: Dict[str, Any] = {}
    if enable_research_context:
        research_context = build_research_context(user)
        if research_context:
            session_logger.log_event("manual_routing_context", research_context)

    tool_called = route_and_maybe_run_tool(user)
    if tool_called:
        session_logger.log_event("manual_tool_invoked", tool_called)
        tool_name = tool_called.get("tool_name", "unknown")
        return ManualRoutingResult(consumed=True, enhanced_input=user, tool_calls=[{"tool_name": tool_name, "score": 3.0}])

    if enable_research_context and research_context.get("has_research_context", False):
        context_prompt = research_context.get("context_for_agent", "")
        enhanced = f"{context_prompt}\n\nUser Query: {user}"
        session_logger.log_event("input_enriched", {"mode": "manual_routing", "length": len(enhanced)})
        return ManualRoutingResult(consumed=False, enhanced_input=enhanced)

    return ManualRoutingResult(consumed=False, enhanced_input=user)


def build_react_enhanced_input(user: str, session_logger: SessionLogManager) -> str:
    try:
        from ..attachments import has_attachments as _has_att, search as _att_search
        from ..runtime.tool_impls import (
            guidelines_tool_fn as _guidelines_tool,  # type: ignore
            experiment_planner_tool_fn as _exp_plan,  # type: ignore
        )
        lower_q = user.lower()
        mentorship_triggers = [
            "novel", "novelty", "methodology", "publish", "publication",
            "problem selection", "career", "taste", "mentor", "guideline",
        ]
        literature_triggers = [
            "related work", "literature", "papers", "sota", "baseline",
            "survey", "prior work",
        ]
        experiment_triggers = [
            "experiment", "experiments", "hypothesis", "ablation",
            "evaluation plan", "setup", "metrics",
        ]
        wants_guidelines = any(k in lower_q for k in mentorship_triggers)
        wants_literature = any(k in lower_q for k in literature_triggers)
        wants_experiments = any(k in lower_q for k in experiment_triggers)

        if not _has_att():
            return user

        results = _att_search(user, k=6)
        if not results:
            return user

        lines: List[str] = ["Attached PDF context (top snippets):"]
        for r in results[:6]:
            file = r.get("file", "file.pdf")
            page = r.get("page", 1)
            text = (r.get("text", "") or "").strip().replace("\n", " ")
            if len(text) > 220:
                text = text[:220] + "…"
            lines.append(f"- [{file}:{page}] {text}")

        if wants_guidelines:
            try:
                gl = _guidelines_tool(user) or ""
                gl = str(gl).strip()
                if gl:
                    lines.append("")
                    lines.append("Mentorship guidelines context (summary):")
                    for ln in gl.splitlines()[:8]:
                        if ln.strip():
                            lines.append(ln.strip())
            except Exception as exc:  # pragma: no cover - best effort logging
                session_logger.log_event("guidelines_preview_error", {"error": str(exc)})

        if wants_literature:
            lines.append("")
            lines.append("Note: After grounding and mentorship guidance, consult literature_search to add 1–2 anchors.")

        if wants_experiments:
            try:
                plan = _exp_plan(user) or ""
                plan = str(plan).strip()
                if plan:
                    lines.append("")
                    lines.append("Experiment plan (preview):")
                    for ln in plan.splitlines()[:12]:
                        if ln.strip():
                            lines.append(ln.strip())
            except Exception as exc:  # pragma: no cover - best effort logging
                session_logger.log_event("experiment_preview_error", {"error": str(exc)})

        context_block = "\n".join(lines)
        enhanced = (
            f"{context_block}\n\n"
            f"Instruction: Ground your answer FIRST on the attached PDF context above when making claims; "
            f"include [file:page] citations. THEN, if it strengthens mentorship advice, incorporate insights from the "
            f"guidelines and literature context (summarize briefly and avoid over-citation).\n\n"
            f"User Question: {user}"
        )
        session_logger.log_event("input_enriched", {"mode": "react", "length": len(enhanced)})
        return enhanced
    except Exception as exc:  # pragma: no cover - defensive path
        session_logger.log_event("enrichment_error", {"error": str(exc)})
        return user


def run_agent_turn(
    agent: Any,
    user: str,
    enhanced_user_input: str,
    *,
    use_manual_routing: bool,
    chat_logger: ChatLogger,
    session_logger: SessionLogManager,
    turn_number: int,
) -> None:
    try:
        agent.print_response(enhanced_user_input, stream=True)  # type: ignore[attr-defined]
        return
    except Exception:
        try:
            reply = agent.run(enhanced_user_input)
            content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
            if use_manual_routing and not hasattr(agent, "set_chat_logger"):
                chat_logger.add_turn(user, [], content)
            else:
                session_logger.finalize_turn(
                    turn_number,
                    {
                        "user_prompt": user,
                        "tool_calls": [],
                        "ai_response": content,
                        "fallback": "agent.run",
                    },
                )
            print_formatted_response(content, "Mentor")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Mentor response failed: {exc}")
            session_logger.log_event("agent_failure", {"error": str(exc)})
            session_logger.finalize_turn(turn_number, {"user_prompt": user, "error": str(exc)})
