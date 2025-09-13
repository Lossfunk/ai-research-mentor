from __future__ import annotations

import os
from typing import Any

from ..rich_formatter import print_formatted_response, print_info, print_error, get_formatter, print_user_input
from ..router import route_and_maybe_run_tool
from ..literature_review import build_research_context
from ..chat_logger import ChatLogger
from .session import cleanup_and_save_session


def online_repl(agent: Any, loaded_variant: str) -> None:
    chat_logger = ChatLogger()

    if hasattr(agent, 'set_chat_logger'):
        agent.set_chat_logger(chat_logger)

    agent_mode = os.environ.get("LC_AGENT_MODE", "react").strip().lower()
    use_manual_routing = agent_mode == "chat"

    formatter = get_formatter()
    formatter.print_rule("Academic Research Mentor")
    print_info(f"Loaded prompt variant: {loaded_variant}")
    print_info(f"Agent mode: {agent_mode}")
    print_info("Type 'exit' to quit")
    formatter.console.print("")

    try:
        while True:
            try:
                formatter.console.print("[bold cyan]You:[/bold cyan] ", end="")
                user = input().strip()
                print_user_input(user)
            except EOFError:
                print_info("\n📝 EOF received. Saving chat session...")
                cleanup_and_save_session(chat_logger, "EOF (Ctrl+D)")
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                cleanup_and_save_session(chat_logger, user)
                break

            if use_manual_routing:
                research_context = build_research_context(user)
                tool_called = route_and_maybe_run_tool(user)
                if tool_called:
                    tool_name = tool_called.get("tool_name", "unknown")
                    chat_logger.add_turn(user, [{"tool_name": tool_name, "score": 3.0}])
                    continue
                if research_context.get("has_research_context", False):
                    context_prompt = research_context.get("context_for_agent", "")
                    enhanced_user_input = f"{context_prompt}\n\nUser Query: {user}"
                else:
                    enhanced_user_input = user
            else:
                # For ReAct/default mode, enrich the user input with attached PDF context if available
                try:
                    from ..attachments import has_attachments as _has_att, search as _att_search
                    if _has_attachments := _has_att():
                        results = _att_search(user, k=6)
                        if results:
                            lines: list[str] = [
                                "Attached PDF context (top snippets):",
                            ]
                            for r in results[:6]:
                                file = r.get("file", "file.pdf")
                                page = r.get("page", 1)
                                text = (r.get("text", "") or "").strip().replace("\n", " ")
                                if len(text) > 220:
                                    text = text[:220] + "…"
                                lines.append(f"- [{file}:{page}] {text}")
                            context_block = "\n".join(lines)
                            enhanced_user_input = (
                                f"{context_block}\n\n"
                                f"Instruction: Ground your answer ONLY on the attached PDF context above when making claims; "
                                f"if insufficient, say so. Always include [file:page] citations.\n\n"
                                f"User Question: {user}"
                            )
                        else:
                            enhanced_user_input = user
                    else:
                        enhanced_user_input = user
                except Exception:
                    enhanced_user_input = user

            try:
                agent.print_response(enhanced_user_input, stream=True)  # type: ignore[attr-defined]
            except Exception:
                try:
                    reply = agent.run(enhanced_user_input)
                    content = getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
                    if use_manual_routing and not hasattr(agent, 'set_chat_logger'):
                        chat_logger.add_turn(user, [], content)
                    print_formatted_response(content, "Mentor")
                except Exception as exc:  # noqa: BLE001
                    print_error(f"Mentor response failed: {exc}")

            formatter.console.print("")
    finally:
        if not any(turn.get("user_prompt", "").lower() in {"exit", "quit", "eof (ctrl+d)"} for turn in chat_logger.current_session):
            cleanup_and_save_session(chat_logger, "unexpected_exit")


def offline_repl(reason: str) -> None:
    formatter = get_formatter()
    formatter.print_rule("Academic Research Mentor (Offline Mode)")
    print_info("Type 'exit' to quit")
    if reason:
        print_error(f"Offline reason: {reason}")
    print_info("Falling back to a simple echo mentor")
    formatter.console.print("")

    chat_logger = ChatLogger()

    try:
        while True:
            try:
                formatter.console.print("[bold cyan]You:[/bold cyan] ", end="")
                user = input().strip()
                print_user_input(user)
            except EOFError:
                print_info("\n📝 EOF received. Saving chat session...")
                cleanup_and_save_session(chat_logger, "EOF (Ctrl+D)")
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                cleanup_and_save_session(chat_logger, user)
                break

            tool_called = route_and_maybe_run_tool(user)
            if tool_called:
                tool_name = tool_called.get("tool_name", "unknown")
                chat_logger.add_turn(user, [{"tool_name": tool_name, "score": 3.0}])
                continue

            chat_logger.add_turn(user, [], "A few quick questions to calibrate: What is your goal, compute budget, and target venue? Then I can suggest next steps.")

            print_formatted_response(
                "A few quick questions to calibrate: What is your goal, compute budget, and target venue? Then I can suggest next steps.",
                "Mentor",
            )

            formatter.console.print("")
    finally:
        if not any(turn.get("user_prompt", "").lower() in {"exit", "quit", "eof (ctrl+d)"} for turn in chat_logger.current_session):
            cleanup_and_save_session(chat_logger, "unexpected_exit")
