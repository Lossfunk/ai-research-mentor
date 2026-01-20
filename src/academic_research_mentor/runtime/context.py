from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..agent import MentorAgent, ToolRegistry, create_default_tools
from ..llm import create_client
from ..prompts_loader import load_instructions_from_prompt_md


@dataclass
class PreparedAgent:
    agent: Optional[Any]
    loaded_variant: str
    offline_reason: Optional[str]


class LegacyAgentAdapter:
    def __init__(self, agent: MentorAgent) -> None:
        self._agent = agent
        self._chat_logger: Any = None
        self._session_logger: Any = None

    def run(self, user_message: Any, **_: Any) -> str:
        return self._agent.chat(user_message)

    def print_response(self, user_message: Any, stream: bool = False, **_: Any) -> None:
        from ..rich_formatter import print_formatted_response

        response = self._agent.chat(user_message)
        print_formatted_response(response, "Mentor")

    def reset_history(self) -> None:
        self._agent.clear_history()

    def set_chat_logger(self, logger: Any) -> None:
        self._chat_logger = logger

    def set_session_logger(self, logger: Any) -> None:
        self._session_logger = logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


def _resolve_prompt_variant(prompt_arg: Optional[str]) -> str:
    if prompt_arg:
        return prompt_arg.strip().lower()
    return (
        os.environ.get("ARM_PROMPT")
        or os.environ.get("LC_PROMPT")
        or os.environ.get("AGNO_PROMPT")
        or "mentor"
    ).strip().lower()


def _resolve_ascii_normalize(ascii_override: Optional[bool]) -> bool:
    if ascii_override is not None:
        return ascii_override
    return bool(
        os.environ.get("ARM_PROMPT_ASCII")
        or os.environ.get("LC_PROMPT_ASCII")
        or os.environ.get("AGNO_PROMPT_ASCII")
    )


def _build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    whitelist_env = os.environ.get("ARM_TOOL_WHITELIST", "")
    whitelist = {name.strip() for name in whitelist_env.split(",") if name.strip()} if whitelist_env else None
    guidelines_off = os.environ.get("ARM_GUIDELINES_MODE", "").strip().lower() == "off"

    for tool in create_default_tools():
        if whitelist and tool.name not in whitelist:
            continue
        if guidelines_off and tool.name == "research_guidelines":
            continue
        registry.register(tool)
    return registry


def _resolve_provider() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "openrouter"


def prepare_agent(
    *,
    prompt_arg: Optional[str] = None,
    ascii_override: Optional[bool] = None,
) -> PreparedAgent:
    baseline_mode = os.environ.get("ARM_BASELINE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if baseline_mode:
        instructions = "You are a helpful research assistant."
        loaded_variant = "baseline"
    else:
        prompt_variant = _resolve_prompt_variant(prompt_arg)
        ascii_normalize = _resolve_ascii_normalize(ascii_override)

        instructions, loaded_variant = load_instructions_from_prompt_md(prompt_variant, ascii_normalize)
        if not instructions:
            instructions = "You are a helpful research mentor."
            loaded_variant = "fallback"

    try:
        provider = _resolve_provider()
        client = create_client(provider=provider)
        tools = _build_tool_registry()
        agent = MentorAgent(
            system_prompt=instructions,
            client=client,
            tools=tools if len(tools) > 0 else None,
        )
    except Exception as exc:  # noqa: BLE001
        return PreparedAgent(agent=None, loaded_variant=loaded_variant, offline_reason=str(exc))

    return PreparedAgent(
        agent=LegacyAgentAdapter(agent),
        loaded_variant=loaded_variant,
        offline_reason=None,
    )
