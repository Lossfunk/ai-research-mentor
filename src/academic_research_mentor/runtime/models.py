from __future__ import annotations

import os
from typing import Any, Optional, Tuple


def _import_langchain_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Lazily import LangChain chat model classes for major providers."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatOpenAI = None  # type: ignore
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatGoogleGenerativeAI = None  # type: ignore
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except Exception:  # pragma: no cover
        ChatAnthropic = None  # type: ignore
    try:
        from langchain_mistralai import ChatMistralAI  # type: ignore
    except Exception:  # pragma: no cover
        ChatMistralAI = None  # type: ignore
    return ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic, ChatMistralAI


def resolve_model() -> Tuple[Optional[Any], Optional[str]]:
    ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic, ChatMistralAI = _import_langchain_models()
    try:
        # Prefer OpenRouter when available (OpenAI-compatible API)
        if os.environ.get("OPENROUTER_API_KEY") and ChatOpenAI is not None:
            model_id = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            llm = ChatOpenAI(
                model=model_id,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=base_url,
                temperature=0,
            )
            return llm, None

        # OpenAI
        if os.environ.get("OPENAI_API_KEY") and ChatOpenAI is not None:
            model_id = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            # Respect optional OPENAI_BASE_URL if provided
            base_url = os.environ.get("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {"model": model_id, "temperature": 0}
            if base_url:
                kwargs["base_url"] = base_url
            llm = ChatOpenAI(**kwargs)
            return llm, None

        # Google Gemini
        if os.environ.get("GOOGLE_API_KEY") and ChatGoogleGenerativeAI is not None:
            model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-latest")
            llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0)
            return llm, None

        # Anthropic
        if os.environ.get("ANTHROPIC_API_KEY") and ChatAnthropic is not None:
            model_id = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            llm = ChatAnthropic(model=model_id, api_key=os.environ.get("ANTHROPIC_API_KEY"), temperature=0)
            return llm, None

        # Mistral
        if os.environ.get("MISTRAL_API_KEY") and ChatMistralAI is not None:
            model_id = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
            llm = ChatMistralAI(model=model_id, api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0)
            return llm, None

        return None, "No supported model API key found (set OPENROUTER_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, or MISTRAL_API_KEY)"
    except Exception as exc:  # pragma: no cover
        return None, f"Model init failed: {exc}"
