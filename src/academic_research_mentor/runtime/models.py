from __future__ import annotations

import os
from typing import Any, Optional, Tuple


def _import_openrouter_model() -> Optional[Any]:
    """Lazily import the OpenRouter-compatible LangChain client."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI
    except Exception:  # pragma: no cover
        return None


def _resolve_openrouter_max_tokens(model_id: str) -> Optional[int]:
    """Return a safe max_tokens value for OpenRouter models with tighter limits."""
    model_overrides = {
        "moonshotai/kimi-k2": 160_000,
    }
    return model_overrides.get(model_id)


def resolve_model() -> Tuple[Optional[Any], Optional[str]]:
    ChatOpenAI = _import_openrouter_model()
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None, "OpenRouter API key not configured. Set OPENROUTER_API_KEY in your environment."
        if ChatOpenAI is None:
            return None, "langchain-openai package unavailable; cannot initialize OpenRouter client."

        model_id = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        max_tokens = _resolve_openrouter_max_tokens(model_id)
        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": 0,
        }
        use_responses_env = os.environ.get("OPENROUTER_USE_RESPONSES_API")
        if use_responses_env is not None:
            llm_kwargs["use_responses_api"] = use_responses_env.lower() in {"1", "true", "yes"}
        else:
            llm_kwargs["use_responses_api"] = False
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        llm = ChatOpenAI(**llm_kwargs)
        return llm, None

    except Exception as exc:  # pragma: no cover
        return None, f"Model init failed: {exc}"

    return None, "OpenRouter model initialization failed for an unknown reason."