from __future__ import annotations

import json
from typing import Any, Dict


class _DummyTavilyClient:
    def __init__(self, response: Dict[str, Any], *, should_raise: bool = False) -> None:
        self.response = response
        self.should_raise = should_raise
        self.calls: list[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("tavily failure")
        return self.response


def test_web_search_returns_structured_results() -> None:
    from academic_research_mentor.tools.web_search.tool import WebSearchTool

    response = {
        "answer": "AI regulation is evolving rapidly with new policy drafts in the EU.",
        "results": [
            {
                "title": "EU unveils new AI regulation",
                "url": "https://example.com/eu-ai-regulation",
                "content": "The European Union released a draft update to its AI Act...",
                "source": "news",
                "score": 0.92,
            }
        ],
    }

    client = _DummyTavilyClient(response)
    tool = WebSearchTool()
    tool.initialize({"client": client})

    result = tool.execute({"query": "latest ai regulation", "limit": 5})

    assert result["results"], "Expected at least one result"
    top = result["results"][0]
    assert top["title"] == "EU unveils new AI regulation"
    assert top["url"].startswith("https://example.com")
    assert "citations" in result and result["citations"]["count"] == 1
    assert result.get("summary") == response["answer"]
    assert client.calls, "Expected Tavily client to be invoked"


def test_web_search_handles_missing_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from academic_research_mentor.tools.web_search.tool import WebSearchTool

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    tool = WebSearchTool()
    tool.initialize()

    result = tool.execute({"query": "test query"})

    assert result["results"] == []
    assert result.get("_degraded_mode") is True
    note = result.get("note", "")
    assert "Web search unavailable" in note
    assert "Tavily" in note and "OpenRouter" in note


def test_web_search_handles_client_exception(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from academic_research_mentor.tools.web_search.tool import WebSearchTool

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = _DummyTavilyClient({}, should_raise=True)
    tool = WebSearchTool()
    tool.initialize({"client": client})

    result = tool.execute({"query": "ai research trends"})

    assert result["results"] == []
    assert result.get("_degraded_mode") is True
    note = result.get("note", "")
    assert "Web search unavailable" in note
    assert "Tavily" in note


def test_web_search_fallbacks_to_openrouter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from academic_research_mentor.tools.web_search.tool import WebSearchTool

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Latest AI policy updates from trusted sources.",
                            "results": [
                                {
                                    "title": "Policy board approves AI charter",
                                    "url": "https://example.org/ai-charter",
                                    "snippet": "Summary of the newly approved AI governance charter.",
                                    "source": "example.org",
                                }
                            ],
                        }
                    )
                }
            }
        ]
    }

    class _DummyResponse:
        status_code = 200

        def json(self):
            return response_payload

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.requests: list[Dict[str, Any]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: Dict[str, Any], json: Dict[str, Any]) -> _DummyResponse:
            self.requests.append({"url": url, "headers": headers, "json": json})
            return _DummyResponse()

    monkeypatch.setattr(
        "academic_research_mentor.tools.web_search.providers.httpx.Client",
        _DummyClient,
        raising=False,
    )

    tool = WebSearchTool()
    tool.initialize()

    result = tool.execute({"query": "latest ai policy", "limit": 3})

    assert result["results"], "Fallback should yield results"
    assert result["metadata"]["provider"] == "openrouter-web"
    assert "OpenRouter" in result["note"]
    assert "citations" in result and result["citations"]["count"] == 1


def test_web_search_reports_available_with_openrouter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from academic_research_mentor.tools.web_search.tool import WebSearchTool

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    tool = WebSearchTool()
    tool.initialize()

    assert tool.is_available() is True