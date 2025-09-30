# Academic Research Mentor

## Objective
Accelerate AI research with AI. We are building an AI Research Mentor that guides researchers through the entire research lifecycle so they can move from idea to published work faster.

## Key Capabilities
- Research-aware CLI powered by LangChain agents and dynamic tool routing.
- O3-backed literature search with graceful fallbacks and citation synthesis.
- Mentorship guidelines and experiment planning helpers to keep projects on track.

## Setup
```bash
# Install dependencies
uv sync

# Run tests (optional)
uv run pytest -q
```

## Environment
```bash
cp .example.env .env
```
Edit `.env` and add your `OPENROUTER_API_KEY` (recommended). Other provider keys are optional fallbacks.

## Usage
```bash
# Verify configuration
uv run academic-research-mentor --check-env

# Start the mentor CLI
uv run academic-research-mentor

# Alternate entrypoint
uv run python main.py
```

## Troubleshooting
- Ensure Python 3.11+ is installed.
- Re-run `uv sync` after dependency changes.
- Set additional API keys (OpenAI, Anthropic, etc.) if you prefer alternative models.
