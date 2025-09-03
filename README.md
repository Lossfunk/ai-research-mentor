# Academic Research Mentor

This is an Academic Research Mentor application that provides AI-powered assistance for academic research tasks. It's built with Python using LangChain for LLM integration and Rich for enhanced console formatting.

## Project layout (WS1)

Key directories introduced to improve structure and maintainability:

- `src/academic_research_mentor/core/`
  - `orchestrator.py`: central coordinator (scaffold)
  - `transparency.py`: in-memory event/run store (scaffold)
  - `agent.py`: agent scaffold (non-functional placeholder)
- `src/academic_research_mentor/tools/`
  - `base_tool.py`: minimal tool interface (scaffold)
  - `__init__.py`: simple registry with register/get/list (scaffold)
  - `o3_search/`: O3 search tool scaffold (no behavior yet)

The existing CLI remains the entrypoint via `academic_research_mentor.cli:main`. A root-level `main.py` shim is also provided for convenience.

## Installation & Setup
```bash
# Install dependencies using uv (preferred)
uv sync

# Or install in development mode
pip install -e .

# Verify environment configuration after installation
academic-research-mentor --check-env
```

## Running the Application
```bash
# Run the main CLI (automatically loads .env file)
uv run academic-research-mentor

# With specific prompt variant
academic-research-mentor --prompt mentor
academic-research-mentor --prompt system

# Alternative: use the shimmed entrypoint
uv run python main.py

```

## Environment Configuration

Required API keys:
- `OPENROUTER_API_KEY` - **REQUIRED** for O3-powered literature review and model access
- `OPENAI_API_KEY` - Alternative for OpenAI GPT models (backup)
- `GOOGLE_API_KEY` - Alternative for Google Gemini models (backup)  
- `ANTHROPIC_API_KEY` - Alternative for Anthropic Claude models (backup)
- `MISTRAL_API_KEY` - Alternative for Mistral models (backup)

**Note**: OpenRouter API key is strongly recommended as it provides access to O3 for intelligent literature review.

Model selection environment variables:
- `OPENROUTER_MODEL` (default: "anthropic/claude-sonnet-4")
- `OPENAI_MODEL` (default: "gpt-4o-mini")
- `GEMINI_MODEL` (default: "gemini-2.5-flash-latest")
- `ANTHROPIC_MODEL` (default: "claude-3-5-sonnet-latest")
- `MISTRAL_MODEL` (default: "mistral-large-latest")

Agent behavior:
- `LC_AGENT_MODE`: "chat" (default), "react", or "router"
- `ARM_PROMPT`/`LC_PROMPT`: "mentor" or "system"

## Development

### Environment
- Conda env: `conda activate lossfunk`
- Use uv for tasks: `uv sync`, `uv run ...`

### Tests
```bash
uv run pytest -q
```

### Architecture docs
See `docs/architecture/` for ADRs and checklists:
- ADR-001: Directory structure and namespacing
- ADR-002: Tool interface and registry (scaffold)
- ADR-003: Orchestrator and agent integration (phased)
- tools-migration-checklist.md
