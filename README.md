# Academic Research Mentor

This is an Academic Research Mentor application that provides AI-powered assistance for academic research tasks. It's built with Python using LangChain for LLM integration and Rich for enhanced console formatting, featuring O3-powered literature review with intelligent fallback mechanisms.

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Core Components
- **CLI Entry Point**: `src/academic_research_mentor/cli.py:main` - Handles argument parsing, environment loading, and REPL loop
- **Agent Runtime**: `src/academic_research_mentor/runtime.py` - Provides LangChain agent wrapper with streaming support
- **Core Orchestrator**: `src/academic_research_mentor/core/orchestrator.py` - Coordinates tool selection and execution with fallback policies
- **Transparency Layer**: `src/academic_research_mentor/core/transparency.py` - In-memory event store with optional persistence and streaming
- **Tool Registry**: `src/academic_research_mentor/tools/__init__.py` - Auto-discovery and management of research tools

### Tool System
- **Base Interface**: `src/academic_research_mentor/tools/base_tool.py` - Standard tool interface with lifecycle methods
- **Tool Categories**: Literature search (O3, arXiv), guidelines injection, and research utilities
- **Auto-Discovery**: Tools automatically registered when placed in appropriate subdirectories

### Guidelines Engine
- **Location**: `src/academic_research_mentor/guidelines_engine/` - Dynamic research mentorship guidelines injection
- **Default Mode**: Dynamic guidelines (preferred over static unified_guidelines.json)
- **Configurable**: Supports static, dynamic, and disabled modes via environment variables

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

# Alternative: use the shimmed entrypoint
uv run python main.py

```

## Environment Configuration

### Required API Keys
At least one of the following API keys is required:
- `OPENROUTER_API_KEY` - **Strongly recommended** for O3-powered literature review and model access
- `OPENAI_API_KEY` - Alternative for OpenAI GPT models
- `GOOGLE_API_KEY` - Alternative for Google Gemini models  
- `ANTHROPIC_API_KEY` - Alternative for Anthropic Claude models
- `MISTRAL_API_KEY` - Alternative for Mistral models

### Model Selection
- `OPENROUTER_MODEL` (default: "anthropic/claude-sonnet-4")
- `OPENAI_MODEL` (default: "gpt-4o-mini")
- `GEMINI_MODEL` (default: "gemini-2.5-flash-latest")
- `ANTHROPIC_MODEL` (default: "claude-3-5-sonnet-latest")
- `MISTRAL_MODEL` (default: "mistral-large-latest")

### Agent Configuration
- `LC_AGENT_MODE`: Controls agent behavior (**default: "react"**, options: "chat", "react", "router")
- `ARM_PROMPT`/`LC_PROMPT`: "mentor" or "system" prompt variant

### Guidelines Engine
- `ARM_GUIDELINES_MODE`: Guidelines integration mode (**default: "dynamic"**, options: "off", "static", "dynamic")
- `ARM_GUIDELINES_MAX`: Maximum guidelines count to include
- `ARM_GUIDELINES_CATEGORIES`: Comma-separated list of guideline categories
- `ARM_GUIDELINES_FORMAT`: Format style ("comprehensive", "concise")
- `ARM_GUIDELINES_INCLUDE_STATS`: Include guideline statistics (true/false)

### Reliability & Transparency
- `FF_TRANSPARENCY_PERSIST`: Enable JSON persistence of tool runs (true/false)
- `ARM_RUNLOG_DIR`: Directory for storing run logs (default: ~/.cache/academic-research-mentor/runs/)
- `ARM_DEBUG_ENV`: Enable environment loading debug output

## Development

### Environment Management
- **Package Manager**: Use `uv` for dependency management (`uv sync`, `uv run ...`)
- **Python Version**: >=3.11 required
- **Environment Loading**: Automatic `.env` file detection from current and parent directories

### Testing
```bash
# Run all tests with minimal output
uv run pytest -q

# Run with verbose output
uv run pytest -v

# Run specific test files
uv run pytest tests/test_specific_file.py -q

# Run reliability tests
uv run pytest tests/test_o3_fallback_reliability.py tests/test_fallback_policy_backoff.py -q
```

### Key Features

#### Reliability Enhancements (WS6)
- **O3 Timeout Protection**: 15-second timeout with automatic fallback to arXiv search
- **Backoff Counters**: Per-tool exponential backoff with recovery mechanisms
- **Circuit Breakers**: Prevent cascading failures with automatic recovery
- **Transparency**: Real-time tool health status and execution monitoring

#### Guidelines Engine
- **Dynamic Mode**: Default behavior using Guidelines Tool instead of static files
- **Configurable**: Environment-controlled guidelines injection
- **Graceful Degradation**: Continues operation when guidelines unavailable

#### Agent Modes
1. **React Mode** (default): Agent autonomously selects and uses tools via LangChain framework
2. **Chat Mode**: Manual tool routing with conversational context building
3. **Router Mode**: Advanced routing strategies (future enhancement)

### Architecture Documentation
See `docs/architecture/` for Architecture Decision Records (ADRs):
- **ADR-001**: Directory structure and namespacing
- **ADR-002**: Tool interface and registry
- **ADR-003**: Orchestrator and agent integration
- **ADR-004**: Event model, persistence, and ReAct streaming approach

### Common Commands
```bash
# Check environment configuration
uv run academic-research-mentor --check-env

# List available tools
uv run academic-research-mentor --list-tools

# Show tool candidates for a goal
uv run academic-research-mentor --show-candidates "find recent papers on LLMs"

# Get tool recommendation
uv run academic-research-mentor --recommend "search academic literature"

# Show recent tool runs (requires FF_TRANSPARENCY_PERSIST=true)
uv run academic-research-mentor --show-runs

# Debug environment loading
ARM_DEBUG_ENV=1 uv run academic-research-mentor
```
