# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Academic Research Mentor application that provides AI-powered assistance for academic research tasks. It's built with Python using LangChain for LLM integration and Rich for enhanced console formatting. The application supports multiple LLM providers (OpenRouter, OpenAI, Google, Anthropic, Mistral) with O3-powered literature review capabilities.

## Development Environment

### Package Management
- **Primary tool**: `uv` (preferred)
- **Installation**: `uv sync` or `pip install -e .`
- **Python version**: >=3.11

### Environment Configuration
The application automatically loads environment variables from `.env` files from current or parent directories. Required configuration:

#### API Keys (At least one required)
- **OPENROUTER_API_KEY**: Strongly recommended for O3-powered literature review
- **OPENAI_API_KEY**: Alternative for OpenAI GPT models
- **GOOGLE_API_KEY**: Alternative for Google Gemini models
- **ANTHROPIC_API_KEY**: Alternative for Anthropic Claude models
- **MISTRAL_API_KEY**: Alternative for Mistral models

#### Agent Configuration
- **LC_AGENT_MODE**: Controls agent behavior (default: "react", options: "chat", "react", "router")
- **ARM_PROMPT`/`LC_PROMPT`: Prompt variant (default: "mentor", options: "mentor", "system")
- **OPENROUTER_MODEL**: Default model for OpenRouter (default: "anthropic/claude-sonnet-4")

#### Guidelines Engine Configuration
- **ARM_GUIDELINES_MODE**: Guidelines integration mode (**default: "dynamic"**, options: "off", "static", "dynamic")
  - **dynamic**: Uses Guidelines Tool for real-time guideline retrieval (preferred)
  - **static**: Uses cached/unified_guidelines.json (legacy)
  - **off**: Disables guidelines injection
- **ARM_GUIDELINES_MAX**: Maximum number of guidelines to include (optional)
- **ARM_GUIDELINES_CATEGORIES**: Comma-separated filter for guideline categories (optional)
- **ARM_GUIDELINES_FORMAT**: Format style (default: "comprehensive", options: "concise")
- **ARM_GUIDELINES_INCLUDE_STATS**: Include guideline statistics (default: false)
- **ARM_GUIDELINES_PATH**: Custom guidelines file path (optional)

#### Reliability & Transparency Configuration
- **FF_TRANSPARENCY_PERSIST**: Enable JSON persistence of tool runs (default: false)
- **ARM_RUNLOG_DIR**: Directory for storing run logs (default: ~/.cache/academic-research-mentor/runs/)
- **ARM_DEBUG_ENV**: Enable environment loading debug output (default: false)

#### Environment Gating
Many features are controlled by environment variables to enable gradual rollout and debugging:
- Use environment variables for feature flags and configuration
- Graceful degradation when optional features are disabled
- Debug mode available for troubleshooting environment issues

### Testing
```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest tests/test_specific_file.py -q

# Run with verbose output
uv run pytest -v
```

### Running the Application
```bash
# Main CLI entrypoint
uv run academic-research-mentor

# Alternative via main.py shim
uv run python main.py

# Check environment configuration
uv run academic-research-mentor --check-env

# List available tools
uv run academic-research-mentor --list-tools
```

## Architecture

### Core Components

1. **CLI Entry Point**: `src/academic_research_mentor/cli.py:main`
   - Handles argument parsing, environment loading, and main REPL loop
   - Supports different agent modes (chat, react, router)
   - Integrates with tool registry and orchestrator

2. **Agent Runtime**: `src/academic_research_mentor/runtime.py`
   - Provides `_LangChainAgentWrapper` for consistent agent interface
   - Handles streaming responses and conversation history
   - Supports multiple LLM providers through LangChain

3. **Core Orchestrator**: `src/academic_research_mentor/core/orchestrator.py`
   - Coordinates tool selection and execution
   - Implements intelligent fallback policies
   - Provides task-based execution with circuit breakers

4. **Tool Registry**: `src/academic_research_mentor/tools/__init__.py`
   - Manages tool registration and discovery
   - Provides `BaseTool` interface for consistent tool implementation
   - Auto-discovers tools in subpackages

### Tool System

**Base Tool Interface**: `src/academic_research_mentor/tools/base_tool.py`
- Standard interface for all tools with lifecycle methods
- Required methods: `execute()`, `can_handle()`, `get_metadata()`
- Optional methods: `initialize()`, `cleanup()`

**Tool Structure**:
```
tools/
├── base_tool.py          # BaseTool interface
├── __init__.py          # Registry and auto-discovery
├── guidelines/          # Guidelines injection tools
├── legacy/              # Legacy tools (arXiv, etc.)
├── o3_search/           # O3 search implementation
├── searchthearxiv/      # arXiv search tools
└── utils/               # Shared utilities
```

### Agent Modes

1. **React Mode** (`LC_AGENT_MODE=react`):
   - Agent decides which tools to use automatically
   - Uses LangChain's agent framework
   - Supports tool calling and streaming

2. **Chat Mode** (`LC_AGENT_MODE=chat`):
   - Manual tool routing via `route_and_maybe_run_tool()`
   - Conversational approach with research context building
   - Fallback to simple responses when tools aren't needed

### Guidelines Engine

Located in `src/academic_research_mentor/guidelines_engine/`:
- **Dynamic Mode**: Default behavior using Guidelines Tool for real-time guideline retrieval
- **Static Mode**: Legacy support for unified_guidelines.json file
- **Configuration**: Environment-controlled behavior and filtering
- **Graceful Degradation**: Continues operation when guidelines are unavailable
- **Categories**: Supports filtering guidelines by research categories

#### Key Features
- **Default Dynamic**: Fresh clones prefer dynamic mode over static files
- **Configurable**: Environment variables control mode, categories, and formatting
- **Extensible**: Easy to add new guideline sources and formats
- **Performance**: Caching and optimization for frequent guideline access

### Transparency Layer

`src/academic_research_mentor/core/transparency.py`:
- **Event Model**: ToolRun and ToolEvent entities for comprehensive execution tracking
- **In-Memory Store**: Process-wide transparency store with global singleton access
- **Optional Persistence**: JSON-based persistence controlled by `FF_TRANSPARENCY_PERSIST`
- **Real-time Streaming**: In-process pub/sub system for live event broadcasting
- **ReAct Integration**: Streaming support for agent execution with incremental updates
- **Export Capabilities**: JSON export for external analysis and debugging

#### Key Features
- **Automatic Tracking**: Tool executions automatically logged with metadata
- **Health Monitoring**: Tool state, backoff status, and performance metrics
- **Debugging Support**: Rich metadata for troubleshooting and optimization
- **Extensible**: Pluggable backend architecture for future storage options

## Key Development Patterns

### Code Organization
- **File size limit**: Keep files under 200 lines of code
- **Single responsibility**: Each file has one clear purpose
- **Scaffolding approach**: Minimal implementations first, then incremental enhancement

### Tool Development
1. Create tool class inheriting from `BaseTool`
2. Implement required methods (`execute`, `can_handle`, `get_metadata`)
3. Place in appropriate subdirectory under `tools/`
4. Tool will be auto-discovered via registry

### Testing Patterns
- Tests located in `tests/` directory
- `conftest.py` ensures src/ is importable during testing
- Focus on scaffolding and core functionality tests
- Integration tests for tool registry and orchestrator

### Environment Handling
- Automatic `.env` file loading from current or parent directories
- Graceful fallback when environment variables are missing
- Debug mode via `ARM_DEBUG_ENV=1`

### Error Handling
- **Graceful Degradation**: Tools fail gracefully without breaking user experience
- **Circuit Breaker Patterns**: Prevent cascading failures with automatic recovery
- **Comprehensive Logging**: Rich transparency data for debugging and monitoring
- **Retry Logic**: Intelligent retry with exponential backoff for transient failures

### WS6 Reliability Enhancements

#### O3 Search Reliability
- **Timeout Protection**: 15-second timeout using signal.alarm() to prevent hanging
- **Automatic Fallback**: Falls back to arXiv search when O3 is unavailable
- **Degraded Mode Notes**: Clear user messaging when fallback occurs
- **Exception Handling**: Comprehensive exception handling for various failure scenarios

#### Backoff & Circuit Breaker System
- **Per-Tool Backoff Counters**: Track consecutive failures for each tool
- **Exponential Backoff**: 5s, 10s, 20s, 40s, 60s (capped) delays
- **Recovery Mechanism**: 3 consecutive successes reset backoff counters
- **Circuit Breaker Integration**: Coordinates backoff with circuit breaker states
- **Health States**: HEALTHY, DEGRADED, CIRCUIT_OPEN with automatic transitions

#### Transparency Enhancements
- **Real-time Status**: Live display of tool health and backoff status
- **Metadata Enrichment**: Execution context includes tool state and backoff information
- **Monitoring**: Comprehensive tool health metrics for operational awareness
- **Debug Support**: Rich metadata for troubleshooting reliability issues

#### Testing Coverage
- **O3 Fallback Tests**: Timeout and exception scenarios with fallback verification
- **Backoff Policy Tests**: Counter logic, circuit breaker integration, health summaries
- **Recommender Tests**: Scoring behavior under degraded tool conditions
- **Integration Tests**: End-to-end reliability with transparency verification

## Common Commands

### Development Workflow
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q

# Run application
uv run academic-research-mentor

# Check environment
uv run academic-research-mentor --check-env

# List tools
uv run academic-research-mentor --list-tools

# Show tool candidates for a goal
uv run academic-research-mentor --show-candidates "find recent papers on LLMs"

# Get tool recommendation
uv run academic-research-mentor --recommend "search academic literature"
```

### Debugging
```bash
# Enable debug output for environment loading
ARM_DEBUG_ENV=1 uv run academic-research-mentor

# Show recent tool runs (requires FF_TRANSPARENCY_PERSIST=true)
uv run academic-research-mentor --show-runs

# Test specific tool registration
python -c "from src.academic_research_mentor.tools import auto_discover; auto_discover(); print(list_tools().keys())"

# Test reliability features
uv run pytest tests/test_o3_fallback_reliability.py tests/test_fallback_policy_backoff.py -q

# Check transparency store status
python -c "from src.academic_research_mentor.core.transparency import get_transparency_store; store = get_transparency_store(); print(f'Runs in memory: {len(store.list_runs())}')"
```

## Important Notes

- The project uses **uv** as the primary package management tool
- **OpenRouter API key** is strongly recommended for O3-powered literature review
- The tool registry system uses **auto-discovery** - tools are automatically registered when placed in the correct directory structure
- **Agent mode** significantly changes behavior - understand the differences between "chat", "react", and "router" modes
- **Guidelines engine** defaults to dynamic mode, gracefully degrading when unavailable
- **Environment gating** controls many features - use environment variables for configuration and debugging
- **Reliability features** include timeout protection, backoff counters, and circuit breakers for robust operation
- All tools should implement the **BaseTool interface** for consistency
- **Transparency system** provides comprehensive monitoring and debugging capabilities
- **WS6 enhancements** significantly improve system reliability and user experience