# Academic Research Mentor

This is an Academic Research Mentor application that provides AI-powered assistance for academic research tasks. It's built with Python using LangChain for LLM integration and Rich for enhanced console formatting.

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

# ASCII-normalized prompts
academic-research-mentor --ascii

# Verify environment setup
academic-research-mentor --check-env

# Show environment configuration help
academic-research-mentor --env-help

# Enable debug mode for .env loading
ARM_DEBUG_ENV=1 academic-research-mentor
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
- `ARM_PROMPT_ASCII`/`LC_PROMPT_ASCII`: ASCII normalization flag

Debug options:
- `ARM_DEBUG_ENV`: Set to "1" to show .env file loading debug information
