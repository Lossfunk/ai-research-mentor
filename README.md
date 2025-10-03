# Academic Research Mentor

## Objective
Accelerate AI research with AI. We are building an AI Research Mentor that guides researchers through the entire research lifecycle so they can move from idea to published work faster.

## Key Capabilities
- Research-aware CLI powered by LangChain agents and dynamic tool routing.
- O3-backed literature search with graceful fallbacks and citation synthesis.
- Mentorship guidelines and experiment planning helpers to keep projects on track.
- File and PDF ingestion so the mentor can ground responses in user-provided material.
- Conversation logging with the ability to resume saved sessions from the CLI.

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

## Showcase Video
![Demo Video](Ghostty.mp4)

## Use Cases
- **Brainstorming and discussing a research proposal (no PDF):** Kick off an idea sprint, explore literature leads, and co-develop an initial plan directly in the CLI.
- **Reviewing a finished proposal (with PDF):** Attach the draft at startup so the mentor can critique structure, highlight gaps, and suggest revisions with citations.

## FAQ
1. **Can I use `pip` instead of `uv`?**  
   You can, but we recommend `uv` because it gives fine-grained control over Python versions and dependency resolution, which improves reproducibility. See a deeper comparison here: <https://blog.kusho.ai/uv-pip-killer-or-yet-another-package-manager/>

2. **When running the mentor I see `ModuleNotFoundError: No module named 'academic_research_mentor'`. How do I fix this?**  
   This is usually a path-resolution issue. Add the following to your `pyproject.toml`:
   ```toml
   [tool.setuptools.packages.find]
   where = ["./"]
   include = ["academic-research-mentor"]
   ```
   More background: <https://stackoverflow.com/questions/79340227/modulenotfounderror-when-installing-my-own-project>

3. **Can I attach a PDF?**  
   Yes. Launch the mentor with `--attach-pdf <path-to-pdf>`, for example: 
   ```bash
   uv run academic-research-mentor --attach-pdf abc.pdf
   ```

4. **Can I resume past conversations?**  
   Absolutely. Start the mentor, run `/resume`, and select the conversation you want to load. The turns will be restored into memory for the current session.

## Troubleshooting
- Ensure Python 3.11+ is installed.
- Re-run `uv sync` after dependency changes.
