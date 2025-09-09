# Scratchpad

Purpose: Track decisions and context while executing WS1 and WS2.

Context:
- Current package layout uses `src/academic_research_mentor` with CLI entrypoint via `pyproject.toml` (script: `academic-research-mentor`).
- We'll avoid breaking the existing CLI (`uv run academic-research-mentor`).
- We will introduce `core/` and `tools/` under the package namespace `src/academic_research_mentor/` instead of top-level modules to preserve imports.
- Keep files concise (<200 LOC each).

WS1 Decisions & Actions (completed):
- Created `src/academic_research_mentor/core/` with `orchestrator.py`, `transparency.py`, `agent.py`, and `__init__.py` exports.
- Created `src/academic_research_mentor/tools/` with `base_tool.py`, `__init__.py` (registry), and `utils/`.
- Added root-level `main.py` shim and updated `pyproject.toml` script to `academic_research_mentor.cli:main`.
- Wrote ADR-001/002/003 and tools migration checklist; updated README with new layout.
- Kept runtime behavior unchanged; CLI sanity checks pass.

WS2 Decisions & Actions (in progress):
- Extended `BaseTool` with lifecycle (`initialize`, `cleanup`), `can_handle`, and `get_metadata`.
- Implemented registry `auto_discover()` with validation; added tests.

## SearchTheArXiv Tool Implementation (Completed)

### Overview
Successfully implemented searchthearxiv.com as a standalone literature search tool that complements the existing O3 search tool. The tool specializes in semantic/natural language search for arXiv papers.

### Implementation Details

#### Tool Structure
- **Location**: `src/academic_research_mentor/tools/searchthearxiv/`
- **Files**: `__init__.py`, `tool.py`
- **Dependencies**: Added `beautifulsoup4>=4.12.0` to pyproject.toml

#### Key Features
- **Semantic Search**: Uses natural language processing to understand conversational queries
- **Web Scraping**: Implements BeautifulSoup-based scraping of searchthearxiv.com
- **Error Handling**: Robust retry logic and graceful degradation
- **Auto-discovery**: Automatically registered via tool registry system

#### Tool Selection Logic
- **Natural Language Queries**: Gets priority (score 1.4+) for queries like "find papers about..."
- **Explicit arXiv Searches**: Lower priority (score 1.1) for queries like "arxiv search transformer"
- **Keyword Queries**: O3 search handles traditional keyword searches

#### Scoring System Updates
- Added `SEMANTIC_SEARCH_NAMES = {"searchthearxiv_search"}` to recommendation engine
- Implemented intelligent scoring based on query type:
  - Natural language: searchthearxiv_search (1.4)
  - Explicit arXiv: o3_search (1.2)
  - Keywords: o3_search (0.8)

#### Integration Status
- ✅ Tool implementation complete
- ✅ Auto-discovery working
- ✅ Recommendation engine updated
- ✅ Tests updated and passing
- ✅ Dependencies installed
- ⚠️ Service Issues: searchthearxiv.com currently experiencing 500 errors

### Testing Results
All tests pass with updated expectations:
- Natural language queries correctly select searchthearxiv_search
- Explicit arXiv queries correctly select o3_search
- Legacy tools properly deprioritized
- Guidelines tool maintains priority for mentorship queries

### Current Limitations
- searchthearxiv.com search endpoint returning 500 errors
- Tool implemented but cannot be fully tested until service is restored
- Reliability score lowered to 0.4 due to service issues

### Future Improvements
- Monitor searchthearxiv.com service status
- Consider fallback to O3 search when service is unavailable
- Add more sophisticated HTML parsing when service is restored
- Implement caching for search results

### Refactoring Complete (200 LOC Compliance)
Successfully broke down the original 393-line `tool.py` into focused modules:

#### Final File Structure:
- **`tool.py`** (191 lines): Main tool class with BaseTool interface
- **`scraper.py`** (80 lines): HTTP client and retry logic
- **`parser.py`** (160 lines): HTML parsing and result extraction
- **`__init__.py`** (4 lines): Package initialization

#### Benefits:
- ✅ All files under 200 LOC limit
- ✅ Clear separation of concerns
- ✅ Easier testing and maintenance
- ✅ Better code organization
- ✅ All functionality preserved
- ✅ Tests still passing
- Added `tools/o3_search/tool.py` with metadata and `can_handle`.
- Added `tools/legacy/arxiv/tool.py` wrapper for legacy arXiv search (auto-discovered as fallback).
- Added `core/bootstrap.py`; wired CLI to bootstrap registry behind `FF_REGISTRY_ENABLED`.
- Added CLI flags:
  - `--list-tools`: force discovery and list tools.
  - `--show-candidates "<goal>"`: run orchestrator selection-only and print candidates.
- Orchestrator now returns candidates, prioritizing `o3_search` (score 10) and treating `legacy_*` as fallback (score 0.5); reduces O3 score if client unavailable.
- Test suite expanded (5 tests), all passing under uv/conda.

Upcoming WS3 (not started):
Plan for WS3 (next incremental changes):
- Add recommendation scoring beyond static weights in `core/recommendation.py` (new, <200 LOC).
- Feature flag: `FF_AGENT_RECOMMENDATION` to gate orchestrator using the scorer.
- Scoring signals: prefer `o3_search`, penalize `legacy_*`, consider `can_handle`, metadata cost and reliability, basic domain keyword match.
- Orchestrator: call recommender when flag on; return candidates with rationales (no execution yet).
- CLI: add `--recommend "<goal>"` to print top tool and reasons.
- Tests: validate scoring orders `o3_search` > `legacy_arxiv_search`.

WS3 Decisions & Actions (completed):
- Implemented `core/recommendation.py` and wired orchestrator under `FF_AGENT_RECOMMENDATION`.
- Added CLI `--recommend` to print top tool and rationale.
- Added WS3 tests:
  - Recommender prefers o3_search over legacy.
  - Orchestrator uses recommender when FF_AGENT_RECOMMENDATION=1.
- Updated scratchpad.md with WS3 progress and next tasks.
- All tests pass (7 total).

WS3 Extension - Tool Integration (completed):
- Enhanced O3SearchTool: Implemented real execution combining arXiv + OpenReview searches.
- Extended Orchestrator: Added `execute_task()` method with tool execution + fallback logic.
- Integrated Context Builder: Modified literature search to use orchestrator-based tool selection when `FF_REGISTRY_ENABLED=1`.
- Maintained backward compatibility: Falls back to legacy search when orchestrator unavailable.
- End-to-end testing: CLI queries now use dynamic tool selection (o3_search preferred, score: 2.1).
- Real literature search results: Papers returned from actual searches via orchestrator.

WS3 Fallback Policy Enhancement (completed):
- Created `core/fallback_policy.py`: Circuit breaker pattern, retry logic, degraded modes (99 LOC).
- Created `core/execution_engine.py`: Tool execution with retries, keeping orchestrator under 200 LOC (95 LOC).
- Enhanced Orchestrator: Uses FallbackPolicy for intelligent execution strategies.
- Circuit breaker: Tools blocked after 3 failures, auto-recovery after 5min timeout.
- Retry logic: Exponential backoff (1s, 2s, 4s max), skip non-retryable errors.
- Fallback chain: Primary tool → retry → fallback tool → degraded mode.
- Health tracking: Tool states (healthy/degraded/circuit_open), failure counts.
- Tested scenarios: Normal execution, circuit breaker trigger, automatic fallback, recovery.
- All 7 tests still passing.

Next: WS4 (Transparency & Streaming) per TODO.md timeline, or additional tool ecosystem expansion.

Guidelines Tool Integration (Starting):
- Goal: Integrate complete guidelines-based mentor from agent-mentor-guidelines-test/ as a BaseTool
- Key Features to Preserve: Caching (24h TTL), cost monitoring, curated research sources (Hamming, LessWrong, etc.)
- Integration Strategy: Create tools/guidelines/ package following existing patterns
- Dependencies: LangChain tools (DuckDuckGoSearchRun), file-based caching, usage tracking
- Router Integration: Detect research methodology/advice/problem selection queries
- Agent Modes: Support chat (manual routing), react (auto tool selection), router (specialized routing)

Understanding from Codebase Analysis:
- BaseTool interface: name, version, initialize(), can_handle(), execute(), get_metadata()
- Tool registry: auto_discover() looks for subpackages with 'tool.py' containing BaseTool subclasses
- Existing patterns: o3_search/, legacy/arxiv/ show structure conventions
- Router patterns: regex-based detection in router.py for tool routing
- CLI integration: Environment vars, --check-env, --list-tools commands available

Implementation Plan - Phase 1 Step 1 (COMPLETED):
1. ✅ Read existing BaseTool implementations to understand patterns
2. ✅ Create tools/guidelines/ package structure  
3. ✅ Port configuration from test project (curated URLs, sources)
4. ✅ Implement basic GuidelinesTool class with BaseTool interface
5. ✅ Test auto-discovery and registration

Guidelines Tool Integration Progress:
- Created tools/guidelines/ package: __init__.py, config.py, tool.py
- Implemented GuidelinesTool as RAG-style retrieval component (not advice generator)
- Tool returns raw guidelines content with guide IDs for agent reasoning
- Enhanced recommendation.py with guidelines-specific keyword matching
- Improved can_handle() patterns to recognize PhD/career/mentoring queries
- Tool successfully discovered by auto_discover(): "research_guidelines"
- Scoring working: PhD career guidance -> research_guidelines (score=0.85)
- Research methodology -> o3_search preferred (score=2.70) but guidelines available (score=0.75)

Remaining Tasks (COMPLETED):
- ✅ Test actual tool execution with search functionality
- ✅ Add router integration for automatic detection in chat mode  
- ✅ Test end-to-end CLI integration
- ✅ Add caching and cost monitoring features

## Integration Results - Phase 1 Complete ✅

### Task 1: Test Actual Tool Execution with Search Functionality
- **Issue**: DuckDuckGo search was failing due to missing `ddgs` package
- **Solution**: Added `uv add ddgs` to install required dependency
- **Result**: Guidelines tool now successfully searches curated sources and retrieves relevant content

### Task 2: Add Router Integration for Automatic Detection in Chat Mode
- **Enhanced router.py** with guidelines query detection patterns:
  - Added `_run_guidelines_and_print()` function with orchestrator support
  - Added `_run_guidelines_fallback()` for direct tool usage when orchestrator unavailable
  - Implemented 9+ regex patterns for research methodology queries
  - Patterns include: "how to choose research problem", "methodology advice", "develop research taste", "phd guidance", etc.
- **Priority**: Guidelines queries checked before venue guidelines to ensure proper routing

### Task 3: Test End-to-End CLI Integration
- **Test Results**: All router patterns working correctly
  - "how to choose a good research problem" → 3 guidelines retrieved
  - "research methodology advice" → 3 guidelines retrieved  
  - "develop research taste and judgment" → 3 guidelines retrieved
  - "phd career guidance" → 3 guidelines retrieved
  - "what makes a good research project" → 3 guidelines retrieved
  - "effective research principles" → 3 guidelines retrieved
  - "guidelines for research methodology" → 3 guidelines retrieved
- **Success Rate**: 100% of test queries properly routed and returned relevant guidelines

### Task 4: Add Caching and Cost Monitoring Features
- **Created cache.py** with comprehensive caching system:
  - File-based caching with TTL support (24 hours by default)
  - Cache directory: `~/.cache/academic-research-mentor/guidelines/`
  - `GuidelinesCache` class for cache management
  - `CostTracker` class for usage statistics and cost estimation
- **Enhanced GuidelinesTool** with caching integration:
  - Cache checked before search execution
  - Cache hits/misses tracked automatically
  - Cost tracking: $0.01 per search query estimate
  - Added `get_cache_stats()` and `clear_cache()` methods
  - Updated metadata to include caching information
- **Test Results**: 
  - Cache working correctly (50% hit rate on repeated queries)
  - Cost tracking monitoring usage accurately
  - Cache stats properly persisted to disk

### Key Features Added:
- **Smart Caching**: 24-hour TTL with file-based storage
- **Cost Monitoring**: Tracks search queries, cache hits/misses, and estimated costs  
- **Router Integration**: Automatic detection of research methodology queries
- **Robust Fallbacks**: Graceful handling of search failures and orchestrator unavailability
- **Curated Sources**: Searches 20+ authoritative research guidance sources

### Test Results Summary:
- All 7 existing tests still pass
- Caching works correctly with proper hit/miss tracking
- Router successfully handles various research guidance query patterns
- Cost tracking monitors usage and provides statistics
- End-to-end integration tested and verified

### Dependencies Added:
- `ddgs>=9.5.5` - Required for DuckDuckGo search functionality

## Status: ✅ GUIDELINES TOOL INTEGRATION COMPLETE

The guidelines tool is now fully integrated and ready for production use. All remaining tasks from the integration plan have been completed successfully.

## Final Integration Results & Testing (Updated):

### Key Issue Resolution:
- **Problem**: Guidelines tool was being overridden by o3_search due to scoring system bias
- **Solution**: Fixed recommendation scoring in `core/recommendation.py` to prioritize guidelines tool for mentorship queries
- **Result**: Guidelines tool now gets scores 4.0-7.2 for mentorship queries vs o3_search scores 3.0-3.5

### Critical Configuration Fix:
- **Disabled unified guidelines**: Set `ARM_GUIDELINES_MODE=off` in `.env` to prevent interference from `unified_guidelines.json`
- **Agent mode**: Set `LC_AGENT_MODE=react` to enable agent-driven tool selection
- **Tool selection**: Agent now intelligently chooses between guidelines tool (mentorship) and o3_search (literature)

### Comprehensive Testing Results:
Tested with 4 research queries:
1. **AI Alignment Research Idea** - Agent engaged technically (no guidelines needed)
2. **Controversial ML Research Idea** - Agent provided critical analysis (no guidelines needed)
3. **Research Taste Development** - Guidelines tool activated (score: 7.2) with comprehensive mentorship advice
4. **PhD Problem Selection** - Guidelines tool activated (score: 4.9) with structured guidance

### Success Criteria Met:
- ✅ Agent-driven tool selection working properly
- ✅ Guidelines tool provides comprehensive, sourced advice with citations
- ✅ Tool scores appropriately prioritize guidelines for mentorship queries
- ✅ No more forced literature search - agent makes intelligent decisions
- ✅ Integration maintains backward compatibility

## Next Steps:
- Ready for WS4 (Transparency & Streaming) per TODO.md timeline
- Or continue with additional tool ecosystem expansion
- Consider adding more curated sources to guidelines configuration
- Monitor real-world usage and refine scoring as needed

## Unified Prompt System Update (COMPLETED):

### Task Summary:
- **Problem**: `prompts_loader.py` was still trying to load two separate prompts (mentor/system) from `prompt.md`
- **Solution**: Updated prompt loader to work with the new single comprehensive prompt system
- **Result**: Simplified prompt loading with unified approach

### Changes Made:

#### 1. Updated `prompts_loader.py`:
- **Removed variant selection logic**: No more mentor/system distinction
- **Simplified heading extraction**: Now looks for main heading `# Research Mentor System Prompt`
- **Extract complete content**: Gets all content after main heading instead of fenced code blocks
- **Updated return value**: Returns "unified" as loaded_variant instead of "mentor" or "system"

#### 2. Key Implementation Details:
- **Heading pattern**: Changed from `### Core Mentor Prompt` / `### Core System Prompt` to `# Research Mentor System Prompt`
- **Content extraction**: No longer looks for fenced code blocks - uses all content after main heading
- **Normalization**: Maintained whitespace normalization and ASCII normalization features
- **Guidelines injection**: Preserved dynamic guidelines injection capability

#### 3. Created Updated `CLAUDE.md`:
- **Documented unified prompt system**: Removed references to mentor/system variants
- **Updated configuration**: Removed `ARM_PROMPT` environment variable references
- **Updated commands**: Simplified CLI usage examples (no more `--prompt mentor/system`)
- **Added WS3.1 status**: Documented completion of unified prompt system

#### 4. Benefits of Unified System:
- **Simplified maintenance**: Single prompt file to manage
- **Better consistency**: All queries use the same comprehensive prompt
- **Easier testing**: No need to test multiple prompt variants
- **Improved documentation**: Clear, single source of truth for system behavior

### Testing Verification:
- Prompt loader successfully extracts unified content from `prompt.md`
- Maintains backward compatibility with existing guidelines injection
- No breaking changes to CLI or runtime behavior
- All existing functionality preserved

## Current System Status:
✅ **WS3.1 Complete**: Unified prompt system successfully implemented
✅ **WS3 Complete**: Guidelines tool integration with agent-driven selection
✅ **WS2 Complete**: Tool system with registry and orchestrator
✅ **WS1 Complete**: Core scaffolding and architecture

**Ready for WS4**: Transparency & Streaming features

Housekeeping (WS3):
- Removed redundant `core/fallback.py` in favor of `core/fallback_policy.py` + `core/execution_engine.py` per RULES.md (avoid duplication, keep files <200 LOC). All tests still pass.

Planned next (keep simple; no new env flags):
- Recommendation: boost `research_guidelines` for mentorship queries and `searchthearxiv_search` for natural language queries. Keep defaults simple; no flags.
- Transparency: record ToolRun start/end events by default (in-memory) during execution. View via `--show-runs`. No streaming yet.
- Tests: add unit for guidelines prioritization and for transparency run recording.

## WS4 – Transparency (ReAct mode) Progress

Status: Partial. Implemented CLI transparency for both chat and ReAct modes; persistence/streaming/UI still pending per TODO.

- What’s implemented
  - ReAct wrappers in `runtime.py` now log:
    - "Using tool: <name>" when a tool is invoked
    - "Found:" top items and "Sources:" URLs/domains
  - Covered tools: `o3_search`, `searchthearxiv_search`, `research_guidelines`, and legacy wrappers (arxiv/openreview)
  - In-memory transparency store records events (start, error, final_result, end); surfaced via `--show-runs`
  - Orchestrator + execution engine print tool selection, retries, and summary/sources

- How to verify
  - Set `FF_REGISTRY_ENABLED=true LC_AGENT_MODE=react`
  - Run: `uv run academic-research-mentor`
  - Example queries:
    - Literature: "I want to know about transformer architecture …" → logs from `o3_search` (and fallbacks)
    - Mentorship: "How do I develop research taste?" → logs from `research_guidelines`
  - Show recorded runs: `uv run academic-research-mentor --show-runs`

- Remaining WS4 items (not done)
  - Persistence backend for transparency (filesystem/sqlite) with retention
  - Event bus abstraction and real-time streaming (WebSocket/SSE)
  - UI panels: results list, execution timeline, live indicator
  - Export of run data (JSON/CSV) with redaction
  - Tests for persistence and stream coalescing/backpressure

Decision: Treat current logging and in-memory runs as WS4-MVP for CLI; defer persistence/streaming/UI to subsequent WS4 steps.
