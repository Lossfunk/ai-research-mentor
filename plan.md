# Integration Plan: Guidelines-Based Mentor Tool

## Overview

Integrate the complete guidelines-based mentor agent from `agent-mentor-guidelines-test/` into the main project as a specialized tool within the existing architecture.

## Current State Analysis

### Main Project (`academic-research-mentor/`)
- ✅ Scaffolded architecture with core/, tools/, orchestrator
- ✅ Tool registry system with BaseTool interface
- ✅ Multiple agent modes (chat, react, router)
- ✅ Existing mentor tools (arxiv, openreview, venue, math, methodology)
- ✅ CLI with environment setup and LLM provider support

### Guidelines Test Project (`agent-mentor-guidelines-test/`)
- ✅ Complete LangChain research mentor agent
- ✅ Guidelines search using curated research sources
- ✅ Response caching system with TTL
- ✅ Cost monitoring and usage tracking
- ✅ Curated research guidance URLs from experts (Hamming, LessWrong, etc.)

## Integration Strategy

### Phase 1: Core Guidelines Tool Creation

#### 1.1 Create Guidelines Tool Structure
```
src/academic_research_mentor/tools/guidelines/
├── __init__.py
├── tool.py                 # Main GuidelinesTool class
├── guidelines_search.py    # Search functionality
├── cache.py               # Response caching
├── cost_monitor.py        # Usage tracking
└── config.py             # Guidelines sources config
```

#### 1.2 Tool Implementation Approach
- **Extract Core Logic**: Move guidelines search from test project
- **BaseTool Compliance**: Implement using main project's BaseTool interface
- **Preserve Features**: Keep caching, cost monitoring, curated sources
- **LangChain Integration**: Maintain LangChain tool compatibility

### Phase 2: Dependencies and Configuration

#### 2.1 Dependencies Integration
- **Existing**: LangChain components already present in main project
- **New Requirements**: Add DuckDuckGoSearchRun (if not present)
- **Caching**: Use existing or minimal file-based cache
- **Cost Tracking**: Lightweight usage monitoring

#### 2.2 Environment Variables
Extend main project's environment configuration:
```env
# Guidelines Tool Settings
ARM_GUIDELINES_ENABLED=true
ARM_GUIDELINES_CACHE_TTL=24
ARM_GUIDELINES_MAX_QUERIES=3
ARM_GUIDELINES_COST_TRACKING=true
```

### Phase 3: Tool Registry Integration

#### 3.1 Auto-Discovery Compliance
- Place `tool.py` in `tools/guidelines/` for auto-discovery
- Implement `can_handle()` method for smart routing
- Register in tool registry during bootstrap

#### 3.2 Router Integration
Add guidelines detection patterns to router.py:
- Research methodology questions
- Academic advice queries
- Problem selection guidance
- Research taste development

### Phase 4: Implementation Steps

#### Step 1: Create Base Tool Structure (Week 1)
1. **Create guidelines tool package**
   ```bash
   mkdir -p src/academic_research_mentor/tools/guidelines
   ```

2. **Implement GuidelinesTool class**
   - Inherit from BaseTool
   - Define name, version, metadata
   - Implement can_handle() for research guidance detection
   - Implement execute() method

3. **Port guidelines search logic**
   - Extract search functionality from test project
   - Adapt to BaseTool execute() interface
   - Maintain curated source URLs

#### Step 2: Add Supporting Components (Week 1-2)
1. **Cache Implementation**
   - File-based cache with TTL support
   - Response deduplication
   - Statistics tracking

2. **Cost Monitor**
   - Track API usage and costs
   - Session-based statistics
   - Monthly usage estimates

3. **Configuration Management**
   - Guidelines sources mapping
   - Search query limits
   - Caching settings

#### Step 3: CLI Integration (Week 2)
1. **Environment Setup**
   - Add guidelines-specific env vars to cli.py
   - Include in --check-env command
   - Update --env-help documentation

2. **Router Patterns**
   - Add detection patterns for guidelines queries
   - Route to guidelines tool when appropriate
   - Fallback to general agent

3. **Agent Mode Support**
   - Works in chat mode (manual routing)
   - Available in react mode (automatic tool selection)
   - Integrated in router mode (specialized routing)

#### Step 4: Testing and Validation (Week 2-3)
1. **Unit Tests**
   - Tool discovery and registration
   - Guidelines search functionality
   - Cache behavior and TTL
   - Cost tracking accuracy

2. **Integration Tests**
   - CLI integration with guidelines tool
   - Agent mode compatibility
   - Router pattern matching

3. **End-to-End Testing**
   - Complete user workflows
   - Performance with caching
   - Cost monitoring accuracy

## Technical Details

### Tool Interface Implementation

```python
# tools/guidelines/tool.py
class GuidelinesTool(BaseTool):
    name = "research_guidelines"
    version = "1.0"
    
    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        # Detect research methodology, advice, problem selection queries
        
    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Search guidelines, apply caching, track costs, return formatted advice
        
    def get_metadata(self) -> Dict[str, Any]:
        # Describe capabilities, cost profile, ideal inputs
```

### Router Integration

```python
# router.py additions
def _is_guidelines_query(text: str) -> bool:
    patterns = [
        r'\b(research\s+methodology|problem\s+selection|research\s+taste)\b',
        r'\b(academic\s+advice|PhD\s+guidance|research\s+strategy)\b',
        r'\b(how\s+to\s+choose|develop\s+taste|research\s+skills)\b'
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def _run_guidelines_and_print(query: str) -> None:
    # Execute guidelines tool and format output
```

### Cost and Cache Management

```python
# Lightweight implementations that integrate with existing patterns
class GuidelinesCache:
    def get(self, key: str) -> Optional[Dict]
    def set(self, key: str, value: Dict, ttl_hours: int = 24)
    def get_stats() -> Dict

class CostMonitor:
    def track_query(self, cost: float, tokens: int)
    def get_summary() -> Dict
```

## Migration Path

### Data Migration
- **Guidelines URLs**: Copy curated sources from test project config
- **Cache Data**: Optional migration of valid cache entries
- **Cost History**: Preserve existing usage statistics

### Feature Preservation
- **All Guidelines Sources**: Maintain complete curated URL list
- **Search Quality**: Preserve search strategy and result formatting
- **Caching Behavior**: Keep TTL-based caching with hit rate tracking
- **Cost Tracking**: Maintain usage monitoring and estimation

### Backward Compatibility
- **Existing Tools**: No changes to arxiv, openreview, venue tools
- **CLI Interface**: No breaking changes to existing commands
- **Agent Modes**: All current modes continue to work
- **Environment**: Existing .env configurations remain valid

## Success Criteria

### Functional Requirements
- ✅ Guidelines tool discoverable via auto-discovery
- ✅ Integrates with all agent modes (chat, react, router)
- ✅ Preserves caching and cost monitoring features
- ✅ Maintains search quality from test project
- ✅ CLI integration with environment configuration

### Performance Requirements
- ✅ Response caching reduces API costs by >70%
- ✅ Guidelines search completes within 10 seconds
- ✅ Cache hit rate >40% for common queries
- ✅ Tool selection accuracy >90% for guidelines queries

### Quality Requirements
- ✅ Comprehensive unit and integration test coverage
- ✅ Follows RULES.md (files <200 lines, incremental changes)
- ✅ Proper error handling and graceful degradation
- ✅ Clear documentation and usage examples

## Timeline

- **Week 1**: Core tool implementation and basic integration
- **Week 2**: CLI integration, router patterns, cost/cache systems
- **Week 3**: Testing, validation, documentation, refinement

## Risk Mitigation

### Technical Risks
- **Dependency Conflicts**: Test compatibility with existing LangChain versions
- **Performance Impact**: Monitor tool discovery and routing overhead
- **Cache Storage**: Implement size limits and cleanup strategies

### Integration Risks
- **Tool Registry**: Ensure auto-discovery works reliably
- **Agent Compatibility**: Test with all agent modes thoroughly
- **Router Conflicts**: Prevent pattern conflicts with existing tools

### Operational Risks
- **API Costs**: Implement proper cost limits and monitoring
- **Cache Management**: Handle cache corruption and cleanup
- **Configuration**: Provide clear environment variable documentation

## Next Steps

1. **Create initial tool structure** following the scaffolding patterns
2. **Port core guidelines search functionality** from test project
3. **Implement caching and cost monitoring** as separate modules
4. **Add router integration** for automatic detection
5. **Test end-to-end workflows** with all agent modes
6. **Document configuration and usage** patterns

This integration transforms the standalone guidelines agent into a first-class tool within the existing architecture while preserving all its advanced features.
