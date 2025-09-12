# ADR-004: Event Model, Persistence, and ReAct Streaming Approach

## Status
Accepted

## Context
As part of WS4 (Transparency & Storage & Streaming), we needed to implement a comprehensive event system that could track tool executions, support real-time streaming for ReAct agent outputs, and provide persistence for debugging and monitoring. The system needed to balance simplicity with extensibility for future enhancements.

## Decision

### Event Model Architecture

We implemented a three-tier event model with the following core entities:

#### 1. ToolRun Entity
Represents a complete tool execution session:
```python
@dataclass
class ToolRun:
    tool_name: str           # Name of the tool being executed
    run_id: str             # Unique identifier for this execution
    status: str             # success | failure | running
    started_ms: int         # Start timestamp in milliseconds
    ended_ms: Optional[int] # End timestamp (None when running)
    metadata: Dict[str, Any] # Execution context and parameters
    events: List[ToolEvent] # Timeline of events during execution
```

#### 2. ToolEvent Entity
Represents individual events within a tool execution:
```python
@dataclass
class ToolEvent:
    timestamp_ms: int       # Event timestamp in milliseconds
    event_type: str         # started | partial_result | final_result | error | ended
    payload: Dict[str, Any] # Event-specific data
```

#### 3. Event Types and Payloads
- **started**: Tool execution begins, contains initial metadata
- **partial_result**: Intermediate results during execution (for streaming)
- **final_result**: Complete execution results
- **error**: Execution failures with error details
- **ended**: Execution completion with success/failure status

### Persistence Strategy

#### In-Memory Primary Store
- **TransparencyStore**: Central in-memory repository for all ToolRun instances
- **Global Singleton**: Process-wide store accessed via `get_transparency_store()`
- **Memory Management**: Runs kept in memory for session duration

#### Optional JSON Persistence
- **Feature Flag**: Controlled by `FF_TRANSPARENCY_PERSIST` environment variable
- **Directory Config**: Configurable via `ARM_RUNLOG_DIR` (defaults to `~/.cache/academic-research-mentor/runs/`)
- **Best-Effort**: Persistence failures don't interrupt execution
- **File Format**: Each run saved as `{run_id}.json` with complete serialization

#### Serialization Schema
```json
{
  "tool_name": "o3_search",
  "run_id": "run-o3_search-1234567890",
  "status": "success",
  "started_ms": 1234567890123,
  "ended_ms": 1234567890567,
  "duration_ms": 454,
  "metadata": {
    "score": 0.95,
    "inputs_keys": ["query", "limit"],
    "tool_state": "healthy",
    "backoff_count": 0
  },
  "events": [
    {
      "timestamp_ms": 1234567890123,
      "event_type": "started",
      "payload": {}
    },
    {
      "timestamp_ms": 1234567890345,
      "event_type": "final_result",
      "payload": {
        "summary": ["Found 3 papers"],
        "sources": ["http://arxiv.org/abs/123"]
      }
    }
  ]
}
```

### Streaming Architecture

#### In-Process Pub/Sub System
- **Listener Pattern**: Simple callback-based event broadcasting
- **Fire-and-Forget**: Event emission doesn't block execution
- **Error Isolation**: Listener failures don't affect other listeners or execution

```python
class TransparencyStore:
    def __init__(self):
        self._listeners: List[Any] = []
    
    def add_listener(self, callback: Any) -> None:
        """Add a listener callback for real-time events."""
        if callback not in self._listeners:
            self._listeners.append(callback)
    
    def _emit(self, event: Dict[str, Any]) -> None:
        """Broadcast event to all listeners with error isolation."""
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception:
                continue  # Isolate listener failures
```

#### ReAct Agent Integration
- **Real-time Streaming**: Incremental output for ReAct agent executions
- **Chunk Processing**: Stream individual message deltas as they're generated
- **UI Integration**: Rich console formatting for live updates

```python
# ReAct streaming implementation
for step in self._agent_executor.stream({"messages": messages}, stream_mode="values"):
    try:
        msgs = step.get("messages", [])
        if msgs:
            latest = msgs[-1].get("content", "")
            delta = latest[len(content):]
            if delta:
                print_streaming_chunk(delta)
            content = latest
    except Exception:
        continue
```

### Integration Points

#### Execution Engine Integration
- **Automatic Tracking**: Tool executions automatically create ToolRun instances
- **Metadata Enrichment**: Includes tool health, backoff status, and scoring
- **Event Timeline**: All execution phases logged as events

#### CLI Integration
- **Live Updates**: Real-time display of tool execution status
- **Streaming Responses**: Incremental output for better user experience
- **Error Handling**: Graceful degradation when streaming fails

#### Debugging Support
- **Run Inspection**: `--show-runs` command for reviewing execution history
- **Export Capability**: JSON export functionality for external analysis
- **Environment Debug**: `ARM_DEBUG_ENV=1` for detailed logging

## Consequences

### Benefits
1. **Observability**: Complete audit trail of all tool executions
2. **Real-time Feedback**: Users see live progress during long-running operations
3. **Debugging Support**: Rich metadata for troubleshooting failures
4. **Extensibility**: Pluggable persistence layer for future storage backends
5. **Performance**: In-memory primary store with optional persistence

### Trade-offs
1. **Memory Usage**: In-memory storage grows with session duration
2. **Persistence Overhead**: JSON serialization adds minimal execution overhead
3. **Complexity**: Event system adds some architectural complexity
4. **Error Handling**: Best-effort persistence may lose data on process crashes

### Future Extensibility
1. **Storage Backends**: Pluggable persistence (SQLite, PostgreSQL, etc.)
2. **Event Filtering**: More sophisticated listener filtering and routing
3. **Metrics Integration**: Performance metrics collection from event data
4. **External Streaming**: WebSocket/SSE support for web UI integration

## Migration Path
1. **Current**: In-memory store with optional JSON persistence
2. **Phase 1**: Add SQLite backend option with feature flag
3. **Phase 2**: Implement external streaming for web UI
4. **Phase 3**: Advanced filtering and routing capabilities

## Testing Strategy
- **Unit Tests**: Store operations, event serialization, listener management
- **Integration Tests**: End-to-end event flow from execution to persistence
- **Performance Tests**: Memory usage and streaming throughput under load
- **Error Scenarios**: Persistence failures, listener exceptions, corrupted data