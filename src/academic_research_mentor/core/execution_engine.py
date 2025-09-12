from __future__ import annotations

"""
Tool execution engine with retry logic (WS3 extension).

Handles the actual tool execution with retries, keeping orchestrator.py under 200 LOC.
"""

from typing import Dict, Any, Optional, List
import time

from ..rich_formatter import print_info, print_error
from .transparency import get_transparency_store


def execute_with_policy(selection_result: Dict[str, Any], strategy: Dict[str, Any],
                       inputs: Dict[str, Any], context: Optional[Dict[str, Any]],
                       policy, list_tools_func) -> Dict[str, Any]:
    """Execute tools using fallback policy."""
    if list_tools_func is None:
        return {
            **selection_result,
            "execution": {"executed": False, "reason": "No tools available"},
            "results": None
        }
    
    tools = list_tools_func()
    primary_name, primary_score = strategy["primary"]
    
    # Try primary tool with retries
    execution_result = try_tool_with_retries(
        tools, primary_name, primary_score, inputs, context, policy
    )
    
    if execution_result["success"]:
        return {
            **selection_result,
            **execution_result,
            "fallback_strategy": strategy
        }
    
    # Primary failed, try fallback if available
    fallback_info = strategy.get("fallback")
    if fallback_info:
        fallback_name, fallback_score = fallback_info
        print_info(f"Trying fallback tool: {fallback_name} (after {primary_name} failed)")
        
        fallback_result = try_tool_with_retries(
            tools, fallback_name, fallback_score, inputs, context, policy
        )
        
        if fallback_result["success"]:
            fallback_result["execution"]["primary_failed"] = primary_name
            fallback_result["execution"]["fallback_used"] = True
            return {
                **selection_result,
                **fallback_result,
                "fallback_strategy": strategy
            }
    
    # All tools failed
    policy.record_failure(primary_name, execution_result["execution"]["reason"])
    return {
        **selection_result,
        "execution": {
            "executed": False,
            "reason": f"All available tools failed. Primary: {execution_result['execution']['reason']}",
            "strategy_used": strategy["strategy"]
        },
        "results": None,
        "fallback_strategy": strategy
    }


def try_tool_with_retries(tools: Dict[str, Any], tool_name: str, score: float,
                         inputs: Dict[str, Any], context: Optional[Dict[str, Any]], policy) -> Dict[str, Any]:
    """Try executing a tool with retry logic."""
    tool = tools.get(tool_name)
    if not tool or not hasattr(tool, "execute"):
        return {
            "success": False,
            "execution": {"executed": False, "reason": f"Tool {tool_name} not executable"}
        }
    
    # Get fallback policy for health status
    from .fallback_policy import get_fallback_policy
    fallback_policy = get_fallback_policy()
    health_summary = fallback_policy.get_tool_health_summary()
    
    # Check if tool is in backoff or degraded mode
    tool_state = health_summary["tool_states"].get(tool_name, "healthy")
    backoff_count = health_summary["backoff_counts"].get(tool_name, 0)
    
    # Transparency: start run
    store = get_transparency_store()
    run_id = f"run-{tool_name}-{int(time.time()*1000)}"
    metadata = {
        "score": score, 
        "inputs_keys": sorted(list(inputs.keys())),
        "tool_state": tool_state,
        "backoff_count": backoff_count
    }
    store.start_run(tool_name, run_id, metadata=metadata)
    
    # Print status information if tool is degraded or in backoff
    status_note = ""
    if tool_state == "degraded":
        status_note = " [DEGRADED]"
        if backoff_count > 0:
            status_note += f" (backoff #{backoff_count})"
    elif tool_state == "circuit_open":
        status_note = " [CIRCUIT OPEN - testing]"
    
    print_info(f"Using tool: {tool_name} (score={score:.2f}){status_note}")

    attempt = 0
    last_error = ""
    
    while attempt < 3:  # Max 3 attempts total
        try:
            if attempt == 0:
                pass
            else:
                print_info(f"Retry {attempt} for {tool_name}")
            
            execution_result = tool.execute(inputs, context)
            
            # Success!
            policy.record_success(tool_name)
            _log_tool_success(store, run_id, execution_result)
            return {
                "success": True,
                "execution": {
                    "executed": True,
                    "tool_used": tool_name,
                    "tool_score": score,
                    "success": True,
                    "attempts": attempt + 1
                },
                "results": execution_result,
                "note": f"Task executed with {tool_name}" + (f" (attempt {attempt + 1})" if attempt > 0 else "")
            }
            
        except Exception as e:
            last_error = str(e)
            attempt += 1
            store.append_event(run_id, "error", {"attempt": attempt, "error": last_error})
            
            # Check if we should retry
            should_retry, delay = policy.should_retry(tool_name, attempt, last_error)
            if should_retry and attempt < 3:
                print_info(f"Retrying {tool_name} in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                break
    
    # All retries failed
    policy.record_failure(tool_name, last_error)
    store.end_run(run_id, success=False, extra_metadata={"error": last_error})
    print_error(f"Tool {tool_name} failed after {attempt} attempts: {last_error}")
    return {
        "success": False,
        "execution": {
            "executed": False,
            "reason": f"{tool_name} failed after {attempt} attempts: {last_error}",
            "attempts": attempt
        }
    }


def _log_tool_success(store, run_id: str, result: Dict[str, Any]) -> None:
    """Append transparency events and print a brief summary and sources."""
    # Summarize findings
    summary_lines: List[str] = []
    sources: List[str] = []

    if isinstance(result, dict):
        # Common schemas we handle
        papers = result.get("papers")
        results = result.get("results")
        threads = result.get("threads")
        retrieved = result.get("retrieved_guidelines")

        if isinstance(papers, list) and papers:
            take = papers[:3]
            for p in take:
                title = p.get("title") or p.get("paper_title") or "paper"
                url = p.get("url") or (p.get("urls", {}) or {}).get("paper")
                if url:
                    sources.append(url)
                summary_lines.append(f"- {title}")
        elif isinstance(results, list) and results:
            take = results[:3]
            for r in take:
                title = r.get("title") or r.get("paper_title") or "result"
                url = r.get("url") or (r.get("urls", {}) or {}).get("paper")
                if url:
                    sources.append(url)
                summary_lines.append(f"- {title}")
        elif isinstance(threads, list) and threads:
            take = threads[:3]
            for t in take:
                title = t.get("paper_title") or "thread"
                url = (t.get("urls", {}) or {}).get("paper")
                if url:
                    sources.append(url)
                summary_lines.append(f"- {title}")
        elif isinstance(retrieved, list) and retrieved:
            take = retrieved[:3]
            for g in take:
                src = g.get("source_domain") or g.get("search_query") or "guideline"
                sources.append(src)
                summary_lines.append(f"- {src}")

    # Log final result
    store.append_event(run_id, "final_result", {"summary": summary_lines[:3], "sources": sources[:5]})
    store.end_run(run_id, success=True, extra_metadata={"num_sources": len(sources)})

    if summary_lines:
        print_info("Found:\n" + "\n".join(summary_lines[:3]))
    if sources:
        print_info("Sources: " + ", ".join(sources[:5]))