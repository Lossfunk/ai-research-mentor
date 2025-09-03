from __future__ import annotations

"""
Core orchestrator skeleton.

Responsibilities (future):
- Coordinate tool selection and execution (via registry)
- Manage timeouts and cancellations
- Emit events to transparency layer

Small, non-invasive scaffolding for WS1: no runtime changes yet.
"""

from typing import Any, Dict, Optional, List, Tuple

try:
    # Optional import; available when WS2 registry is bootstrapped
    from ..tools import list_tools, BaseTool
    from .recommendation import score_tools
except Exception:  # pragma: no cover
    list_tools = None  # type: ignore
    BaseTool = object  # type: ignore
    score_tools = None  # type: ignore


class Orchestrator:
    """Thin orchestrator surface (placeholder for WS3).

    Keep API minimal and stable for now. We will integrate this with the CLI
    and agent later via a feature flag without breaking current behavior.
    """

    def __init__(self) -> None:
        # Placeholder for future dependencies (registry, transparency)
        self._version: str = "0.1"

    @property
    def version(self) -> str:
        return self._version

    def run_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a high-level task (placeholder).

        For WS1, just return a structured no-op result to validate plumbing.
        """
        candidates: List[Tuple[str, float]] = []
        if list_tools is not None:
            try:
                tools = list_tools()
                # Use recommender when flag enabled
                import os

                if score_tools is not None and os.getenv("FF_AGENT_RECOMMENDATION", "false").lower() in ("1", "true", "yes", "on"):
                    scored = score_tools(str((context or {}).get("goal", "")), tools)
                    candidates = [(n, s) for (n, s, _r) in scored]
                else:
                    for name, tool in tools.items():
                        # type: ignore[attr-defined]
                        can = getattr(tool, "can_handle", lambda *_: True)(context or {})
                        if can:
                            # Prefer O3 as primary; legacy as fallback
                            score = 1.0
                            if name == "o3_search":
                                score = 10.0
                                # If O3 client unavailable, reduce score but keep as candidate
                                try:
                                    from ..literature_review.o3_client import get_o3_client  # type: ignore
                                    if not get_o3_client().is_available():
                                        score = 2.0
                                except Exception:
                                    # If import fails, keep default O3 priority
                                    pass
                            elif name.startswith("legacy_"):
                                score = 0.5
                            candidates.append((name, score))
            except Exception:
                pass

        return {
            "ok": True,
            "orchestrator_version": self._version,
            "task": task,
            "context_keys": sorted(list((context or {}).keys())),
            "candidates": sorted(candidates, key=lambda x: x[1], reverse=True),
            "note": "Orchestrator scaffold active. Selection-only; no execution.",
        }
    
    def execute_task(self, task: str, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using the best available tool.
        
        This extends run_task by actually executing the selected tool.
        Returns both selection metadata and execution results.
        """
        # Step 1: Get tool candidates
        selection_result = self.run_task(task, context)
        candidates = selection_result.get("candidates", [])
        
        if not candidates:
            return {
                **selection_result,
                "execution": {"executed": False, "reason": "No suitable tools found"},
                "results": None
            }
        
        # Step 2: Try to execute with the best candidate
        best_tool_name, best_score = candidates[0]
        
        if list_tools is not None:
            try:
                tools = list_tools()
                tool = tools.get(best_tool_name)
                
                if tool and hasattr(tool, "execute"):
                    print(f"🔧 Executing with {best_tool_name} (score: {best_score:.1f})")
                    
                    # Execute the tool
                    execution_result = tool.execute(inputs, context)
                    
                    return {
                        **selection_result,
                        "execution": {
                            "executed": True,
                            "tool_used": best_tool_name,
                            "tool_score": best_score,
                            "success": True
                        },
                        "results": execution_result,
                        "note": f"Task executed successfully with {best_tool_name}"
                    }
                else:
                    return {
                        **selection_result,
                        "execution": {"executed": False, "reason": f"Tool {best_tool_name} not executable"},
                        "results": None
                    }
                    
            except Exception as e:
                # Tool execution failed, try fallback if available
                if len(candidates) > 1:
                    fallback_name, fallback_score = candidates[1]
                    try:
                        tools = list_tools()
                        fallback_tool = tools.get(fallback_name)
                        
                        if fallback_tool and hasattr(fallback_tool, "execute"):
                            print(f"⚠️  {best_tool_name} failed, trying fallback {fallback_name}")
                            execution_result = fallback_tool.execute(inputs, context)
                            
                            return {
                                **selection_result,
                                "execution": {
                                    "executed": True,
                                    "tool_used": fallback_name,
                                    "tool_score": fallback_score,
                                    "success": True,
                                    "primary_failed": best_tool_name,
                                    "failure_reason": str(e)
                                },
                                "results": execution_result,
                                "note": f"Task executed with fallback {fallback_name} after {best_tool_name} failed"
                            }
                    except Exception as fallback_error:
                        return {
                            **selection_result,
                            "execution": {
                                "executed": False,
                                "reason": f"Primary tool {best_tool_name} failed: {e}. Fallback {fallback_name} also failed: {fallback_error}"
                            },
                            "results": None
                        }
                
                return {
                    **selection_result,
                    "execution": {"executed": False, "reason": f"Tool {best_tool_name} failed: {e}"},
                    "results": None
                }
        
        return {
            **selection_result,
            "execution": {"executed": False, "reason": "No tools available"},
            "results": None
        }
