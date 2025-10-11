"""Orchestrator for single-turn prompt evaluation across multiple systems."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

from academic_research_mentor.runtime import _LangChainAgentWrapper
from academic_research_mentor.core.transparency import get_transparency_store


class SingleTurnOrchestrator:
    """Manages single-turn evaluation across multiple LLM systems."""
    
    def __init__(self, systems_to_test: Sequence[str]):
        """
        Initialize with list of provider:model system specs.
        
        Example: ["openai/gpt-4o", "claude/sonnet-4"]
        """
        self.systems_to_test = list(systems_to_test)
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize agent wrappers for all target systems."""
        from academic_research_mentor.core.config import get_providers_config
        
        providers_config = get_providers_config()
        transparency_store = get_transparency_store()
        
        for system_spec in self.systems_to_test:
            try:
                provider, model = system_spec.split("/", 1)
                provider_config = providers_config.get(provider)
                
                if not provider_config:
                    raise ValueError(f"Provider '{provider}' not configured")
                
                # Build model config for LangChain
                model_config = {
                    "provider": provider,
                    "model": model,
                    "temperature": 0.7,  # Consistent temperature for evaluation
                    **provider_config
                }
                
                # Create agent wrapper
                agent = _LangChainAgentWrapper(
                    model_config=model_config,
                    transparency_store=transparency_store,
                )
                
                self.agents[system_spec] = agent
                
            except Exception as exc:
                print(f"Failed to initialize agent for {system_spec}: {exc}")
                # Continue with other systems, will mark as unavailable during processing
    
    def process_single_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        expected_tools: List[str],
        stage: str,
        timeout: int = 120,
    ) -> Dict[str, Dict]:
        """
        Process a single prompt across all configured systems.
        
        Returns mapping of system_id -> response_data dict containing:
        - response: str (system output)
        - tool_trace: List[Dict] (tool execution trace)
        - elapsed: float (time in seconds)
        - success: bool (whether response was generated)
        - timestamp: str (ISO timestamp)
        """
        results: Dict[str, Dict] = {}
        
        for system_spec in self.systems_to_test:
            agent = self.agents.get(system_spec)
            
            if not agent:
                # System not available during initialization
                results[system_spec] = {
                    "response": "",
                    "tool_trace": [],
                    "elapsed": 0,
                    "success": False,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": "Agent not initialized"
                }
                continue
            
            try:
                # Clear previous runs for clean evaluation
                transparency_store = get_transparency_store()
                transparency_store.clear_runs()  # Clear previous state
                
                start_time = time.time()
                
                # Execute single turn
                response = agent.execute_turn(
                    user_message=prompt_text,
                    conversation_context={
                        "prompt_id": prompt_id,
                        "stage": stage,  
                        "expected_tools": expected_tools
                    }
                )
                
                elapsed = time.time() - start_time
                
                # Capture tool execution trace
                runs = transparency_store.list_runs()
                tool_trace = []
                for run in runs:
                    tool_trace.append({
                        "tool_name": run.tool_name,
                        "success": run.success,
                        "duration_seconds": run.duration_seconds,
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "error": run.error,
                        "backoff_count": run.get_backoff_count(),
                    })
                
                results[system_spec] = {
                    "response": response or "",
                    "tool_trace": tool_trace,
                    "elapsed": elapsed,
                    "success": bool(response),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                
                # Check for timeout
                if elapsed > timeout:
                    results[system_spec]["timeout"] = True
                    print(f"⚠️  {system_spec} exceeded {timeout}s timeout (took {elapsed:.1f}s)")
                
            except Exception as exc:
                results[system_spec] = {
                    "response": "",
                    "tool_trace": [],
                    "elapsed": 0,
                    "success": False,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": str(exc)
                }
        
        return results
    
    def batch_process(
        self,
        prompts_data: List[Dict],
        stage: str,
        timeout: int = 120,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Process multiple prompts in batch.
        
        Returns nested structure: prompt_id -> system_id -> response_data
        """
        batch_results: Dict[str, Dict[str, Dict]] = {}
        
        for i, prompt_data in enumerate(prompts_data):
            prompt_id = prompt_data["prompt_id"]
            prompt_text = prompt_data["prompt"]
            expected_tools = prompt_data.get("metadata", {}).get("expected_tools", [])
            
            if progress_callback:
                progress_callback(i + 1, len(prompts_data), prompt_id)
            
            results = self.process_single_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                expected_tools=expected_tools,
                stage=stage,
                timeout=timeout
            )
            
            batch_results[prompt_id] = results
        
        return batch_results
    
    def get_system_status(self) -> Dict[str, Dict]:
        """Get status of all configured systems."""
        status = {}
        for system_spec in self.systems_to_test:
            agent = self.agents.get(system_spec)
            status[system_spec] = {
                "initialized": agent is not None,
                "error": "Not initialized" if not agent else None
            }
        return status
