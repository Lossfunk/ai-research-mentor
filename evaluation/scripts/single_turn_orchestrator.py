"""Orchestrator for single-turn prompt evaluation across multiple systems."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from academic_research_mentor.core.transparency import get_transparency_store
from academic_research_mentor.runtime.context import prepare_agent

from .stage_directives import apply_stage_directives


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return str(value)


@dataclass
class _AgentEntry:
    system_spec: str
    provider: str
    model: str
    agent: Any
    prompt_variant: str
    model_params: Dict[str, Any]


def _safe_system_alias(system_spec: str) -> str:
    return system_spec.replace("/", "_").replace(":", "-")


@contextlib.contextmanager
def _override_env(env_overrides: Dict[str, Optional[str]]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in env_overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SingleTurnOrchestrator:
    """Manages single-turn evaluation across multiple LLM systems."""

    def __init__(
        self,
        systems_to_test: Sequence[str],
        *,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        baseline_mode: bool = False,
        tool_whitelist: Optional[Sequence[str]] = None,
    ) -> None:
        self.systems_to_test = list(systems_to_test)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        env_baseline = _is_truthy(os.environ.get("ARM_BASELINE_MODE"))
        self.baseline_mode = baseline_mode or env_baseline
        self.tool_whitelist = list(tool_whitelist) if tool_whitelist else None
        self.agents: Dict[str, _AgentEntry] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        for system_spec in self.systems_to_test:
            base_provider, sep, provider_model = system_spec.partition(":")
            if not base_provider:
                print(f"Skipping invalid system spec '{system_spec}'")
                continue

            provider_key = base_provider.strip().lower()
            model_spec = provider_model.strip() if sep else ""
            if not model_spec:
                # Fallback: try splitting on first whitespace
                _, _, alt = system_spec.partition(" ")
                model_spec = alt.strip()
            if not model_spec:
                print(f"Missing model identifier in system spec '{system_spec}'")
                continue

            env_overrides: Dict[str, Optional[str]] = {}
            if provider_key == "openrouter":
                env_overrides["OPENROUTER_MODEL"] = model_spec
            else:
                print(f"Provider '{system_spec}' unsupported for automatic orchestrator runs")
                continue

            if self.baseline_mode:
                env_overrides["ARM_BASELINE_MODE"] = "1"
                env_overrides["ARM_GUIDELINES_MODE"] = "off"
                if self.tool_whitelist:
                    env_overrides["ARM_TOOL_WHITELIST"] = ",".join(self.tool_whitelist)
                else:
                    env_overrides["ARM_TOOL_WHITELIST"] = "attachments_search,web_search"
            elif self.tool_whitelist:
                env_overrides["ARM_TOOL_WHITELIST"] = ",".join(self.tool_whitelist)

            with _override_env(env_overrides):
                prep = prepare_agent()

            agent = prep.agent
            if agent is None:
                print(f"Failed to initialize agent for {system_spec}: {prep.offline_reason}")
                continue

            self._configure_agent(agent)

            params = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "seed": self.seed,
            }

            self.agents[system_spec] = _AgentEntry(
                system_spec=system_spec,
                provider=provider_key,
                model=model_spec,
                agent=agent,
                prompt_variant=prep.loaded_variant,
                model_params=params,
            )

    def _configure_agent(self, agent: Any) -> None:
        llm = getattr(agent, "_llm", None)
        if llm is None:
            return
        try:
            if hasattr(llm, "temperature"):
                llm.temperature = self.temperature
        except Exception:
            pass
        if self.max_output_tokens is not None:
            applied = False
            for attr in ("max_tokens", "max_output_tokens"):
                try:
                    if hasattr(llm, attr):
                        setattr(llm, attr, self.max_output_tokens)
                        applied = True
                        break
                except Exception:
                    continue
        if self.seed is not None and hasattr(llm, "seed"):
            try:
                setattr(llm, "seed", self.seed)
            except Exception:
                pass

        # Update model kwargs when present to keep metadata in sync
        kwargs = getattr(llm, "model_kwargs", None)
        if isinstance(kwargs, dict):
            kwargs["temperature"] = self.temperature
            if self.max_output_tokens is not None:
                kwargs.pop("max_output_tokens", None)
                kwargs["max_tokens"] = self.max_output_tokens
            if self.seed is not None:
                kwargs["seed"] = self.seed

    def process_single_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        expected_tools: List[str],
        stage: str,
        *,
        expected_checks: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ) -> Dict[str, Dict[str, Any]]:
        """Process a single prompt across all configured systems."""

        results: Dict[str, Dict[str, Any]] = {}
        transparency_store = get_transparency_store()
        metadata_payload = dict(metadata or {})
        expected_checks_list = list(expected_checks or [])

        for system_spec in self.systems_to_test:
            entry = self.agents.get(system_spec)
            alias = _safe_system_alias(system_spec)
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            if entry is None:
                results[system_spec] = {
                    "response": "",
                    "tool_trace": [],
                    "elapsed": 0.0,
                    "success": False,
                    "timestamp": timestamp,
                    "error": "agent_not_initialized",
                    "meta": {
                        "prompt_id": prompt_id,
                        "stage": stage,
                        "expected_checks": expected_checks_list,
                        "metadata": metadata_payload,
                        "system_id": system_spec,
                        "system_alias": alias,
                        "expected_tools": list(expected_tools),
                        "baseline_mode": self.baseline_mode,
                    },
                }
                continue

            try:
                transparency_store.clear_runs()
            except Exception:
                pass

            start = time.time()
            response_text = ""
            error_payload: Optional[str] = None
            try:
                augmented_prompt, directive_id = apply_stage_directives(stage, prompt_text)
                reply = entry.agent.run(augmented_prompt)
                response_text = getattr(reply, "content", "") or getattr(reply, "text", "") or ""
            except Exception as exc:  # noqa: BLE001
                error_payload = str(exc)
                directive_id = None

            elapsed = time.time() - start

            runs = transparency_store.list_runs()
            tool_trace: List[Dict[str, Any]] = []
            for run in runs:
                run_status = getattr(run, "status", None)
                success_flag: Optional[bool]
                if isinstance(run_status, str):
                    success_flag = run_status.lower() == "success"
                else:
                    success_flag = getattr(run, "success", None)

                tool_trace.append(
                    {
                        "tool_name": getattr(run, "tool_name", None),
                        "status": run_status,
                        "success": success_flag,
                        "duration_seconds": getattr(run, "duration_seconds", None),
                        "input_tokens": getattr(run, "input_tokens", None),
                        "output_tokens": getattr(run, "output_tokens", None),
                        "error": _json_safe(getattr(run, "error", None)),
                        "backoff_count": getattr(run, "get_backoff_count", lambda: None)(),
                        "metadata": _json_safe(getattr(run, "metadata", {}) or {}),
                    }
                )

            results[system_spec] = {
                "response": response_text,
                "tool_trace": tool_trace,
                "elapsed": elapsed,
                "success": bool(response_text) and error_payload is None,
                "timestamp": timestamp,
                "error": error_payload,
                "meta": {
                    "prompt_id": prompt_id,
                    "stage": stage,
                    "prompt": prompt_text,
                    "expected_checks": expected_checks_list,
                    "metadata": metadata_payload,
                    "system_id": system_spec,
                    "system_alias": alias,
                    "provider": entry.provider,
                    "model": entry.model,
                    "model_params": entry.model_params,
                    "prompt_variant": entry.prompt_variant,
                    "expected_tools": list(expected_tools),
                    "baseline_mode": self.baseline_mode,
                    "generated_at": timestamp,
                    "timeout_seconds": timeout,
                    "elapsed_seconds": elapsed,
                    "tool_runs_count": len(tool_trace),
                },
            }
            if directive_id:
                results[system_spec]["meta"]["stage_directive"] = directive_id

            if elapsed > timeout:
                results[system_spec]["meta"]["timeout_exceeded"] = True

        return results

    def batch_process(
        self,
        prompts_data: List[Dict[str, Any]],
        stage: str,
        timeout: int = 120,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        batch_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for index, prompt_data in enumerate(prompts_data, start=1):
            prompt_id = prompt_data["prompt_id"]
            prompt_text = prompt_data["prompt"]
            expected_tools = prompt_data.get("metadata", {}).get("expected_tools", [])

            if progress_callback:
                progress_callback(index, len(prompts_data), prompt_id)

            batch_results[prompt_id] = self.process_single_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                expected_tools=expected_tools,
                stage=stage,
                expected_checks=prompt_data.get("expected_checks"),
                metadata=prompt_data.get("metadata"),
                timeout=timeout,
            )

        return batch_results

    def get_system_status(self) -> Dict[str, Dict[str, Any]]:
        status: Dict[str, Dict[str, Any]] = {}
        for system_spec in self.systems_to_test:
            status[system_spec] = {
                "initialized": system_spec in self.agents,
                "provider": self.agents.get(system_spec).provider if system_spec in self.agents else None,
                "model": self.agents.get(system_spec).model if system_spec in self.agents else None,
            }
        return status
