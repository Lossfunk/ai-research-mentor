from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from academic_research_mentor.cli.session import load_env_file
from academic_research_mentor.core.bootstrap import bootstrap_registry_if_enabled
from academic_research_mentor.guidelines_engine import create_guidelines_injector
from academic_research_mentor.prompts_loader import load_instructions_from_prompt_md
from academic_research_mentor.runtime import build_agent
from academic_research_mentor.core.transparency import get_transparency_store, ToolRun, ToolEvent
from academic_research_mentor.rich_formatter import print_info, print_error

from .stage_directives import apply_stage_directives

# Optional attachments API for grounding runs
try:  # pragma: no cover - optional
    from academic_research_mentor.attachments import attach_pdfs as _attach_pdfs, get_summary as _att_summary
except Exception:  # pragma: no cover
    _attach_pdfs = None  # type: ignore
    _att_summary = None  # type: ignore


RUNTIME_PRELUDE = (
    "Use the selected core prompt variant only; never combine prompts. "
    "Default to conversational answers; call tools only when they would materially change advice. "
    "When user-attached PDFs are present, FIRST use attachments_search to ground your answer with [file:page] citations. "
    "For research queries about papers, literature, or getting started in a field: PREFER unified_research tool which combines papers and guidelines with [P#] and [G#] citations. "
    "For mentorship, hypothesis-generation, getting-started, novelty, experiments, methodology: AFTER grounding, call mentorship_guidelines (research_guidelines) BEFORE any literature_search; "
    "then, if helpful, run literature_search. In your final answer include (1) at least three concrete, falsifiable experiments and (2) one to two literature anchors (titles with links). "
    "Always keep claims grounded in attached snippets with [file:page] citations. "
    "IMPORTANT: Your advice must avoid hyperbole, and claims must be substantiated by evidence presented. "
    "Science is evidence-based; never present unsubstantiated claims. If a claim is speculative, pose it as conjecture, not a conclusion."
)


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


@contextlib.contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
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

RUNTIME_CITATION_DIRECTIVE = (
    " Always include citations to sources when giving research advice. "
    "When using unified_research tool: embed inline bracketed citations [P#] for papers and [G#] for guidelines right after the specific sentences they support. "
    "When using other tools: embed inline bracketed citations [n] right after the specific sentences they support, where [n] refers to the numbered source from the tool output. "
    "Soft guidance: Prefer citing relevant papers [P#] when available for research recommendations. If no relevant papers exist, use guidelines [G#] for methodology advice. "
    "Also include a final 'Citations' section listing [ID] Title — URL."
)

ANNOTATION_COLUMNS: Sequence[str] = (
    "prompt_id",
    "stage",
    "annotator",
    "system_id",
    "run_timestamp",
    "response_path",
    "tool_trace_path",
    "tool_routing",
    # Meta capability metrics (binary)
    "citation_validity",
    "fallback_robustness",
    "evidence_integrity",
    # Meta capability metrics (scaled)
    "rag_fidelity_score",
    "citation_relevance_score",
    "source_fit_score",
    "actionability_score",
    "clarification_quality_score",
    "citation_quality_score",
    # Mentorship capability metrics
    "persona_compliance_score",
    "stage_awareness_score",
    "tone_constructive_score",
    "timeline_guidance",
    "expectation_management",
    "novelty_assessment",
    "evidence_gap_detection",
    "hypothesis_generation",
    "distractor_rejection",
    "experiment_design",
    "scope_feasibility",
    "feasibility_analysis",
    "skills_gap_guidance",
    "domain_mapping",
    "risk_analysis",
    "plan_completeness",
    "resource_estimation",
    "timeline_quality",
    "risk_mitigation",
    "additional_notes",
    # Student-outcome judge metrics (style-agnostic)
    "student_actionability_score",
    "student_clarity_score",
    "student_constraint_fit_score",
    "student_confidence_gain_score",
    "student_path_ready",
    "student_failure_modes",
    "student_outcome_score",
)


class StageRunError(RuntimeError):
    pass


def normalize_stage(stage: str) -> Tuple[str, str]:
    value = stage.strip().lower()
    if value in {"a", "stage_a"}:
        return "A", "stage_a"
    if value in {"b", "stage_b"}:
        return "B", "stage_b"
    if value in {"c", "stage_c"}:
        return "C", "stage_c"
    if value in {"d", "stage_d"}:
        return "D", "stage_d"
    if value in {"e", "stage_e"}:
        return "E", "stage_e"
    if value in {"f", "stage_f"}:
        return "F", "stage_f"
    raise StageRunError(f"Unknown stage: {stage}")


def load_prompt_records(stage_letter: str, prompt_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    data_path = Path("evaluation/data/evals_single_turn.jsonl")
    if not data_path.exists():
        raise StageRunError(f"Missing prompt dataset: {data_path}")
    records: List[Dict[str, Any]] = []
    wanted = set(prompt_ids) if prompt_ids else None
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageRunError(f"Invalid JSONL entry: {exc}") from exc
            metadata = payload.get("metadata") or {}
            if str(metadata.get("stage", "")).upper() != stage_letter:
                continue
            pid = str(payload.get("prompt_id"))
            if wanted and pid not in wanted:
                continue
            records.append(payload)
    if not records:
        raise StageRunError(f"No prompts found for stage {stage_letter}")
    return records


def prepare_agent() -> Tuple[Any, str]:
    load_env_file()
    bootstrap_registry_if_enabled()

    prompt_variant = (
        os.environ.get("ARM_PROMPT")
        or os.environ.get("LC_PROMPT")
        or os.environ.get("AGNO_PROMPT")
        or "mentor"
    ).strip().lower()
    ascii_normalize = bool(
        os.environ.get("ARM_PROMPT_ASCII")
        or os.environ.get("LC_PROMPT_ASCII")
        or os.environ.get("AGNO_PROMPT_ASCII")
    )

    instructions, loaded_variant = load_instructions_from_prompt_md(prompt_variant, ascii_normalize)
    if not instructions:
        instructions = (
            "You are an expert research mentor. Ask high-impact questions first, then provide concise, actionable guidance."
        )
        loaded_variant = "fallback"

    effective_instructions = f"{RUNTIME_PRELUDE}{RUNTIME_CITATION_DIRECTIVE}\n\n{instructions}"

    try:
        injector = create_guidelines_injector()
        stats = injector.get_stats()
        cfg = stats.get("config", {}) if isinstance(stats, dict) else {}
        enabled = bool(cfg.get("is_enabled"))
        if enabled:
            effective_instructions = injector.inject_guidelines(effective_instructions)  # type: ignore[attr-defined]
            total = (stats.get("guidelines_stats", {}) or {}).get("total_guidelines")
            print_info(f"Guidelines injector enabled (total={total})")
    except Exception:
        pass

    agent, reason = build_agent(effective_instructions)
    if agent is None:
        raise StageRunError(reason or "Agent initialization failed; ensure API keys are configured")
    return agent, str(loaded_variant)


def serialize_tool_event(evt: ToolEvent) -> Dict[str, Any]:
    return {
        "timestamp_ms": evt.timestamp_ms,
        "event_type": evt.event_type,
        "payload": _json_safe(evt.payload),
    }


def serialize_tool_run(run: ToolRun) -> Dict[str, Any]:
    events = [serialize_tool_event(evt) for evt in (run.events or [])]
    duration = None
    if run.started_ms and run.ended_ms:
        duration = run.ended_ms - run.started_ms
    return {
        "tool_name": run.tool_name,
        "run_id": run.run_id,
        "status": run.status,
        "started_ms": run.started_ms,
        "ended_ms": run.ended_ms,
        "duration_ms": duration,
        "metadata": _json_safe(run.metadata or {}) or {},
        "events": events,
    }


def ensure_stage_directories(stage_folder: str) -> Tuple[Path, Path, Path]:
    base = Path("evals-for-papers/results")
    raw_dir = base / "raw_logs" / stage_folder
    analysis_dir = base / "analysis_reports" / stage_folder
    iaa_dir = base / "inter_annotator_agreement" / stage_folder
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    iaa_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, analysis_dir, iaa_dir


def upsert_annotation_row(csv_path: Path, row: Dict[str, str], force: bool) -> None:
    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for existing in reader:
                if not existing:
                    continue
                rows.append(existing)

    replaced = False
    for existing in rows:
        if existing.get("prompt_id") == row.get("prompt_id"):
            if not force:
                return
            existing.update(row)
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        for record in rows:
            writer.writerow({col: record.get(col, "") for col in ANNOTATION_COLUMNS})


def execute_prompt(
    agent: Any,
    prompt_record: Dict[str, Any],
    stage_letter: str,
    stage_folder: str,
    raw_dir: Path,
    analysis_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    prompt_id = str(prompt_record.get("prompt_id"))
    prompt_text = str(prompt_record.get("prompt"))
    expected_checks = list(prompt_record.get("expected_checks") or [])
    metadata = dict(prompt_record.get("metadata") or {})
    system_id = metadata.get("system_id") or metadata.get("system") or "mentor_manual"

    response_path = raw_dir / f"{prompt_id}.txt"
    tool_path = raw_dir / f"{prompt_id}_tools.json"
    meta_path = analysis_dir / f"{prompt_id}_meta.json"

    if not force and response_path.exists():
        print_info(f"Skipping {prompt_id}; outputs already exist (use --force to overwrite)")
        return {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "skipped": True,
            "response_path": str(response_path),
            "tool_trace_path": str(tool_path),
            "meta_path": str(meta_path),
            "expected_checks": expected_checks,
            "metadata": metadata,
        }

    if hasattr(agent, "reset_history"):
        try:
            agent.reset_history()
        except Exception:
            pass

    store = get_transparency_store()
    before_ids = {run.run_id for run in store.list_runs()}
    augmented_prompt, directive_id = apply_stage_directives(stage_letter, prompt_text)
    started = time.time()
    reply = agent.run(augmented_prompt)
    elapsed = time.time() - started
    content = getattr(reply, "content", "") or getattr(reply, "text", "") or ""

    after_runs = store.list_runs()
    new_runs = [serialize_tool_run(run) for run in after_runs if run.run_id not in before_ids]

    response_path.write_text(content, encoding="utf-8")
    tool_path.write_text(json.dumps(new_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_payload = {
        "prompt_id": prompt_id,
        "stage": stage_letter,
        "prompt": prompt_text,
        "expected_checks": expected_checks,
        "metadata": metadata,
        "system_id": system_id,
        "system_alias": metadata.get("system_alias"),
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": elapsed,
        "response_path": str(response_path),
        "tool_trace_path": str(tool_path),
        "tool_runs_count": len(new_runs),
    }
    if directive_id:
        meta_payload["stage_directive"] = directive_id
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    annotation_row = {col: "" for col in ANNOTATION_COLUMNS}
    annotation_row.update(
        {
            "prompt_id": prompt_id,
            "stage": stage_letter,
            "system_id": system_id,
            "run_timestamp": meta_payload["run_timestamp"],
            "response_path": str(response_path),
            "tool_trace_path": str(tool_path),
        }
    )
    upsert_annotation_row(analysis_dir / "annotation_placeholders.csv", annotation_row, force=force)

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "response_path": str(response_path),
        "tool_trace_path": str(tool_path),
        "meta_path": str(meta_path),
        "expected_checks": expected_checks,
        "metadata": metadata,
        "tool_runs": new_runs,
    }


def run_stage(
    stage: str,
    prompt_ids: Optional[Sequence[str]] = None,
    force: bool = False,
    *,
    attach_pdfs: Optional[Sequence[str]] = None,
    attachments_dir: Optional[str] = None,
) -> Dict[str, Any]:
    stage_letter, stage_folder = normalize_stage(stage)
    records = load_prompt_records(stage_letter, prompt_ids)
    # Load attachments before building the agent so tools can reflect presence
    try:
        pdfs: List[str] = []
        if attachments_dir:
            base = Path(attachments_dir)
            if base.exists() and base.is_dir():
                pdfs.extend([str(p) for p in base.glob("*.pdf")])
        if attach_pdfs:
            pdfs.extend([str(Path(p)) for p in attach_pdfs if p])
        if pdfs and _attach_pdfs is not None:
            _attach_pdfs(pdfs)
            if _att_summary is not None:
                summ = _att_summary()
                print_info(
                    f"Attachments: files={summ.get('files')}, pages={summ.get('pages')}, chunks={summ.get('chunks')}"
                )
        elif pdfs and _attach_pdfs is None:
            print_error("Attachments support not available; skipping PDF attach")
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to attach PDFs: {exc}")
    agent, loaded_variant = prepare_agent()
    raw_dir, analysis_dir, _ = ensure_stage_directories(stage_folder)

    run_started = datetime.utcnow().isoformat() + "Z"
    results: List[Dict[str, Any]] = []
    for record in records:
        outcome = execute_prompt(agent, record, stage_letter, stage_folder, raw_dir, analysis_dir, force)
        results.append(outcome)

    summary = {
        "stage": stage_letter,
        "stage_folder": stage_folder,
        "run_started": run_started,
        "prompt_variant": loaded_variant,
        "total_prompts": len(records),
        "results": results,
    }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = analysis_dir / f"{stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_info(f"Stage {stage_letter} processed {len(records)} prompt(s); summary -> {summary_path}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated single-turn eval prompts for a stage")
    parser.add_argument("--stage", required=True, help="Stage identifier (A, B, C or stage_a, stage_b, stage_c)")
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids", help="Run only specified prompt_id (can repeat)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts for prompts")
    parser.add_argument("--attach-pdf", dest="attach_pdfs", action="append", help="Attach one or more PDFs (repeatable)")
    parser.add_argument("--attachments-dir", help="Directory containing PDFs to attach (all *.pdf loaded)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        summary = run_stage(
            args.stage,
            prompt_ids=args.prompt_ids,
            force=args.force,
            attach_pdfs=getattr(args, "attach_pdfs", None),
            attachments_dir=getattr(args, "attachments_dir", None),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except StageRunError as exc:
        print_error(str(exc))
        return 1
    except KeyboardInterrupt:
        print_error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
