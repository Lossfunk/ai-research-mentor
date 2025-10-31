from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from academic_research_mentor.rich_formatter import print_error, print_info
from academic_research_mentor.cli.session import load_env_file

from .config_loader import load_metrics_config, metrics_config_digest
from .judge_utils import (
    build_context,
    build_judge_clients,
    iso_timestamp,
    load_annotation_rows,
    resolve_metric_column,
    resolve_metric_spec,
    save_judge_payload,
    truncate_text,
    upsert_annotation,
)
from .run_manual_stage import ensure_stage_directories, normalize_stage


STUDENT_PROMPT_PATH = Path("evaluation/judges/student_outcome_judge.md")


def _digest_file(path: Path) -> str:
    data = path.read_bytes() if path.exists() else b""
    return hashlib.sha256(data).hexdigest()


def _load_persona_card(meta: Dict[str, Any]) -> str:
    persona_id = str((meta.get("metadata") or {}).get("persona") or "").strip()
    if not persona_id:
        # Rich generic fallback to stabilize judge context
        return "Student seeking clear, actionable research guidance with limited time and compute resources."
    import glob
    try:
        import yaml  # type: ignore
        for path in glob.glob("evaluation/personas/*.yaml"):
            with open(path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
                if str(payload.get("id", "")).strip() == persona_id:
                    # Compact textual card
                    lines: List[str] = []
                    name = payload.get("name")
                    if name:
                        lines.append(f"id: {persona_id} — {name}")
                    bg = payload.get("background")
                    if bg:
                        lines.append(f"background: {bg}")
                    lvl = payload.get("knowledge_level")
                    if lvl:
                        lines.append(f"knowledge_level: {lvl}")
                    constraints = payload.get("constraints") or {}
                    if isinstance(constraints, dict) and constraints:
                        lines.append("constraints: " + ", ".join(f"{k}={v}" for k, v in constraints.items()))
                    prefs = payload.get("style_prefs") or []
                    if isinstance(prefs, list) and prefs:
                        lines.append("style_prefs: " + ", ".join(str(p) for p in prefs))
                    focus = payload.get("evaluation_focus") or []
                    if isinstance(focus, list) and focus:
                        lines.append("evaluation_focus: " + ", ".join(str(p) for p in focus))
                    return "\n".join(lines)
    except Exception:
        return persona_id
    # Fallback: use a generic student description if we couldn't resolve a YAML card
    return "Student seeking clear, actionable research guidance with limited time and compute resources."


def _render_student_prompt(template: str, persona_card: str, task_card: str, stage: str, agent_response: str) -> Tuple[str, str]:
    # Substitute placeholders; system content is the template header; user content carries the filled blocks
    system = "You are a student persona outcome judge. Output JSON only."
    body = template
    body = body.replace("{persona_card}", persona_card or "<unknown persona>")
    body = body.replace("{task_card}", task_card or "<unknown task>")
    body = body.replace("{stage}", stage)
    body = body.replace("{agent_response}", agent_response)
    return system, body


def _call_student_judge(client: Any, system_prompt: str, user_prompt: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    result = client.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return getattr(result, "content", None) or getattr(result, "text", None) or str(result)


def _parse_student_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].strip("`\n ")
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _aggregate_student(judge_outputs: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    # Average scaled metrics; majority for binaries. Ignore missing.
    scaled_keys = (
        "student_actionability",
        "student_clarity",
        "student_constraint_fit",
        "student_confidence_gain",
    )
    binary_keys = ("student_path_ready", "student_failure_modes")

    acc: Dict[str, List[float]] = {k: [] for k in scaled_keys}
    bins: Dict[str, List[float]] = {k: [] for k in binary_keys}

    for out in judge_outputs:
        sc = (out.get("scores") or {})
        bc = (out.get("binary_checks") or {})
        if isinstance(sc, dict):
            if (v := sc.get("actionability_for_student")) is not None:
                acc["student_actionability"].append(float(v))
            if (v := sc.get("clarity_for_student")) is not None:
                acc["student_clarity"].append(float(v))
            if (v := sc.get("constraint_fit_for_student")) is not None:
                acc["student_constraint_fit"].append(float(v))
            if (v := sc.get("confidence_gain_for_student")) is not None:
                acc["student_confidence_gain"].append(float(v))
        if isinstance(bc, dict):
            if (v := bc.get("path_ready")) is not None:
                bins["student_path_ready"].append(float(v))
            if (v := bc.get("failure_modes_flagged")) is not None:
                bins["student_failure_modes"].append(float(v))

    def _mean(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    out_scores: Dict[str, Optional[float]] = {k: _mean(v) for k, v in acc.items()}
    for k, v in bins.items():
        if v:
            out_scores[k] = 1.0 if (sum(1.0 for x in v if x >= 0.5) / len(v)) >= 0.5 else 0.0
        else:
            out_scores[k] = None

    # Composite: 0.35*A + 0.25*C + 0.25*F + 0.15*G
    a = out_scores.get("student_actionability")
    c = out_scores.get("student_clarity")
    f = out_scores.get("student_constraint_fit")
    g = out_scores.get("student_confidence_gain")
    if all(v is not None for v in (a, c, f, g)):
        out_scores["student_outcome_score"] = (
            0.35 * float(a) + 0.25 * float(c) + 0.25 * float(f) + 0.15 * float(g)
        )
    else:
        out_scores["student_outcome_score"] = None

    return out_scores


def run_student_judges(
    stage: str,
    prompt_ids: Optional[Sequence[str]],
    judge_specs: Sequence[str],
    annotator: str,
    force: bool,
    output_label: Optional[str],
    system_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if not judge_specs:
        raise ValueError("At least one --judge is required")

    stage_letter, stage_folder = normalize_stage(stage)
    if stage_letter not in {"A", "C", "F"}:
        print_error(f"Student judge is scoped to stages A/C/F; requested stage {stage_letter}")

    judge_clients = build_judge_clients(judge_specs)
    judge_models = [spec for spec, _ in judge_clients]

    _, analysis_dir, _ = ensure_stage_directories(stage_folder)
    label = output_label or "student_outcome_judge"
    out_dir = analysis_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholder_csv = out_dir / "annotation_placeholders.csv"

    metrics_cfg = load_metrics_config()
    m_version = metrics_cfg.get("version")
    m_digest = metrics_config_digest()

    student_template = STUDENT_PROMPT_PATH.read_text(encoding="utf-8")
    student_digest = _digest_file(STUDENT_PROMPT_PATH)

    meta_files = sorted(analysis_dir.glob("*_meta.json"))
    wanted = set(prompt_ids) if prompt_ids else None

    summaries: List[Dict[str, Any]] = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt_id = meta.get("prompt_id")
        if not prompt_id:
            continue
        if wanted and prompt_id not in wanted:
            continue
        if system_filter and meta.get("system_id") != system_filter:
            continue

        # Build context
        response_path = Path(meta.get("response_path", ""))
        if not response_path.exists():
            print_error(f"Missing response for {prompt_id}: {response_path}")
            continue
        full_text = response_path.read_text(encoding="utf-8")
        response_text = truncate_text(full_text)
        context = build_context(meta, response_text, "", raw_runs=[], full_response=full_text)

        persona_card = _load_persona_card(meta)
        task_card = str(meta.get("prompt") or context.get("user_prompt") or "")
        system_msg, user_msg = _render_student_prompt(
            student_template,
            persona_card,
            task_card,
            stage_letter,
            context.get("agent_response", ""),
        )

        judge_outputs: List[Dict[str, Any]] = []
        for name, client in judge_clients:
            try:
                raw = _call_student_judge(client, system_msg, user_msg)
                parsed = _parse_student_json(raw) or {}
                judge_outputs.append({
                    "judge": name,
                    "raw": raw,
                    "parsed": parsed,
                })
            except Exception as exc:  # noqa: BLE001
                judge_outputs.append({
                    "judge": name,
                    "error": str(exc),
                })

        # Aggregate across judges
        agg_ready_inputs: List[Dict[str, Any]] = []
        for r in judge_outputs:
            parsed = r.get("parsed") or {}
            if isinstance(parsed, dict):
                agg_ready_inputs.append(parsed)
        aggregated = _aggregate_student(agg_ready_inputs)

        # Prepare scores for CSV upsert
        metric_scores: Dict[str, Optional[float]] = {
            "student_actionability": aggregated.get("student_actionability"),
            "student_clarity": aggregated.get("student_clarity"),
            "student_constraint_fit": aggregated.get("student_constraint_fit"),
            "student_confidence_gain": aggregated.get("student_confidence_gain"),
            "student_path_ready": aggregated.get("student_path_ready"),
            "student_failure_modes": aggregated.get("student_failure_modes"),
            "student_outcome_score": aggregated.get("student_outcome_score"),
        }

        # Upsert annotation CSV under the label directory
        timestamp = iso_timestamp()
        upsert_annotation(
            placeholder_csv,
            prompt_id,
            stage_letter,
            annotator,
            meta.get("system_id"),
            metric_scores,
            timestamp,
            str(response_path),
            meta.get("tool_trace_path") or "",
            force=force,
        )

        # Save detailed judge payload
        payload = {
            "prompt_id": prompt_id,
            "stage": stage_letter,
            "generated_at": timestamp,
            "student_metrics": metric_scores,
            "judges": judge_outputs,
            "judge_models": judge_models,
            "metrics_version": m_version,
            "metrics_config_digest": m_digest,
            "student_prompt_digest": student_digest,
            "model_spec": {
                "system_id": meta.get("system_id"),
                "provider": meta.get("provider"),
                "model": meta.get("model"),
            },
        }
        save_judge_payload(out_dir / f"{prompt_id}_student_judges.json", payload)
        summaries.append(payload)
        sys_id = meta.get("system_id") or meta.get("system") or "unknown"
        print_info(f"Student-judged {prompt_id} ({sys_id}) [{label}]")

    return {
        "stage": stage_letter,
        "processed": len(summaries),
        "judged_prompts": [d["prompt_id"] for d in summaries],
        "output_label": label,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run student-persona LLM judges over evaluation artifacts")
    parser.add_argument("--stage", required=True, help="Stage to score (A/C/F or stage_a/stage_c/stage_f)")
    parser.add_argument("--prompt-id", dest="prompt_ids", action="append", help="Limit to prompt id (repeatable)")
    parser.add_argument("--judge", dest="judges", action="append", required=True, help="Judge spec provider:model (repeat)")
    parser.add_argument("--annotator", required=True, help="Name for CSV rows (e.g., student_judge_oct21)")
    parser.add_argument("--label", help="Optional output label under analysis_reports/<stage>")
    parser.add_argument("--force", action="store_true", help="Overwrite existing rows")
    parser.add_argument("--system", dest="system_filter", help="Filter meta by system_id")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        load_env_file()
        args = parse_args(argv)
        summary = run_student_judges(
            stage=args.stage,
            prompt_ids=args.prompt_ids,
            judge_specs=args.judges,
            annotator=args.annotator,
            force=args.force,
            output_label=args.label,
            system_filter=args.system_filter,
        )
        print_info(
            f"Completed student judge for {summary['stage']} ({summary['output_label']}) — processed {summary['processed']} prompt(s)."
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print_error(f"Student judge run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
