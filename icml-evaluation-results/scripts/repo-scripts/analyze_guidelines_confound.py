#!/usr/bin/env python3
"""
Observational sensitivity check for guidelines tool usage.

Computes Mentor win-rates in pairwise evaluations conditional on whether
the `research_guidelines` tool was actually invoked (based on tool traces
already saved in analysis reports), without running any new model calls.

Usage:
  uv run python scripts/analyze_guidelines_confound.py

Outputs:
  - Conditional win counts and rates vs GPT-5 and vs Claude Sonnet 4.5.
  - Totals exclude ties.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("evals-for-papers/results/analysis_reports")


def load_guidelines_usage() -> dict[str, set[str]]:
    stages: dict[str, set[str]] = {}
    for stage_dir in ROOT.iterdir():
        if not stage_dir.is_dir() or not re.match(r"stage_[a-f]$", stage_dir.name):
            continue
        used: set[str] = set()
        for summ in stage_dir.glob("*_summary.json"):
            try:
                data = json.loads(summ.read_text())
            except Exception:
                continue
            for r in data.get("results", []) or []:
                pid = r.get("prompt_id")
                if not pid:
                    continue
                for tr in r.get("tool_runs", []) or []:
                    if str(tr.get("tool_name")).lower() == "research_guidelines":
                        used.add(pid)
                        break
        stages[stage_dir.name] = used
    return stages


def load_pairwise() -> dict[str, dict[str, dict[str, str | None]]]:
    # pairwise[key][stage][prompt_id] -> winner_system_id (or None for tie)
    pairwise: dict[str, dict[str, dict[str, str | None]]] = {"gpt5": {}, "claude": {}}
    for stage_dir in ROOT.iterdir():
        if not stage_dir.is_dir() or not re.match(r"stage_[a-f]$", stage_dir.name):
            continue
        for pair_dir in stage_dir.rglob("pairwise/*__vs__*"):
            p = str(pair_dir)
            key: str | None = None
            if "mentor_manual__vs__openrouter:openai_gpt-5" in p or "openrouter:openai_gpt-5__vs__mentor_manual" in p:
                key = "gpt5"
            elif "mentor_manual__vs__openrouter:anthropic_claude-sonnet-4.5" in p or "openrouter:anthropic_claude-sonnet-4.5__vs__mentor_manual" in p:
                key = "claude"
            if key is None:
                continue
            for jf in pair_dir.glob("stage_*_*.json"):
                try:
                    jd = json.loads(jf.read_text())
                except Exception:
                    continue
                pid = jd.get("prompt_id")
                if not pid:
                    continue
                pairwise[key].setdefault(stage_dir.name, {})[pid] = jd.get("winner_system_id")
    return pairwise


def summarize_conditional(pairwise: dict, used: dict) -> None:
    def compute(key: str):
        wins_with = Counter()
        wins_without = Counter()
        tot_with = 0
        tot_without = 0
        for stage, results in pairwise.get(key, {}).items():
            used_set = used.get(stage, set())
            for pid, winner in results.items():
                if winner is None:
                    continue  # tie
                if pid in used_set:
                    tot_with += 1
                    if winner == "mentor_manual":
                        wins_with["w"] += 1
                else:
                    tot_without += 1
                    if winner == "mentor_manual":
                        wins_without["w"] += 1
        return wins_with["w"], tot_with, wins_without["w"], tot_without

    for key, label in (("gpt5", "GPT-5"), ("claude", "Claude Sonnet 4.5")):
        w_with, t_with, w_wo, t_wo = compute(key)
        rate_with = f"{(w_with/t_with*100):.1f}%" if t_with else "NA"
        rate_wo = f"{(w_wo/t_wo*100):.1f}%" if t_wo else "NA"
        print(f"vs {label}: Mentor wins with guidelines {w_with}/{t_with} ({rate_with}), without {w_wo}/{t_wo} ({rate_wo})")

    # Optional: per-stage breakdown (diagnostic)
    print("\nPer-stage diagnostic (ties excluded):")
    for key, label in (("gpt5", "GPT-5"), ("claude", "Claude Sonnet 4.5")):
        print(f"\nvs {label} by stage:")
        for stage in sorted(pairwise.get(key, {}).keys()):
            used_set = used.get(stage, set())
            w_with = w_wo = 0
            t_with = t_wo = 0
            for pid, winner in pairwise[key][stage].items():
                if winner is None:
                    continue
                bucket = (pid in used_set)
                if bucket:
                    t_with += 1
                    if winner == "mentor_manual":
                        w_with += 1
                else:
                    t_wo += 1
                    if winner == "mentor_manual":
                        w_wo += 1
            rate_with = f"{(w_with/t_with*100):.0f}%" if t_with else "NA"
            rate_wo = f"{(w_wo/t_wo*100):.0f}%" if t_wo else "NA"
            print(f"  {stage.upper()}: with {w_with}/{t_with} ({rate_with}), without {w_wo}/{t_wo} ({rate_wo})")


def main() -> None:
    used = load_guidelines_usage()
    pairwise = load_pairwise()
    summarize_conditional(pairwise, used)


if __name__ == "__main__":
    main()
