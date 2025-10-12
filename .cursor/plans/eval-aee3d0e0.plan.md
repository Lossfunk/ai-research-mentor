<!-- aee3d0e0-87a7-448a-ba18-1cb44fa20444 986576e8-fc1a-4238-87d4-bbefcf0c10a4 -->
# Evaluation Plan (single-turn, persona, multi-turn)

### Goals

- **Demonstrate mentoring wins** vs baselines on questioning, adaptation, literature guidance, and plans.
- **Quantify research outcomes and process** (coverage, groundedness, adherence, cost/latency).
- **Ablate sources of gain** (guidelines, tools, memory/persona prompting).

### Tracks (how we run evals)

- **Single-turn probes**: targeted checks for guidelines, citations, tool-use sanity, safety/compliance.
- Inputs from `evaluation/tasks/*.yaml` (short prompts) and policy probes.
- **Persona-conditioned runs**: personas applied across both single- and multi-turn to test adaptation.
- Persona cards in `evaluation/personas/*.yaml`.
- **Multi-turn dialogues**: longitudinal sessions (turn_budget=10) to test stage-awareness, pivots, and memory.
- Orchestrator uses `evaluation/matrix.yaml` and logs to `evaluation/results/`.

Note: Persona and multi-turn are linked but orthogonal—personas condition both; multi-turn is its own track.

### Study matrix (scope)

- Defined in `evaluation/matrix.yaml`: topics × difficulty × personas × systems × seeds × tasks, `turn_budget: 10`.
- Pilot slice: 3 topics × 2 personas × 2 systems × 2 seeds → ~24 dialogues (8–12 turns each).

### Systems under test (SUTs)

- `mentor_full`: guidelines + tools + memory (primary).
- `baseline_tutor`: strong LLM, no tools/guidelines.
- `baseline_tools_only`: tools on, guidelines off.

### Dialog protocol

- Initialize with persona + task; enforce `turn_budget`, style normalization, and equalized length.
- Persist transcripts and tool traces under `evaluation/results/raw_logs/<scenario_id>/`.
- Reuse single-turn script and create a minimal multi-turn runner per `evaluation/README.md` guidance.

### Judging & metrics

- **Pairwise LLM-as-a-judge** using `evaluation/judges/pairwise_judge_prompt.md` + `judges/aspect_rubrics.yaml`.
- Aspects: inquiry quality; follow-up depth; persona adaptation; methodology critique; plan completeness; literature guidance; actionability; risk mitigation.
- **Automatic checks**:
- Literature Coverage@K (curated topic lists), Citation Groundedness (DOI/URL resolve + span match), Tool-use efficiency, Guideline adherence, Cost/latency.
- **Aggregation**: win-rate with bootstrap CIs; ELO/TrueSkill; macro (per-topic/persona) and micro (overall).
- **Human calibration**: 10–15% subset for judge–human correlation and IAA.

### Ablations

- Guidelines on/off; tools on/off; memory/history on/off; persona prompting on/off.
- Optional: router/model variants.

### Artifacts & reproducibility

- Config seeds and providers pinned in `.env` and `evaluation/matrix.yaml`.
- Export transcripts, tool traces, judgments, and analysis JSON to `evaluation/results/`.
- Brief runbook in `eval-runs/` with exact commands.

### Deliverables (pilot → scale)

- Pilot report: win-rates + CIs, ELO, key aspect deltas, coverage/groundedness, cost/latency.
- Ablation deltas (guidelines/tools/memory/persona) on the pilot slice.
- Scaled matrix run results (post-pilot), updated figures/tables.
- Appendix: persona/task cards, judge prompts, and metric definitions.

### Acceptance criteria (A*/A paper bar)

- Statistically significant pairwise wins vs baselines on mentoring/outcome aspects.
- Demonstrated contributions from guidelines/tools/memory via ablations.
- Transparent, reproducible artifacts and partial human validation.

### To-dos

- [ ] Finalize pilot slice in `evaluation/matrix.yaml` and select personas/tasks
- [ ] Implement minimal multi-turn orchestrator and logging to results/
- [ ] Run ~24 dialogues (3×2×2×2) with `turn_budget=10`
- [ ] Execute pairwise LLM judging with aspect rubrics and save reports
- [ ] Compute win-rate CIs, ELO, coverage@K, groundedness, cost/latency
- [ ] Run guidelines/tools/memory/persona ablations on pilot slice
- [ ] Label 10–15% subset; compute judge–human correlation and IAA
- [ ] Assemble pilot plots/tables and brief write-up for Slack deck