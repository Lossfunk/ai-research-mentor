# Research Mentor Evaluation Plan (v1)

This plan specifies the end-to-end evaluation for the academic research mentor with a paper-oriented focus. It defines tracks, scenarios, personas, metrics, judges, baselines/ablations, a study matrix, logging, analysis, and reproducibility. Files referenced here live under `evaluation/` and `eval-runs/`.

## 1. Scope & Goals
- Demonstrate that guideline-driven, tool-using mentoring improves research outcomes vs strong generic assistants.
- Measure mentoring quality (questioning, adaptation), research outcomes (plans, critique, coverage), and process (tooling, cost, adherence).
- Support pairwise LLM-judge comparisons with a small human-rated subset for calibration.

## 2. Tracks
- Conversational Mentoring: Socratic questioning, personalization, stage-awareness.
- Research Outcomes: research-question quality, literature coverage, critique depth, experimental design feasibility.
- Process & Compliance: tool-use efficiency, guideline adherence, citation groundedness and validity, latency/cost.

## 3. Scenarios & Tasks
- Scope a research question.
- Design a baseline and evaluation plan.
- Produce a 10-paper mini-review.
- Critique a proposed method.
- Reproducibility plan for a paper.
- Ablation/experiments plan for a proposed idea.

Author task cards in `evaluation/tasks/` with: `id`, `title`, `instructions`, `topic`, `difficulty`, `turn_budget`, `success_criteria`, and `expected_tools`.

## 4. Personas
Represent backgrounds/constraints in YAML under `evaluation/personas/`. Minimum set:
- Novice ML student
- Competent ML student
- Domain switcher (e.g., physics → ML)
- Domain specialist (e.g., NLP)
- Time-constrained student

Persona cards include: `background`, `knowledge_level`, `goals`, `constraints`, `style_prefs`, and `evaluation_focus` (what the judge should prioritize).

## 5. Systems Under Test
- Our mentor (guidelines + tools + memory as configured in `.env`).
- Baseline A: strong LLM “tutor” prompt, no tools, no guidelines.
- Baseline B: tools enabled (RAG/search) but guidelines disabled.
- Optional: open-source agent if comparable.

## 6. Ablations
- Guidelines on/off.
- Tools on/off and provider variants (`src/academic_research_mentor/tools/web_search/providers.py`).
- Memory/history on/off.
- Router/model choices (`src/academic_research_mentor/runtime/models.py`).
- Persona-aware prompting on/off.
- Self-critique/reflection step on/off.

## 7. Metrics
Combine judge rubrics with automatable checks.

7.1 Judge (aspect-based; pairwise preferred)
- Inquiry Quality (clarity, scope, feasibility, novelty)
- Socratic Ratio and Follow-up Depth
- Persona Adaptation (alignment to persona card)
- Methodology Critique (confounds, baselines, metrics, leakage, ablations)
- Plan Completeness & Ordering
- Literature Guidance Quality (relevance, recency)
- Actionability & Risk Mitigation

7.2 Automatic
- Literature Coverage@K: match against curated topic lists (precision/recall, diversity by year/venue).
- Citation Groundedness: URL/DOI resolve, quote-span fuzzy matching, hallucination rate.
- Tool Use Efficiency: success rate, retries, duplicate queries, latency, token/cost.
- Guideline Adherence: presence/usage of required policies.
- Calibration: self-rated confidence vs verifiable correctness (Brier-style surrogate via checks).

References and scale anchors live in `evaluation/judges/aspect_rubrics.yaml`.

## 8. Judges (LLM-as-a-judge)
- Use pairwise comparisons with aspect decisions → final preference; log reasons.
- Multi-judge ensemble (2–3 different models); aggregate via majority or mean score.
- Style bias controls: anonymize system IDs, normalize format/length, randomized order, equal turn budgets.
- Human calibration: 10–15% subset labeled by 2 humans; report judge–human correlation and IAA.

Judge prompts reside in `evaluation/judges/pairwise_judge_prompt.md`.

## 9. Study Matrix
- Topics: {vision, NLP/RAG, alignment, RL, fairness} × Difficulty {easy, med, hard}
- Personas: 4–6 persona cards
- Systems: 3 (ours + 2 baselines)
- Seeds: 2–3 per cell

Example size: 3 topics × 3 difficulty × 4 personas × 3 systems × 2 seeds ≈ 216 dialogues (8–12 turns each).

Matrix and sampling spec in `evaluation/matrix.yaml`.

## 10. Dialog Generation Protocol
- Initialize mentor with persona and task card.
- Enforce `turn_budget`; normalize length/formatting; prevent external context leaks.
- Persist transcripts and tool traces (re-use `evaluation/scripts/run_manual_stage.py` patterns).
- Store under `evaluation/results/raw_logs/<scenario_id>/`.

## 11. Judging Protocol
- For each item, create pairs (system A vs system B) with same persona/task; randomize order.
- Prompt LLM judges with `pairwise_judge_prompt.md` + `aspect_rubrics.yaml` and the persona/task cards.
- Save raw judgments/justifications under `evaluation/results/analysis_reports/<scenario_id>/<judge_model>/`.

## 12. Analysis & Statistics
- Compute win-rate, ELO/TrueSkill; bootstrap 95% CIs; paired tests on prefs.
- Report macro (per-topic/persona) and micro (overall) scores; cost/latency.
- Human calibration subset: judge–human correlation; IAA (Cohen’s κ / Krippendorff’s α).
- Multiple metrics correction via Holm–Bonferroni.

## 13. Reproducibility
- Seed runs; pin models/providers in `.env` and `evaluation/matrix.yaml`.
- Log model names/versions and provider endpoints.
- Export transcripts, tool traces, judgments, and analysis JSON.

## 14. Artifacts & Layout
- `evaluation/personas/` — persona YAMLs
- `evaluation/tasks/` — task YAMLs
- `evaluation/judges/` — judge prompts and rubrics
- `evaluation/matrix.yaml` — run matrix and sampling policy
- `evaluation/results/` — raw logs and analysis reports
- `eval-runs/` — run notes and step-by-step procedures

## 15. Pilot Plan
- Small slice: 3 topics × 2 personas × 2 systems × 2 seeds (24 dialogues).
- Validate rubrics, judge reliability, coverage/citation checkers.
- Iterate on prompt/controls; then scale to full matrix.

## 16. Timeline (suggested)
- Week 1: finalize personas/tasks/rubrics; build judge harness; pilot.
- Week 2: expand to full matrix; run judges; human subset.
- Week 3: analysis, ablations, failure-mode report; write-up figures.

## 17. Acceptance Criteria (for paper submission)
- Statistically significant pairwise wins vs baselines on key mentoring and research-outcome aspects.
- Demonstrated gains from guidelines/tools via ablations.
- Transparent artifacts and reproducibility scripts with partial human validation.

