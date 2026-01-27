# Reproducibility Information (ICML 2026)

Last updated: 2026-01-27

## 1. Model IDs + Provider + Snapshot Date

Systems under evaluation:

| System | Provider | Model ID |
| --- | --- | --- |
| MENTOR | OpenRouter | moonshotai/kimi-k2-thinking |
| GPT-5 | OpenRouter | openai/gpt-5 |
| Claude | OpenRouter | anthropic/claude-sonnet-4.5 |
| Gemini | OpenRouter | google/gemini-3-pro-preview |

LLM judges (3-judge ensemble):

| Judge | Provider | Model ID |
| --- | --- | --- |
| Qwen3-Max | OpenRouter | qwen/qwen3-max |
| DeepSeek | OpenRouter | deepseek/deepseek-v3.2-exp |
| Grok-4 | OpenRouter | x-ai/grok-4-fast |

Snapshot date: January 2026 (evaluations run Jan 20-27, 2026)

---

## 2. Temperature, Seeds, Sampling Params

| Component | Temperature | Max Tokens | Notes |
| --- | --- | --- | --- |
| MENTOR responses | 0.7 (default) | 4096 | From src/academic_research_mentor/llm/client.py |
| Baseline responses | Provider default | - | Via OpenRouter API |
| LLM judges | 0.0 | 1536 | From scripts/judge_utils.py:505 |
| Multi-turn student | Provider default | - | Uses OpenRouter |

Seeds: Not explicitly set for generation; evaluations use deterministic judge ordering. Pairwise judging script supports --seed but defaults to no seed.

---

## 3. Prompt Files + Tool Configs

System prompts:
- prompt.md (main MENTOR system prompt with stage-aware mentoring)

Judge prompts (icml-evaluation-results/judges/):
- single_turn_holistic_prompt.md
- holistic_conversation_judge.md
- pairwise_judge_prompt.md
- student_outcome_judge.md

Config files (icml-evaluation-results/config/):
- metrics.yaml (metric definitions and weights)
- citation_domains.yaml (valid citation domain patterns)

Tool configs:
- Guidelines engine: src/academic_research_mentor/guidelines_engine/
- Literature search: src/academic_research_mentor/tools/web_search/
- Attachments: src/academic_research_mentor/attachments/

---

## 4. Exact Dataset/Prompt List

Single-turn dataset:
- File: icml-evaluation-results/prompts/evals_single_turn.jsonl
- Size: 90 prompts (15 per stage x 6 stages A-F)
- Format: JSONL with prompt_id, prompt, expected_checks, metadata (stage, persona, domain, constraints)

Multi-turn scenarios:
- File: icml-evaluation-results/scripts/multi_turn/scenarios.jsonl
- Size: 20 scenarios
- Format: JSONL with scenario_id, topic, persona, constraints, stage

Human evaluation:
- Files: icml-evaluation-results/human-baseline-votes/*.csv (15 rater files)
- Size: 218 pairwise comparisons

---

## 5. Commit Hash + Scripts to Rerun

Commit hash: 2dce26b6bd9bfcb912dc0d06c607de24ef1e3b39

Scripts to reproduce:

```bash
# Single-turn generation (per stage)
uv run python -m evaluation.scripts.run_manual_stage --stage a

# Single-turn judges (all stages, all systems)
bash icml-evaluation-results/scripts/evals-for-papers-scripts/run_stage_a_judges.sh

# Multi-turn evaluation
uv run python icml-evaluation-results/scripts/run_multi_turn_evals.py

# Ablations (90 prompts)
uv run python icml-evaluation-results/ablations/run_ablations_90.py
uv run python icml-evaluation-results/ablations/run_ablation_judges.py

# Compute IAA
uv run python icml-evaluation-results/inter_annotator_agreement/compute_iaa.py

# Generate figures
uv run python icml-evaluation-results/scripts/generate_icml_figures.py
```

---

## 6. Judge Setup (Models, Prompts, Aggregation)

Judge models:
- 3-judge ensemble: qwen/qwen3-max, deepseek/deepseek-v3.2-exp, x-ai/grok-4-fast
- All via OpenRouter at temperature=0.0

Judge prompts:
- Single-turn: single_turn_holistic_prompt.md (holistic 0-2 scale with calibration anchors)
- Multi-turn: holistic_conversation_judge.md (4 dimensions: helpfulness, progress, effectiveness, efficiency)
- Requires explicit weakness identification before scoring

Aggregation:
- Score aggregation: mean across 3 judges (judge_utils.py:aggregate_scores)
- Holistic score: weighted mean of 4 dimensions (equal weights 0.25 each)
- Success threshold: 1.5/2.0 for multi-turn
- Pairwise: majority vote with tie-breaking

IAA results:
- Mean ICC: 0.46
- Mean pairwise correlation: 0.49
- Best agreement: conversation efficiency (ICC=0.53)
- Worst agreement: student progress (ICC=0.36)

---

## Key points for the paper

1. All models accessed via OpenRouter for consistent API interface.
2. Judges run at temperature=0 for deterministic evaluation.
3. 3-judge ensemble reduces single-judge bias by averaging.
4. 90 prompts x 4 systems = 360 single-turn responses.
5. 20 scenarios x 4 systems = 80 multi-turn conversations.
6. 218 human pairwise comparisons from 15 raters.

