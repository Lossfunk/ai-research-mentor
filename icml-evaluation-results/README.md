# ICML 2026 Evaluation Results — MENTOR

This directory contains all evaluation data, scripts, and reports for the MENTOR paper submission to ICML 2026.

## Directory Structure

```
icml-evaluation-results/
├── README.md                    # This file
│
├── config/                      # Evaluation configuration
│   ├── metrics.yaml             # Metric definitions for judges
│   └── citation_domains.yaml    # Valid citation domain patterns
│
├── judges/                      # LLM judge prompt templates
│   ├── single_turn_holistic_prompt.md
│   ├── single_turn_absolute_prompt.md
│   ├── pairwise_judge_prompt.md
│   ├── holistic_conversation_judge.md
│   └── student_outcome_judge.md
│
├── prompts/                     # Evaluation input prompts
│   └── evals_single_turn.jsonl  # 90 single-turn prompts (15 per stage)
│
├── scripts/                     # Evaluation scripts
│   ├── *.py                     # Core evaluation orchestrators
│   ├── analysis/                # Post-hoc analysis scripts
│   ├── evals-for-papers-scripts/# Shell scripts for running stages
│   ├── repo-scripts/            # Utility scripts
│   └── multi_turn/              # Multi-turn scenario definitions
│
├── reports/                     # Generated reports & analysis
│   ├── icml_eval_gaps.md        # Master tracking document
│   ├── gap_analysis_report.md   # Statistical analysis summary
│   ├── cost_latency_report.md   # Cost & latency metrics
│   ├── stage_awareness_report.md# Stage detector validation
│   └── claims_evidence_audit.md # Claims ↔ evidence mapping
│
├── results/                     # Aggregated JSON results
│   ├── gap_analysis_results.json
│   ├── single_turn_holistic_results.json
│   ├── cost_latency_results.json
│   └── stage_awareness_results.json
│
├── raw_logs/                    # Raw LLM response text files
│   └── stage_{a-f}/             # Organized by research stage
│
├── analysis_reports/            # Per-prompt judge scores
│   ├── mentor/                  # MENTOR (Kimi-K2)
│   ├── claude-baseline/         # Claude Sonnet 4.5
│   ├── gpt5-baseline/           # GPT-5
│   └── gemini-baseline/         # Gemini 3 Pro
│
├── holistic_scoring_v2/         # Multi-turn holistic results
│   ├── holistic_results.csv     # Summary scores
│   └── *.png, *.svg, *.pdf      # Visualizations
│
├── human-baseline-votes/        # Human pairwise evaluations
│   └── *.csv                    # 15 rater files, 218 total comparisons
│
├── transcripts/                 # Multi-turn conversation logs
│   └── *.json                   # 80 conversations (20 per system)
│
├── student_terminations/        # Conversations ended by student agent
│   └── *.json                   # Success/satisfaction signals
│
├── ablations/                   # Ablation study results
│   ├── stage_ablation/          # -Stage awareness: -54.9%
│   └── guidelines_ablation/     # -Guidelines: -33.7%
│
└── inter_annotator_agreement/   # IAA analysis
    ├── iaa_report.json
    └── compute_iaa.py
```

## Key Results Summary

| Evaluation | Result |
|------------|--------|
| **Multi-turn holistic** (80 convos) | MENTOR: **1.705/2.0**, 100% success |
| **Human pairwise** (218 comparisons) | MENTOR wins **64.7%** overall |
| **Single-turn** (90 prompts) | MENTOR: **1.545/2.0** |
| **Stage ablation** | -54.9% without stage awareness |
| **Guidelines ablation** | -33.7% without mentor prompt |

## Systems Evaluated

| System | Model | Description |
|--------|-------|-------------|
| MENTOR | moonshotai/kimi-k2-thinking | Full system with stage awareness + guidelines |
| GPT-5 | openai/gpt-5 | Baseline with generic prompt |
| Claude | anthropic/claude-sonnet-4.5 | Baseline with generic prompt |
| Gemini | google/gemini-3-pro | Baseline with generic prompt |

## Reproducing Results

```bash
# Run single-turn evaluation for stage A
uv run python -m evaluation.scripts.run_manual_stage --stage a

# Run judges on results
uv run python -m evaluation.scripts.run_judge_scores --results-root icml-evaluation-results

# Generate analysis
uv run python icml-evaluation-results/scripts/analysis/compute_all_gaps.py
```

## License

Evaluation data is provided for reproducibility purposes as supplementary material.
