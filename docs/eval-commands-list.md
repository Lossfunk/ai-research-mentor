# Evaluation Command Reference

Commands below assume you are in the repository root and running inside the `uv` environment. Replace stage identifiers (`stage_a`, `stage_b`, …), system IDs, and paths as needed.

## 1. Single-Turn Evaluation (Mentor Prompt)
Generates mentor responses and tool traces for a stage.
```bash
uv run python -m evaluation.scripts.run_manual_stage --stage stage_b --force
```
Use `--prompt-id <id>` to limit to specific prompts.

### Baseline Prompt (Guidelines Off)
Runs the same prompts with baseline settings and (optionally) judges.
```bash
uv run python scripts/eval_baseline.py --stage stage_b --skip-judges --force
```
Omit `--skip-judges` and add `--judge <provider:model>` flags to score the baseline in the same pass.

## 2. Single-Turn Judges (Expert Persona)
Scores existing single-turn responses with the expert “absolute” rubric.
```bash
uv run python -m evaluation.scripts.run_judge_scores \
  --stage stage_b \
  --system mentor_manual \
  --annotator expert_judge_upgrade \
  --label expert_absolute_pro \
  --judge openrouter:google/gemini-2.5-flash \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --force
```
Add `--prompt-id <id>` to limit scoring to specific prompts.

## 3. Student Judges (Student Persona)
Scores single-turn responses with the student outcome rubric.
```bash
uv run python -m evaluation.scripts.run_student_judge_scores \
  --stage stage_b \
  --system mentor_manual \
  --annotator student_judge_upgrade \
  --label student_outcome_judge \
  --judge openrouter:google/gemini-2.5-flash \
  --judge openrouter:deepseek/deepseek-v3.2-exp \
  --judge openrouter:x-ai/grok-4-fast \
  --force
```

## 4. Multi-Turn Evaluations
Runs synthetic multi-turn conversations for one or more mentor systems.
```bash
uv run python -m evaluation.scripts.run_multi_turn_batch \
  --input evaluation/multi_turn_scenarios.jsonl \
  --mentors mentor_manual \
  --student-model google/gemini-2.5-flash-lite \
  --max-turns 3 \
  --output evaluation/results/multi_turn_mentor
```
Provide your own scenario JSONL (`--input`) and list any additional mentors or tool whitelist options as required.

## 5. LOFO Judge Analysis
Computes Leave-One-Family-Out summaries for a stage/judge label.
```bash
uv run python -m evaluation.scripts.analyze_lofo \
  --stage stage_b \
  --label student_outcome_judge \
  --output evals-for-papers/results/analysis_reports/stage_b/student_outcome_judge/lofo_summary.json
```
Omit `--output` to write the summary alongside the label directory automatically.
