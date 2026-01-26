# Guidelines Ablation

## Research Question
**Does the curated research mentor prompt matter, or would a generic assistant perform equally well?**

## Ablation Design

### Conditions

| Condition | System Prompt | Description |
|-----------|---------------|-------------|
| **Full MENTOR** | Full mentor prompt | Critical feedback, stage awareness, response structure |
| **No-Guidelines** | Generic assistant | "You are a helpful research assistant" |

### What's Removed in No-Guidelines

The full MENTOR prompt includes:
- Critical evaluation philosophy
- Adaptive behavior based on user level
- Specific feedback format
- Length rules
- Research taste guidance
- Tool usage guidance

The ablated prompt is minimal:
- "You are a helpful research assistant"
- Basic response guidelines
- No structure requirements

### Test Set
- Same 18 prompts as stage ablation (3 per stage A-F)
- Same 3-judge ensemble

## How to Run

```bash
cd /Users/majortimberwolf/Projects/lossfunk/ai-research-mentor/academic-research-mentor

# Step 1: Generate responses
cd icml-evaluation-results/ablations/guidelines_ablation
RUN_ABLATIONS=1 uv run python run_ablation.py

# Step 2: Prepare for judges
uv run python prepare_for_judges.py

# Step 3: Run judges
cd /Users/majortimberwolf/Projects/lossfunk/ai-research-mentor/academic-research-mentor
for stage in a b c d e f; do
  uv run python -m evaluation.scripts.run_judge_scores \
    --stage stage_$stage \
    --judge openrouter:qwen/qwen3-max \
    --judge openrouter:deepseek/deepseek-v3.2-exp \
    --judge openrouter:x-ai/grok-4-fast \
    --annotator ablation_judge \
    --label ablation_no_guidelines \
    --results-root icml-evaluation-results/ablations/guidelines_ablation/judge_inputs/no_guidelines \
    --system-subdir .
done
```

## Expected Outcomes

If the curated mentor prompt matters:
- No-Guidelines should show lower holistic scores
- Especially lower on persona_compliance and clarification_quality

If it doesn't matter:
- Scores should be similar
- Would suggest the base model is sufficient
