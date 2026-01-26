# Stage Awareness Ablation

## Research Question
**How much does stage-aware mentoring contribute to MENTOR's performance?**

## Ablation Design

### Conditions

| Condition | System Prompt | Stage Directives | Description |
|-----------|---------------|------------------|-------------|
| **Full MENTOR** | Stage-aware | Enabled | Baseline (existing results) |
| **-Stage Prompt** | Generic | Enabled | Remove stage inference from prompt |
| **-Stage Directives** | Stage-aware | Disabled | Remove Stage C special instructions |
| **-All Stage** | Generic | Disabled | Remove all stage awareness |

### Prompt Modifications

**Full MENTOR (original):**
```
What stage are they at (exploring, have an idea, writing)?
```

**-Stage Prompt (ablated):**
```
[Removed - no stage inference guidance]
```

### Test Set
- 3 prompts per stage (A-F) = 18 prompts total
- Selected to represent typical queries at each stage
- Same prompts used for all ablation conditions

### Metrics
- Holistic score (primary)
- Stage awareness score (expected to drop in ablated conditions)
- Win rate vs Full MENTOR

## Baseline Results (Full MENTOR on 18 prompts)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| holistic_score | **1.547** | 1.417 | 1.683 |
| stage_awareness | **1.889** | 0.667 | 2.0 |
| actionability | 1.259 | 0.833 | 1.5 |
| clarification_quality | 1.389 | 0.667 | 2.0 |

### By Stage
| Stage | Holistic | Stage Awareness |
|-------|----------|-----------------|
| A (Pre-idea) | 1.500 | 2.00 |
| B (Idea) | 1.500 | 2.00 |
| C (Research Plan) | 1.583 | 2.00 |
| D (First Draft) | 1.589 | 2.00 |
| E (Second Draft) | 1.556 | 1.33 ← lowest |
| F (Final) | 1.556 | 2.00 |

## Files

- `prompt_no_stage.md` - Ablated system prompt (stage inference removed)
- `run_ablation.py` - Script to run ablation evaluation
- `select_prompts.py` - Script to extract test prompts
- `selected_prompts.json` - 18 prompts (3 per stage)
- `results/full_mentor_baseline.json` - Baseline results

## Expected Outcomes

If stage awareness helps significantly:
- `-Stage Prompt` should show lower stage_awareness and holistic scores
- `-All Stage` should show the largest drop
- Effect should be strongest for Stage C (which has explicit directives)

If stage awareness doesn't help much:
- Scores should be similar across conditions
- Would suggest the model naturally adapts without explicit guidance

## How to Run

```bash
# Setup only (no API calls)
uv run python run_ablation.py

# Run full ablation (calls LLM API, ~$10-20)
RUN_ABLATIONS=1 uv run python run_ablation.py
```

## Cost Estimate

- 18 prompts × 3 conditions = 54 LLM calls
- ~$0.15-0.30 per call (Kimi-K2) = **~$8-16**
- Plus judge calls: 54 × 3 judges × 10 metrics = **~$10-15**
- **Total: ~$18-30**
