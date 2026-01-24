# Stage Awareness Validation Report

**Generated:** 2026-01-24

The `stage_awareness` metric (0-2 scale) measures whether responses correctly
identify and adapt to the research stage (A-F).

---

## 1. Overall Stage Awareness by System

| System | n | Mean | Std | Perfect (≥1.95) | Good (≥1.5) | Low (<1.0) |
|--------|---|------|-----|-----------------|-------------|------------|
| MENTOR | 90 | 1.95 | ±0.20 | 90% | 97% | 1% |
| GEMINI | 90 | 1.96 | ±0.13 | 89% | 98% | 0% |
| GPT5 | 90 | 1.89 | ±0.33 | 78% | 93% | 2% |
| CLAUDE | 90 | 1.88 | ±0.25 | 72% | 92% | 1% |

---

## 2. Stage Awareness by Stage

### Stage A: Pre-idea (Orientation)

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 2.00 | 100% | 0% |
| GEMINI | 15 | 2.00 | 100% | 0% |
| GPT5 | 15 | 1.94 | 93% | 0% |
| CLAUDE | 15 | 1.98 | 100% | 0% |

### Stage B: Idea (Feasibility)

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 1.98 | 100% | 0% |
| GEMINI | 15 | 1.98 | 100% | 0% |
| GPT5 | 15 | 1.87 | 93% | 0% |
| CLAUDE | 15 | 1.87 | 93% | 0% |

### Stage C: Research Plan

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 2.00 | 100% | 0% |
| GEMINI | 15 | 1.96 | 93% | 0% |
| GPT5 | 15 | 1.69 | 87% | 13% |
| CLAUDE | 15 | 1.96 | 100% | 0% |

### Stage D: First Draft (Methodology)

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 1.98 | 100% | 0% |
| GEMINI | 15 | 2.00 | 100% | 0% |
| GPT5 | 15 | 2.00 | 100% | 0% |
| CLAUDE | 15 | 1.84 | 93% | 7% |

### Stage E: Second Draft (Discussion)

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 1.73 | 80% | 7% |
| GEMINI | 15 | 1.84 | 100% | 0% |
| GPT5 | 15 | 1.86 | 93% | 0% |
| CLAUDE | 15 | 1.74 | 80% | 0% |

### Stage F: Final (Venue/Release)

| System | n | Mean | Good Rate | Low Rate |
|--------|---|------|-----------|----------|
| MENTOR | 15 | 1.98 | 100% | 0% |
| GEMINI | 15 | 1.96 | 93% | 0% |
| GPT5 | 15 | 1.96 | 93% | 0% |
| CLAUDE | 15 | 1.91 | 87% | 0% |

---

## 3. Potential Stage Misclassifications

Found **18** responses with stage_awareness < 1.5:

- **Critical (< 1.0):** 4 cases
- **Low (1.0-1.5):** 14 cases

### Worst Cases (score < 1.0)

| System | Stage | Prompt | Score |
|--------|-------|--------|-------|
| gpt5 | C | stage_c_13 | 0.00 |
| gpt5 | C | stage_c_12 | 0.00 |
| mentor | E | stage_e_10 | 0.67 |
| claude | D | stage_d_01 | 0.67 |

---

## 4. Key Findings

- **MENTOR** achieves 1.95/2.0 mean stage awareness
- MENTOR has **3** responses with low stage awareness
- Hardest stage for MENTOR: **E** (1.73)
- Easiest stage for MENTOR: **C** (2.00)