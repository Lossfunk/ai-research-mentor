# Cost & Latency Analysis

**Generated:** 2026-01-24

---

## 1. Multi-Turn Conversations (Wall-Clock Time)

| System | Conversations | Total Time | Avg Time | Avg Turns | Time/Turn |
|--------|---------------|------------|----------|-----------|-----------|
| MENTOR | 20 | 149.0 min | 447s | 39 | 11.2s |
| GEMINI | 20 | 459.3 min | 1378s | 36 | 38.2s |
| GPT5 | 20 | 647.0 min | 1941s | 30 | 67.9s |
| CLAUDE | 20 | 820.7 min | 2462s | 36 | 63.6s |

### Detailed Statistics

| System | Min Time | Max Time | Std Dev |
|--------|----------|----------|---------|
| MENTOR | 58s | 1047s | ±322s |
| GEMINI | 363s | 3897s | ±1068s |
| GPT5 | 202s | 4630s | ±1407s |
| CLAUDE | 143s | 5417s | ±1892s |

---

## 2. Single-Turn Token Usage (Judge Evaluation)

| System | Prompts | Total Tokens | Avg Input | Avg Output | Avg Total |
|--------|---------|--------------|-----------|------------|-----------|
| MENTOR | 90 | 5,297,431 | 47,858 | 11,003 | 58,860 |
| GEMINI | 90 | 7,813,710 | 74,429 | 12,390 | 86,819 |
| GPT5 | 90 | 8,445,705 | 81,340 | 12,501 | 93,841 |
| CLAUDE | 90 | 12,617,612 | 127,218 | 12,977 | 140,196 |

---

## 3. Key Findings

### Multi-Turn Efficiency
- **MENTOR**: 447s avg (1.0x vs MENTOR)
- **GEMINI**: 1378s avg (3.1x vs MENTOR)
- **GPT5**: 1941s avg (4.3x vs MENTOR)
- **CLAUDE**: 2462s avg (5.5x vs MENTOR)

### Observations
- MENTOR completes conversations in **447s** average
- Claude is **5.5x slower** than MENTOR
- MENTOR averages **39 turns** per conversation
- Time per turn: MENTOR (11.2s) vs Claude (63.6s)
