# ICML Evaluation Gaps and Additions (MENTOR)

This list consolidates remaining gaps from the AAAI workshop reviews plus additional evaluation upgrades aligned with ICML review expectations.

**Last updated:** 2026-01-24

---

## A. AAAI Workshop Review Gaps — Status

| Item | Status | Notes |
|------|--------|-------|
| Human evaluation (comparative) | ✅ DONE | **218 pairwise comparisons, 15 human raters.** MENTOR wins **64.7%** overall (79.7% vs Claude, 58.7% vs GPT-5, 53.1% vs Gemini) |
| Multi-turn sample size | ✅ DONE | Expanded from 5 → 20 scenarios. Changed from per-turn to holistic scoring (first turn, final turn, overall) |
| Success threshold arbitrary | ✅ DONE | Threshold sensitivity analysis: MENTOR 95% at 1.6 (vs GPT-5 65%, Claude 25%). See `gap_analysis_report.md` |
| Baselines weak/mismatched | ⚠️ PARTIAL | Added Gemini 3 Pro as 4th baseline. Still missing agentic/tool-augmented baselines |
| Ablations missing | ❌ TODO | Need component ablations: guidelines, tools, memory, stage routing, response structure |
| Failure modes under-specified | ✅ DONE | Taxonomy computed: missed_constraints (30%), resource_awareness (20%), redundancy (16%). See `gap_analysis_report.md` |
| Cross-session context unclear | ❌ TODO | Explain how memory/context persists; evaluate its impact |
| Metric weights lack validation | ⚠️ PARTIAL | Threshold sensitivity done; metric weight sensitivity still needed |

---

## B. ICML-Aligned Additions — Status

| Item | Status | Notes |
|------|--------|-------|
| Claims ↔ evidence audit | ✅ DONE | `icml-evaluation-results/claims_evidence_audit.md` — maps claims to data, identifies gaps |
| Experimental design validation | ❌ TODO | Seeds, temperature, prompt variants |
| Broader baseline coverage | ⚠️ PARTIAL | Added Gemini 3 Pro; still no agentic/structured baselines |
| OOD/generalization checks | ❌ TODO | Domain shift, new personas, adversarial prompts |
| Judge reliability (IAA) | ✅ DONE | Mean pairwise correlation **r=0.49**, 3-judge ensemble. Human distributed design (no Fleiss' Kappa applicable) |
| Reproducibility artifacts | ⚠️ PARTIAL | Have scripts; need to release prompts, configs, seeds, model versions |
| Ethics/limitations | ⚠️ PARTIAL | Mentioned in abstract; needs concrete risk analysis |

---

## C. Additional ICML Risks (from external review)

These are additional failure modes ICML reviewers will likely target:

### C.1 Contribution clarity and novelty boundary
- Core delta reads as "prompted router + curated guidelines + RAG + checklists" with no learned component
- ICML-safe positioning requires: stronger empirical claim (validated methodology) OR stronger technical claim (learned routing) OR benchmark contribution

### C.2 Stage detector validity
- ✅ **VALIDATED** — See `icml-evaluation-results/stage_awareness_report.md`
- MENTOR: **1.95/2.0** mean stage awareness (90% perfect, 97% good)
- Only **3 responses** with low stage awareness out of 90
- Weakest: Stage E (Discussion) at 1.73 mean; all others ≥1.98
- GPT-5 has the most misclassifications (4 critical cases)

### C.3 Judge robustness controls
Beyond "we used 3 judges," reviewers will ask for:
- [ ] A/B position swapping to measure order bias
- [ ] Length normalization to quantify verbosity bias
- [ ] Inter-judge agreement (pairwise + rubric)
- [ ] Correlation to human subset

### C.4 Tool parity and confounds
METIS has extra guidelines tool + tighter structure vs baselines. Need:
- [ ] Tool-parity baselines (give GPT-5/Claude the same guidelines retrieval), OR
- [ ] METIS-minus-X ablations (no guidelines, no methodology checks, no memory, no stage routing, no Intuition/Why blocks)

### C.5 Cost/latency profile
- ✅ **DONE** — See `icml-evaluation-results/cost_latency_report.md`
- MENTOR is **5.5x faster** than Claude (447s vs 2462s avg)
- MENTOR uses **2.4x fewer tokens** (58K vs 140K per prompt)
- MENTOR: 11.2s per turn vs Claude: 63.6s per turn

---

## D. Ablation Suite (Priority)

If time is limited, these are the highest-leverage ablations:

| Ablation | What to remove | What it tests |
|----------|----------------|---------------|
| METIS − Guidelines | Research Guidelines tool | Does curated advice matter? |
| METIS − Stage routing | Use flat routing (no stage detection) | Does stage-awareness matter? |
| METIS − Memory | Session memory | Does context persistence matter? |
| METIS − Structure | Intuition/Why blocks | Does response structure matter? |
| Tool-parity baseline | Give GPT-5 the guidelines tool | Is gain from tools or from METIS? |

---

## E. User Study Summary

- **n = 50** participants
- **Helpfulness:** 4.3/5 mean
- **Reuse intent:** 90% Yes, 10% Maybe, 0% No
- **Time spent:** 16-32 hours (median ~24)
- **Note:** Descriptive only (no randomization/blinding). Duration not logged in CSV.

---

## F. Current Evidence Summary (for abstract/claims)

| Evaluation | Result |
|------------|--------|
| Multi-turn holistic (LLM judges, n=80) | **MENTOR: 1.705**, 100% success (vs Gemini 1.670, GPT-5 1.631, Claude 1.493) |
| Human pairwise (218 comparisons, 15 raters) | **MENTOR wins 64.7%** overall (79.7% vs Claude, 58.7% vs GPT-5, 53.1% vs Gemini) |
| User study (n=50) | 4.3/5 helpfulness, 90% reuse intent |
| Single-turn (90 prompts, 4 systems) | MENTOR #2 overall (1.545), beats Claude 60.9%. See `single_turn_holistic_results.json` |

---

## G. Inter-Annotator Agreement (IAA) Summary

### LLM Judges (3-judge ensemble on 80 multi-turn conversations)

| Metric | Mean Pairwise r | ICC |
|--------|-----------------|-----|
| Overall Helpfulness | 0.52 | 0.49 |
| Student Progress | 0.40 | 0.36 |
| Mentor Effectiveness | 0.48 | 0.46 |
| Conversation Efficiency | 0.58 | 0.53 |
| **Mean** | **0.49** | **0.46** |

**Design rationale:** We use 3 diverse LLM judges (Qwen3-Max, DeepSeek-v3.2-exp, Grok-4-Fast) to enable majority-based aggregation and reduce single-judge bias.

**For the paper:**
> "We employ a 3-judge ensemble to enable majority-based aggregation. Judges show moderate pairwise agreement (mean r=0.49), with scores averaged to reduce individual bias."

### Human Raters (15 raters, 218 comparisons)

**Design:** Distributed evaluation — each rater evaluated a distinct subset of ~10-20 pairwise comparisons.

**Why no Fleiss' Kappa:** Traditional IAA metrics require all raters to evaluate the same items. In our design, comparisons were distributed across raters, making Fleiss' Kappa invalid. We report aggregate win rates instead.

| Matchup | n | MENTOR Win Rate |
|---------|---|-----------------|
| MENTOR vs Claude | 79 | **79.7%** |
| MENTOR vs GPT-5 | 75 | **58.7%** |
| MENTOR vs Gemini | 64 | **53.1%** |
| **Overall** | 218 | **64.7%** |

**Rater calibration indicators:**
- Mean mentor preference per rater: 60.5%
- Std: 30.0% (expected variance given different item sets)
- Ties: 8 (3.7%)

**For the paper:**
> "Human pairwise evaluation (218 comparisons, 15 raters) shows MENTOR preferred over Claude (79.7%), GPT-5 (58.7%), and comparable to Gemini (53.1%). Each rater evaluated distinct comparison pairs; we report aggregate statistics rather than inter-rater agreement, as the distributed design does not support traditional IAA metrics."

---

## Sources

- ICML Reviewer Instructions 2024/2025 (claims & evidence, experimental design, limitations)
- ICML 2026 Call for Papers: abstract deadline Jan 23 AoE, paper deadline Jan 28 AoE
- Neel Nanda, "Highly Opinionated Advice on How to Write ML Papers" (alignmentforum.org)
