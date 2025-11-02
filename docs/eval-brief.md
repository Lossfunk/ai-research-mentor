### Evaluation Brief

**Objective**: Demonstrate that the Mentor outperforms strong LLM baselines across stages (A–F) on quality metrics and pairwise preference, while remaining competitive on latency.

**Answering Systems**
- Mentor (tools + guidelines + memory)
- Baselines (no tools, no guidelines):
  - Claude Sonnet 4.5 (primary)
  - GPT‑5 (sensitivity check)

**Judge (LLM-as-a-judge)**
- Primary: Gemini 2.5 Pro (absolute + pairwise) to reduce self-family bias
- Optional: add Claude Sonnet 4.5 for inter-annotator agreement (IAA)

**Dataset & Protocol**
- Stages A–F; 6 prompts per stage
- Seeds: 1 and 2
- Temperature: 0.0 (deterministic); equal max output tokens implicitly matched
- Pairwise: all available system pairs per prompt/stage/seed

**Metrics (absolute)**
- Scaled: actionability, rag_fidelity, clarification_quality, persona_compliance, stage_awareness, tone_constructive, citation_relevance, source_fit, citation_quality
- Binary: citation_validity, fallback_robustness, evidence_integrity, plus stage-specific checks (timeline_guidance, expectation_management, etc. as applicable)

**Expected Outcomes**
- Overall: Mentor > Claude on weighted overall quality and pairwise win rate; competitive latency
- By stage:
  - A/B: higher clarification_quality, novelty_assessment, expectation_management
  - C: higher plan_completeness, experiment_design, resource_estimation
  - D/E: stronger methodology_critique, risk_analysis, evidence_gap_detection
  - F: stronger plan_completeness and expectation_management for submission readiness
- Meta: higher citation_validity and evidence_integrity; tool_routing monitored diagnostically where tools are expected

**Reporting**
- Per-stage CSV/Markdown tables, pairwise win rates, metric deltas, Pareto (quality vs latency)
- Sensitivity: replicate headline with GPT‑5 baseline and second seed; report IAA when dual judge used


