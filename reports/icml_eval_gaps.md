# ICML Evaluation Gaps and Additions (MENTOR)

This list consolidates remaining gaps from the AAAI workshop reviews plus additional evaluation upgrades aligned with ICML review expectations.

## A. Still-missing items from AAAI workshop reviews

- [ ] **Human evaluation: add comparative + rubric-aligned calibration**: A 50-person satisfaction/utility survey exists; remaining gap is comparative or task-based human scoring aligned to the rubric (IAA + judge–human correlation) to validate LLM-judge metrics.
- [ ] **Multi-turn sample size is too small**: Expand multi-turn scenarios well beyond 5; diversify topics, personas, and difficulty; report CIs and sensitivity to dialogue length.
- [ ] **Success threshold is arbitrary**: The “overall ≥ 1.6” success definition needs justification; run threshold sensitivity analysis and/or set threshold based on human-validated anchors.
- [ ] **Baselines are weak/mismatched**: Compare against closer baselines (agentic systems for research, tool-augmented frameworks, or staged mentorship baselines) not just general LLMs.
- [ ] **Ablations are missing**: Add component ablations for guidelines/pipeline/tools/memory/persona to isolate contributions.
- [ ] **Failure modes are under-specified**: Define how failures are flagged, include a taxonomy, and quantify false positives/negatives.
- [ ] **Cross-session context is unclear**: Explain how memory/context persists across sessions and evaluate its impact.
- [ ] **Metric weights lack validation**: The weighted overall score (0.35/0.25/0.25/0.15) needs sensitivity analysis or alternative weighting checks.

## B. ICML-aligned evaluation additions (from official review criteria)

These items map to the ICML reviewer criteria emphasizing sound evidence, valid experimental design, and reproducibility.

- [ ] **Claims ↔ evidence audit**: For each key claim, add explicit evidence mapping, effect sizes, and statistical tests; clarify which claims are exploratory vs. confirmatory.
- [ ] **Experimental design validation**: Add robustness checks (seeds, temperature, prompt variants), and document assumptions/controls so the design is defensible.
- [ ] **Broader baseline coverage**: Include at least one strong agentic/structured baseline and one retrieval/tool-augmented baseline; justify why each baseline is appropriate.
- [ ] **OOD/generalization checks**: Add domain shift (new disciplines), new personas, longer contexts, and adversarial prompts to test generality.
- [ ] **Judge reliability analysis**: Inter-judge agreement and bias checks; show judge-human correlation where possible.
- [ ] **Reproducibility artifacts**: Release prompts, configs, seeds, model versions, and evaluation scripts; document cost/latency for reproducibility.
- [ ] **Ethics/limitations completeness**: Add concrete risk analysis (e.g., misinformation, inappropriate guidance) and mitigation tests.

## C. Items noted in the preprint but not yet evidenced in results

These are mentioned in `text.tex` but are not clearly demonstrated in current evaluation artifacts.

- [ ] **Full ablation suite**: guidelines/tools/memory/persona/stage detector.
- [ ] **Robustness & sensitivity**: seeds, tool degradation, OOD personas, long-context stress.
- [ ] **Error analysis with rates**: taxonomy + quantitative breakdown.
- [ ] **Human calibration**: human subset with IAA + judge–human correlation.

## Sources (ICML review expectations)

- ICML Reviewer Instructions 2024 (main-track review form; evaluation, soundness, limitations, and rating criteria).
- ICML Reviewer Instructions 2025 (claims & evidence, experimental design validity, relation to prior work).
- ICML Reviewer FAQ 2020 (reproducibility checklist reference).
