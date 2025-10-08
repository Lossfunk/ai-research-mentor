# Evaluation Rubrics

This reference defines scoring guidance for manual annotations and automated heuristics. Each metric used in `evals_single_turn.jsonl` is listed with its scale and acceptance criteria.

## Shared Scales

### Actionability (0.0–1.0)
- **1.0** – Fully actionable: concrete commands, parameters, timelines, expected outcomes.
- **0.8–0.9** – Highly actionable: specific tools and next steps with minor gaps.
- **0.6–0.7** – Moderately actionable: clear direction but requires user to fill execution details.
- **0.4–0.5** – Partially actionable: general suggestions lacking methods.
- **0.2–0.3** – Minimally actionable: vague advice with unclear steps.
- **0.0–0.1** – Non-actionable: abstract guidance only.

### Question Quality (0.0–2.0)
- **2.0** – Targeted clarifying questions grounded in user context; directly improve recommendations.
- **1.0** – Relevant but generic questions; limited personalization.
- **0.0** – Irrelevant, missing, or counterproductive questions.

### Citation Quality (0.0–2.0)
- **2.0** – Citations real, recent, and directly support guidance.
- **1.0** – Citations real but outdated or tangential.
- **0.0** – Missing, hallucinated, or irrelevant citations.

## Binary / Heuristic Metrics

- **tool_routing** – Expected tools invoked and no disallowed tools used.
- **constraint_handling** – Explicit acknowledgment of stated constraint plus tailored mitigation path.
- **timeline_guidance** – Provides schedule-aware plan with milestones within supplied deadline.
- **expectation_management** – Sets realistic boundaries, flags infeasible goals, and proposes alternatives.
- **novelty_assessment** – Distinguishes prior work vs. new contributions using literature evidence.
- **evidence_gap_detection** – Identifies missing experiments or knowledge gaps the user should address.
- **hypothesis_generation** – Produces testable hypotheses with measurable outcomes.
- **distractor_rejection** – Recognizes injected irrelevant documents and avoids using them as evidence.
- **experiment_design** – Recommends concrete experiments or ablations with variables and metrics.
- **scope_feasibility** – Flags unrealistic project scope and scales recommendations appropriately.
- **feasibility_analysis** – Evaluates idea viability considering skills, data, and compute requirements.
- **skills_gap_guidance** – Suggests skill-building steps or adjusted plans for capability mismatches.
- **domain_mapping** – Connects cross-domain idea elements and highlights domain-specific needs.
- **risk_analysis** – Notes technical or ethical risks with mitigation strategies.
- **plan_completeness** – Includes hypotheses, methodology, evaluation metrics, resources, and milestones.
- **resource_estimation** – Specifies datasets, compute (GPU/CPU), tooling, or budget requirements.
- **timeline_quality** – Provides sequenced timeline with duration estimates and dependencies.
- **risk_mitigation** – Lists foreseeable blockers with contingency plans.

## Logging Metadata Aids

Annotators should capture supporting citations, tool traces, and notable omissions alongside scores. Disagreements >0.5 on rubric-based metrics trigger adjudication by a third reviewer.
