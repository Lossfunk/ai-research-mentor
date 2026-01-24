from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    description: str
    kind: str  # scaled | binary
    min_score: float
    max_score: float


METRIC_SPECS: dict[str, MetricSpec] = {
    "actionability": MetricSpec(
        "actionability",
        "2.0: concrete executable steps with commands, parameters, and expected outcomes; 1.5: clear next steps with minor gaps; 1.0: clear direction but user must fill important gaps; 0.5: generic suggestions; 0.0: vague or unusable guidance.",
        "scaled",
        0.0,
        2.0,
    ),
    # Additional Agentic Capability metrics from eval plan
    "rag_fidelity": MetricSpec(
        "rag_fidelity",
        "2.0: synthesizes cited evidence accurately with clear attribution and no contradictions; 1.0: largely faithful with minor omissions or heuristic thresholds; 0.5: general best-practice guidance with limited grounding but no fabrications; 0.0: hallucinated, contradicts evidence, or ignores cited material.",
        "scaled",
        0.0,
        2.0,
    ),
    "citation_validity": MetricSpec(
        "citation_validity",
        "Return 1 when cited sources are valid scholarly links/records (e.g., arXiv, DOI, publisher); 0 when clearly invalid or hallucinated.",
        "binary",
        0.0,
        1.0,
    ),
    "citation_relevance": MetricSpec(
        "citation_relevance",
        "2.0: citations directly support claims made; 1.0: tangential but related; 0.0: irrelevant.",
        "scaled",
        0.0,
        2.0,
    ),
    "source_fit": MetricSpec(
        "source_fit",
        "2.0: sources appropriate for user goal and expertise (recency, venue); 1.0: acceptable but suboptimal; 0.0: poor fit.",
        "scaled",
        0.0,
        2.0,
    ),
    "clarification_quality": MetricSpec(
        "clarification_quality",
        "2.0: targeted clarifying questions or explicit assumptions that materially improve guidance quality; 1.5: useful probes or stated assumptions with minor gaps; 1.0: optional clarifications present or clear assumptions stated when none were needed; 0.5: generic probes that add little value; 0.0: needed clarifications are missing AND no assumptions are stated, leaving guidance ambiguous.",
        "scaled",
        0.0,
        2.0,
    ),
    # Mentorship capability metrics from eval plan
    "persona_compliance": MetricSpec(
        "persona_compliance",
        "2.0: consistently encouraging, guiding mentor persona; 1.0: neutral or mixed tone; 0.0: dismissive or answer-only persona.",
        "scaled",
        0.0,
        2.0,
    ),
    "stage_awareness": MetricSpec(
        "stage_awareness",
        "2.0: response clearly recognizes the user's research stage and tailors guidance; 1.0: partially aligned; 0.0: misaligned (e.g., jumps ahead of stage).",
        "scaled",
        0.0,
        2.0,
    ),
    "tone_constructive": MetricSpec(
        "tone_constructive",
        "2.0: constructive, motivating tone that reinforces progress without resorting to fluff; 1.0: neutral or mildly encouraging; 0.0: discouraging, dismissive, or fear-inducing language.",
        "scaled",
        0.0,
        2.0,
    ),
    # Robustness
    "fallback_robustness": MetricSpec(
        "fallback_robustness",
        "Return 1 when, after a tool error or degraded mode, the agent provides an alternate usable path; 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "citation_quality": MetricSpec(
        "citation_quality",
        "2.0: citations are real, well-formatted, and directly support claims (scholarly or authoritative guideline/portal). 1.0: citations real but less authoritative (e.g., blogs/portals) or partially aligned. 0.0: missing, fabricated, or clearly irrelevant citations.",
        "scaled",
        0.0,
        2.0,
    ),
    "tool_routing": MetricSpec(
        "tool_routing",
        "Diagnostic only: return 1 when every expected tool was invoked at least once, 0 when an expected tool is missing. If no tools were expected, return null instead of forcing a score.",
        "binary",
        0.0,
        1.0,
    ),
    "timeline_guidance": MetricSpec(
        "timeline_guidance",
        "Return 1 when schedule-aware milestones respect the supplied deadline. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "expectation_management": MetricSpec(
        "expectation_management",
        "Return 1 when the response sets realistic expectations or reframes infeasible goals. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "novelty_assessment": MetricSpec(
        "novelty_assessment",
        "Return 1 when literature is analysed to judge novelty, highlighting overlaps and differentiators. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "evidence_gap_detection": MetricSpec(
        "evidence_gap_detection",
        "Return 1 when missing experiments or validation steps are identified. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "hypothesis_generation": MetricSpec(
        "hypothesis_generation",
        "Return 1 when at least one testable hypothesis with measurable outcomes is proposed. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "distractor_rejection": MetricSpec(
        "distractor_rejection",
        "Return 1 when distractor documents are ignored or flagged as irrelevant. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "experiment_design": MetricSpec(
        "experiment_design",
        "Return 1 when concrete experiments or ablations with variables and metrics are proposed. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "scope_feasibility": MetricSpec(
        "scope_feasibility",
        "Return 1 when scope is right-sized for available resources. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "feasibility_analysis": MetricSpec(
        "feasibility_analysis",
        "Return 1 when feasibility is evaluated across skills, data, and compute. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "skills_gap_guidance": MetricSpec(
        "skills_gap_guidance",
        "Return 1 when the response offers skill-building steps or adjusted plans for capability gaps. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "domain_mapping": MetricSpec(
        "domain_mapping",
        "Return 1 when cross-domain connections are mapped accurately with domain-specific needs. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "risk_analysis": MetricSpec(
        "risk_analysis",
        "Return 1 when technical or ethical risks are noted with mitigation ideas. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "plan_completeness": MetricSpec(
        "plan_completeness",
        "Return 1 when hypotheses, methodology, evaluation, resources, and milestones are all present. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "resource_estimation": MetricSpec(
        "resource_estimation",
        "Return 1 when datasets, compute, or tooling requirements are estimated. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "timeline_quality": MetricSpec(
        "timeline_quality",
        "Return 1 when activities are sequenced with durations or dependencies. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "risk_mitigation": MetricSpec(
        "risk_mitigation",
        "Return 1 when risks are paired with mitigation strategies. Return 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    # ---------------------- Student-outcome metrics (style-agnostic) ----------------------
    "student_actionability": MetricSpec(
        "student_actionability",
        "2.0: at least three concrete, sequenced steps anchored in the response (datasets, tools, deliverables) that a student could start within a few days; 1.0: some concrete actions but gaps or missing anchors; 0.0: generic suggestions with no executable steps.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_clarity": MetricSpec(
        "student_clarity",
        "2.0: instructions are easy to follow, explain why each step matters, and reference persona details; 1.0: partially clear but forces the student to infer rationale or ordering; 0.0: confusing, overwhelming, or contradictory guidance.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_constraint_fit": MetricSpec(
        "student_constraint_fit",
        "2.0: explicitly respects the persona's time, compute, and skill constraints using details from the prompt; 1.0: mostly aligned but overlooks at least one stated constraint; 0.0: unrealistic or dismissive of constraints.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_confidence_gain": MetricSpec(
        "student_confidence_gain",
        "2.0: directly addresses likely anxieties, sets expectations, and explains why the plan will work; 1.0: some reassurance but limited specificity; 0.0: no meaningful confidence gain or introduces new doubt.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_path_ready": MetricSpec(
        "student_path_ready",
        "Return 1 when a student could start immediately without major missing prerequisites; 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "student_failure_modes": MetricSpec(
        "student_failure_modes",
        "Return 1 when likely pitfalls or blockers are explicitly flagged; 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "student_outcome_score": MetricSpec(
        "student_outcome_score",
        "Composite outcome for students: 0.35*actionability + 0.25*clarity + 0.25*constraint_fit + 0.15*confidence_gain.",
        "scaled",
        0.0,
        2.0,
    ),
    # ---------------------- Holistic single-turn metric ----------------------
    "holistic_score": MetricSpec(
        "holistic_score",
        "Holistic assessment of the complete response as a user experience. 2.0: exceptional and rare (<10%); 1.5: good, clear actionable guidance with minor gaps; 1.0: adequate, reasonable direction but notable gaps; 0.5: minimally helpful, generic advice; 0.0: unhelpful or misleading. Use single_turn_holistic_prompt.md which requires weakness identification.",
        "scaled",
        0.0,
        2.0,
    ),
}


def metric_instruction(spec: MetricSpec) -> str:
    if spec.kind == "binary":
        return 'Return JSON {"score": <0 or 1>, "rationale": <string>, "confidence": <high|medium|low>}'
    return (
        f'Return JSON {{"score": <float between {spec.min_score} and {spec.max_score}>, '
        f'"rationale": <string>, "confidence": <high|medium|low>}}'
    )
