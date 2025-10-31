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
        "1.0: concrete executable steps with commands, parameters, and expected outcomes. 0.8: clear next steps with minor gaps. 0.6: clear direction but user must fill gaps. 0.4: generic suggestions. 0.2: vague advice. 0.0: unusable guidance.",
        "scaled",
        0.0,
        1.0,
    ),
    # Additional Agentic Capability metrics from eval plan
    "rag_fidelity": MetricSpec(
        "rag_fidelity",
        "2.0: synthesizes cited evidence accurately with clear attribution and no contradictions; 1.0: largely faithful with minor omissions or heuristic thresholds; 0.5: general best-practice guidance with limited grounding but no fabrications; 0.0: hallucinated, contradicts evidence, or ignores cited material.",
        "scaled",
        0.0,
        2.0,
    ),
    "citation_presence": MetricSpec(
        "citation_presence",
        "Return 1 when the final answer includes inline citations or a citations section; 0 otherwise.",
        "binary",
        0.0,
        1.0,
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
    "question_quality": MetricSpec(
        "question_quality",
        "2.0: targeted clarifying questions grounded in context. 1.0: relevant but generic questions. 0.0: missing or counterproductive questions.",
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
    "asks_questions": MetricSpec(
        "asks_questions",
        "Return 1 when the agent asks clarifying questions in ambiguous cases; 0 otherwise.",
        "binary",
        0.0,
        1.0,
    ),
    "tone_constructive": MetricSpec(
        "tone_constructive",
        "2.0: constructive, motivating tone; 1.0: neutral; 0.0: discouraging or harsh.",
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
        "Return 1 when every expected tool was invoked at least once. Return 0 when any expected tool is missing.",
        "binary",
        0.0,
        1.0,
    ),
    "constraint_handling": MetricSpec(
        "constraint_handling",
        "Return 1 when the response acknowledges constraints and adapts advice. Return 0 otherwise.",
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
        "2.0: three specific, sequenced steps executable in 1–3 days with concrete resources; 1.0: some concrete steps with gaps; 0.0: generic or unexecutable next steps.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_clarity": MetricSpec(
        "student_clarity",
        "2.0: easy to follow; steps and rationale are unambiguous; 1.0: partly clear with missing links; 0.0: unclear or overwhelming.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_constraint_fit": MetricSpec(
        "student_constraint_fit",
        "2.0: respects persona time/compute/skills constraints; 1.0: minor mismatches; 0.0: unrealistic for persona.",
        "scaled",
        0.0,
        2.0,
    ),
    "student_confidence_gain": MetricSpec(
        "student_confidence_gain",
        "2.0: clearly reduces uncertainty and increases confidence to proceed; 1.0: some reassurance; 0.0: no meaningful confidence change.",
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
}


def metric_instruction(spec: MetricSpec) -> str:
    if spec.kind == "binary":
        return "Return JSON {\"score\": <0 or 1>, \"rationale\": <string>, \"confidence\": <high|medium|low>}"
    return (
        "Return JSON {\"score\": <float between "
        f"{spec.min_score} and {spec.max_score}>, \"rationale\": <string>, \"confidence\": <high|medium|low>}"
    )
