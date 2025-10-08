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
    "question_quality": MetricSpec(
        "question_quality",
        "2.0: targeted clarifying questions grounded in context. 1.0: relevant but generic questions. 0.0: missing or counterproductive questions.",
        "scaled",
        0.0,
        2.0,
    ),
    "citation_quality": MetricSpec(
        "citation_quality",
        "2.0: citations real, recent, and directly support claims. 1.0: citations real but outdated or tangential. 0.0: missing, hallucinated, or irrelevant citations.",
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
}


def metric_instruction(spec: MetricSpec) -> str:
    if spec.kind == "binary":
        return "Return JSON {\"score\": <0 or 1>, \"rationale\": <string>, \"confidence\": <high|medium|low>}"
    return (
        "Return JSON {\"score\": <float between "
        f"{spec.min_score} and {spec.max_score}>, \"rationale\": <string>, \"confidence\": <high|medium|low>}}"
    )
