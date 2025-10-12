import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.scripts.judge_utils import (
    apply_evidence_integrity,
    extract_citations_from_answer,
    score_citation_validity,
    score_citation_validity_v2,
)


def test_extract_citations_parses_stage_example() -> None:
    answer = (
        "Some guidance.\n\n"
        "Citations\n"
        "[G1] A Beginner's Guide to AI Research — https://haofei.vip/downloads/AI-lecture-Hao-publish.pdf\n"
        "[G2] What Advice Do I Give To My Students — https://thoughtforms.life/what-advice-do-i-give-to-my-students/\n"
        "[P1] arXiv Sample Paper — https://arxiv.org/abs/2412.05683\n"
    )

    citations = extract_citations_from_answer(answer)

    assert len(citations) == 3
    assert citations[0].id == "G1"
    assert citations[2].url.endswith("2412.05683")


def test_score_citation_validity_v2_counts_domains() -> None:
    answer = (
        "Citations\n"
        "[G1] A Beginner's Guide to AI Research — https://haofei.vip/downloads/AI-lecture-Hao-publish.pdf\n"
        "[G2] What Advice — https://thoughtforms.life/what-advice-do-i-give-to-my-students/\n"
        "[P1] arXiv Sample Paper — https://arxiv.org/abs/2412.05683\n"
    )

    citations = extract_citations_from_answer(answer)
    result = score_citation_validity_v2(citations)

    details = result["details"]
    assert details["total_count"] == 3
    assert details["scholarly_count"] == 1
    assert details["guideline_count"] >= 1
    assert result["score"] == 1.0


def test_invalid_links_reduce_score() -> None:
    answer = "Citations\n[G1] Example — https://example.com/path\n"

    citations = extract_citations_from_answer(answer)
    result = score_citation_validity_v2(citations)

    assert result["details"]["total_count"] == 1
    assert result["score"] == 0.0


def test_apply_evidence_integrity_gates_metrics() -> None:
    metric_scores = {
        "citation_validity": 0.0,
        "citation_relevance": 1.5,
        "citation_quality": 1.2,
    }
    metric_results = {
        "citation_relevance": {"score": 1.5},
        "citation_quality": {"score": 1.2},
    }

    evidence = apply_evidence_integrity(metric_scores, metric_results)

    assert evidence["score"] == 0.0
    assert metric_scores["citation_relevance"] == 0.0
    assert metric_results["citation_quality"]["score"] == 0.0


def test_apply_evidence_integrity_scales_with_rag() -> None:
    metric_scores = {
        "citation_validity": 1.0,
        "rag_fidelity": 0.6,
    }
    metric_results = {}

    evidence = apply_evidence_integrity(metric_scores, metric_results)

    assert evidence["score"] == 0.6
    assert evidence["details"]["validity"] == 1.0
    assert evidence["details"]["rag_fidelity"] == 0.6
